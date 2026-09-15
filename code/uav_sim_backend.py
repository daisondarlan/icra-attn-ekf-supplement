!pip install paho-mqtt==1.6.1 -q
# ============================================================================
# AUTONOMOUS UAV DUAL-ARENA SIMULATION — COLAB BACKEND (T4 GPU)
# ----------------------------------------------------------------------------
# Paste this entire cell into Google Colab (Runtime > Change runtime type >
# T4 GPU) and run it in a single cell. The line above is a Colab/IPython
# "magic" command and only works inside a notebook cell — that's expected.
#
# This process:
#   1. Connects to the public MQTT broker broker.hivemq.com (TCP, port 1883).
#   2. Waits for the HTML dashboard to say "hello" on a session channel.
#   3. Runs two independent physics arenas at 100 Hz using RK4 integration:
#        Arena 1 — 11 drones, identical estimator, 11 different controllers.
#        Arena 2 — 5 drones,  identical controller, 5 different estimators.
#   4. Streams throttled telemetry to the dashboard over MQTT for rendering.
#   5. On a "terminate" command, writes two publication-grade CSV logs,
#      zips them, and downloads the zip via google.colab.files.download().
#
# No plotting of any kind is performed here — this is a headless physics /
# inference engine. All visualization happens in the HTML5 dashboard.
# ============================================================================

import time
import json
import uuid
import zipfile
import shutil
import threading
import csv
import os
import math

import numpy as np
import torch
import torch.nn as nn
import paho.mqtt.client as mqtt

# ----------------------------------------------------------------------------
# 0. DEVICE SETUP
# ----------------------------------------------------------------------------
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"[device] CUDA available -> using {torch.cuda.get_device_name(0)}")
else:
    DEVICE = torch.device("cpu")
    print("[device] CUDA not available -> falling back to CPU. "
          "(Runtime > Change runtime type > T4 GPU to enable acceleration.)")

torch.manual_seed(7)
np.random.seed(7)

# ----------------------------------------------------------------------------
# 1. PHYSICAL CONSTANTS & SHARED HELPERS
# ----------------------------------------------------------------------------
G = 9.81           # gravity, m/s^2
MASS = 1.5         # kg, identical airframe mass for every drone (fair comparison)
DRAG_C = 0.15      # quadratic drag coefficient
DT = 0.01          # physics + logging step, 100 Hz
CYCLE_LEN = 70.0    # seconds; obstacle + setpoint schedule repeats on this period
MAX_DURATION_SEC = 900.0  # safety auto-terminate if kill switch is never pressed

TELEMETRY_EVERY_N_STEPS = 5   # publish MQTT telemetry at 100/5 = 20 Hz
CSV_FLUSH_EVERY_N_ROWS = 500  # batch CSV writes


def clip(value, lo, hi):
    return lo if value < lo else (hi if value > hi else value)


def softmax_weights(residuals, beta):
    """Numerically-stable softmax over -beta * |residual|."""
    r = np.asarray(residuals, dtype=np.float64)
    logits = -beta * r
    logits = logits - np.max(logits)
    w = np.exp(logits)
    total = np.sum(w)
    if total <= 0 or not np.isfinite(total):
        return np.full_like(w, 1.0 / len(w))
    return w / total


def target_altitude(t):
    """Piecewise setpoint trajectory so controllers have continuous work."""
    setpoints = [3.0, 8.0, 5.0, 12.0, 2.0]
    seg_len = CYCLE_LEN / len(setpoints)
    idx = int((t % CYCLE_LEN) // seg_len) % len(setpoints)
    return setpoints[idx]


def obstacle_state(t):
    """Scripted, recurring 'sudden and aggressive' environmental anomalies."""
    tm = t % CYCLE_LEN
    state = {"ceiling": None, "microburst_accel": 0.0, "lidar_fault": False}
    # Sudden moving altitude ceiling blocks the flight path for a window.
    if 15.0 <= tm < 25.0:
        # Ceiling itself drifts downward slightly during the window ("moving").
        state["ceiling"] = 4.0 - 0.05 * (tm - 15.0)
    # Sudden aggressive microburst (strong transient downdraft).
    if 35.0 <= tm < 38.0:
        state["microburst_accel"] = -14.0
    # Instant ~100% lidar signal reflection fault (confidently-wrong sensor).
    if 50.0 <= tm < 55.0:
        state["lidar_fault"] = True
    return state


# ----------------------------------------------------------------------------
# 2. PYTORCH ENVIRONMENT GENERATOR (GAN-STYLE GENERATOR NET, RUNS ON GPU)
# ----------------------------------------------------------------------------
class EnvGenerator(nn.Module):
    """Small generator network mapping a slowly-drifting latent code + time
    features to continuous environmental disturbances. Runs on `cuda` when
    available. The latent code follows a discrete Ornstein-Uhlenbeck-style
    random walk so outputs are temporally smooth rather than pure hash noise,
    which is what makes the generated wind/fog/etc. feel physically continuous
    frame-to-frame instead of flickering randomly."""

    def __init__(self, latent_dim=8):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 2, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 4),
            nn.Tanh(),
        )

    def forward(self, z, t_features):
        x = torch.cat([z, t_features], dim=-1)
        return self.net(x)


class EnvironmentEngine:
    def __init__(self, device):
        self.device = device
        self.gen = EnvGenerator().to(device).eval()
        self.z_latent = torch.randn(1, self.gen.latent_dim, device=device)

    @torch.no_grad()
    def step(self, t):
        # Slowly-drifting latent code (discrete OU process) for temporal smoothness.
        noise = torch.randn_like(self.z_latent)
        self.z_latent = 0.985 * self.z_latent + 0.03 * noise
        t_features = torch.tensor(
            [[math.sin(0.15 * t), math.cos(0.05 * t)]],
            dtype=torch.float32, device=self.device
        )
        out = self.gen(self.z_latent, t_features).squeeze(0).cpu().numpy()
        wind_velocity = float(out[0]) * 3.0            # m/s^2 disturbance, +/-3
        fog_density = float((out[1] + 1.0) / 2.0)       # in [0,1]
        lidar_scattering = float((out[2] + 1.0) / 2.0)   # in [0,1]
        baro_shift = float(out[3]) * 0.4                 # meters of slow bias
        return {
            "wind": wind_velocity,
            "fog": fog_density,
            "lidar_scatter": lidar_scattering,
            "baro_shift": baro_shift,
        }


# ----------------------------------------------------------------------------
# 3. PHYSICS: RK4 INTEGRATION OF 1D VERTICAL DYNAMICS
# ----------------------------------------------------------------------------
def state_derivative(x, u, w_wind):
    z, v = x
    dz = v
    dv = (1.0 / MASS) * (u - MASS * G - DRAG_C * v * abs(v)) + w_wind
    return np.array([dz, dv])


def rk4_step(x, u, w_wind, dt):
    k1 = state_derivative(x, u, w_wind)
    k2 = state_derivative(x + 0.5 * dt * k1, u, w_wind)
    k3 = state_derivative(x + 0.5 * dt * k2, u, w_wind)
    k4 = state_derivative(x + dt * k3, u, w_wind)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def apply_ceiling(x, ceiling):
    if ceiling is not None and x[0] > ceiling:
        x = np.array([ceiling, min(x[1], 0.0)])
    return x


# ----------------------------------------------------------------------------
# 4. CONTROLLERS
# ----------------------------------------------------------------------------
class PID:
    def __init__(self, kp, ki, kd, integral_limit=8.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.integral = 0.0
        self.prev_error = None
        self.integral_limit = integral_limit

    def compute(self, error, dt, kp_override=None):
        kp = kp_override if kp_override is not None else self.kp
        self.integral = clip(self.integral + error * dt,
                              -self.integral_limit, self.integral_limit)
        derivative = 0.0 if self.prev_error is None else (error - self.prev_error) / dt
        self.prev_error = error
        u_p = kp * error
        u_i = self.ki * self.integral
        u_d = self.kd * derivative
        return u_p + u_i + u_d + MASS * G, u_p, u_i, u_d  # + gravity feedforward


PID_CONFIGS = [
    {"name": "Conservative",      "kp": 3.0, "ki": 0.15, "kd": 1.2},
    {"name": "Aggressive",        "kp": 9.0, "ki": 0.60, "kd": 1.8},
    {"name": "Ziegler-Nichols",   "kp": 6.4, "ki": 1.28, "kd": 0.8},
    {"name": "Sluggish",          "kp": 1.2, "ki": 0.05, "kd": 0.4},
    {"name": "Over-damped",       "kp": 2.5, "ki": 0.05, "kd": 3.2},
    {"name": "Under-damped",      "kp": 5.5, "ki": 0.05, "kd": 0.15},
    {"name": "Integral-heavy",    "kp": 2.0, "ki": 1.60, "kd": 0.5},
    {"name": "Derivative-heavy",  "kp": 3.2, "ki": 0.10, "kd": 4.0},
    {"name": "Balanced",          "kp": 4.2, "ki": 0.35, "kd": 1.5},
    {"name": "Critically-damped", "kp": 4.5, "ki": 0.20, "kd": 2.68},
]
ATTENTION_PID_BASE = {"kp": 4.2, "ki": 0.35, "kd": 1.5}  # Drone 11 base gains
ATTENTION_BETA = 6.0  # softmax sharpness for confidence gating / fault isolation

# ----------------------------------------------------------------------------
# 5. ESTIMATORS
# ----------------------------------------------------------------------------
def scalar_update(x, P, meas, R):
    """Kalman scalar measurement update for H = [1, 0] (both sensors read z)."""
    innov = meas - x[0]
    S = P[0, 0] + R
    K = P[:, 0] / S
    x_new = x + K * innov
    P_new = P - np.outer(K, P[0, :])
    return x_new, P_new, innov


class EKF2D:
    """Extended Kalman Filter over a mildly nonlinear (quadratic-drag) model."""

    def __init__(self, r_lidar, r_baro, q=(1e-4, 2e-3)):
        self.x = np.array([0.0, 0.0])
        self.P = np.diag([1.0, 1.0])
        self.Q = np.diag(q)
        self.r_lidar_base = r_lidar
        self.r_baro_base = r_baro

    def predict(self, u, dt):
        z, v = self.x
        dv = (1.0 / MASS) * (u - MASS * G - DRAG_C * v * abs(v))
        x_pred = np.array([z + dt * v, v + dt * dv])
        f22 = 1.0 - dt * (2.0 * DRAG_C * abs(v) / MASS)
        F = np.array([[1.0, dt], [0.0, f22]])
        P_pred = F @ self.P @ F.T + self.Q
        self.x_pred, self.P_pred = x_pred, P_pred
        return x_pred

    def update(self, lidar_meas, baro_meas, attention=False):
        x, P = self.x_pred.copy(), self.P_pred.copy()
        pred_z = x[0]
        r_lidar_raw = abs(lidar_meas - pred_z)
        r_baro_raw = abs(baro_meas - pred_z)

        r_lidar_eff, r_baro_eff = self.r_lidar_base, self.r_baro_base
        w_lidar = w_baro = None
        if attention:
            w = softmax_weights([r_lidar_raw, r_baro_raw], ATTENTION_BETA)
            w_lidar, w_baro = float(w[0]), float(w[1])
            eps = 1e-3
            r_lidar_eff = self.r_lidar_base / (w_lidar + eps)
            r_baro_eff = self.r_baro_base / (w_baro + eps)

        x, P, innov_lidar = scalar_update(x, P, lidar_meas, r_lidar_eff)
        x, P, innov_baro = scalar_update(x, P, baro_meas, r_baro_eff)
        self.x, self.P = x, P
        return {
            "x": x, "P": P,
            "innov_lidar": innov_lidar, "innov_baro": innov_baro,
            "w_lidar": w_lidar, "w_baro": w_baro,
            "confidence": max(w_lidar, w_baro) if attention else None,
        }


class UKF2D:
    """Unscented Kalman Filter over the same nonlinear process model, using
    the standard scaled unscented transform (no Jacobians required)."""

    def __init__(self, r_lidar, r_baro, q=(1e-4, 2e-3), alpha=1e-3, beta_ut=2.0, kappa=0.0):
        self.n = 2
        self.x = np.array([0.0, 0.0])
        self.P = np.diag([1.0, 1.0])
        self.Q = np.diag(q)
        self.r_lidar = r_lidar
        self.r_baro = r_baro
        self.alpha, self.beta_ut, self.kappa = alpha, beta_ut, kappa
        self.lam = (alpha ** 2) * (self.n + kappa) - self.n
        n, lam = self.n, self.lam
        self.wm = np.full(2 * n + 1, 1.0 / (2 * (n + lam)))
        self.wc = self.wm.copy()
        self.wm[0] = lam / (n + lam)
        self.wc[0] = lam / (n + lam) + (1 - alpha ** 2 + beta_ut)

    def _sigma_points(self, x, P):
        n, lam = self.n, self.lam
        P_reg = P + 1e-9 * np.eye(n)
        try:
            L = np.linalg.cholesky((n + lam) * P_reg)
        except np.linalg.LinAlgError:
            L = np.linalg.cholesky((n + lam) * np.eye(n) * 1e-6)
        pts = [x]
        for i in range(n):
            pts.append(x + L[:, i])
            pts.append(x - L[:, i])
        return np.array(pts)

    def _process(self, x, u):
        z, v = x
        dv = (1.0 / MASS) * (u - MASS * G - DRAG_C * v * abs(v))
        return np.array([z + DT * v, v + DT * dv])

    def predict(self, u, dt):
        sigmas = self._sigma_points(self.x, self.P)
        sigmas_pred = np.array([self._process(s, u) for s in sigmas])
        x_pred = np.sum(self.wm[:, None] * sigmas_pred, axis=0)
        P_pred = self.Q.copy()
        for i in range(sigmas_pred.shape[0]):
            d = (sigmas_pred[i] - x_pred).reshape(-1, 1)
            P_pred += self.wc[i] * (d @ d.T)
        self.x_pred, self.P_pred, self._sigmas_pred = x_pred, P_pred, sigmas_pred
        return x_pred

    def _scalar_ut_update(self, x, P, sigmas, meas, R):
        z_sigma = sigmas[:, 0]
        z_pred = np.sum(self.wm * z_sigma)
        Pzz = R
        Pxz = np.zeros(self.n)
        for i in range(sigmas.shape[0]):
            dz = z_sigma[i] - z_pred
            dx = sigmas[i] - x
            Pzz += self.wc[i] * dz * dz
            Pxz += self.wc[i] * dx * dz
        K = Pxz / Pzz
        innov = meas - z_pred
        x_new = x + K * innov
        P_new = P - np.outer(K, K) * Pzz
        return x_new, P_new, innov

    def update(self, lidar_meas, baro_meas):
        x, P = self.x_pred, self.P_pred
        sigmas = self._sigmas_pred
        x, P, innov_lidar = self._scalar_ut_update(x, P, sigmas, lidar_meas, self.r_lidar)
        sigmas = self._sigma_points(x, P)  # redraw before second sequential update
        x, P, innov_baro = self._scalar_ut_update(x, P, sigmas, baro_meas, self.r_baro)
        self.x, self.P = x, P
        return {"x": x, "P": P, "innov_lidar": innov_lidar, "innov_baro": innov_baro}


EST_CONFIGS = [
    {"name": "Standard EKF (Tuned R)", "type": "ekf", "r_lidar": 0.05, "r_baro": 0.15},
    {"name": "EKF (Trust Lidar)",      "type": "ekf", "r_lidar": 0.001, "r_baro": 2.0},
    {"name": "EKF (Trust Baro)",       "type": "ekf", "r_lidar": 2.0, "r_baro": 0.01},
    {"name": "UKF",                    "type": "ukf", "r_lidar": 0.05, "r_baro": 0.15},
    {"name": "Attention-Gated EKF",    "type": "ekf_attn", "r_lidar": 0.05, "r_baro": 0.15},
]

# ----------------------------------------------------------------------------
# 6. SENSOR MODEL
# ----------------------------------------------------------------------------
LIDAR_BASE_SIGMA = 0.03
BARO_BASE_SIGMA = 0.12


def sense(z_true, env, obstacle):
    if obstacle["lidar_fault"]:
        # Confidently-wrong sensor: near-zero noise but badly biased reading.
        lidar_meas = z_true + 2.5
    else:
        sigma_lidar = LIDAR_BASE_SIGMA * (1 + 5 * env["fog"]) * (1 + 8 * env["lidar_scatter"])
        lidar_meas = z_true + np.random.normal(0, sigma_lidar)
    baro_meas = z_true + env["baro_shift"] + np.random.normal(0, BARO_BASE_SIGMA)
    return lidar_meas, baro_meas


# ----------------------------------------------------------------------------
# 7. ARENA STATE CONTAINERS
# ----------------------------------------------------------------------------
class Arena1Drone:
    def __init__(self, cfg, idx):
        self.cfg = cfg
        self.idx = idx
        self.x = np.array([0.0, 0.0])
        self.pid = PID(cfg["kp"], cfg["ki"], cfg["kd"])
        self.ekf = EKF2D(r_lidar=0.05, r_baro=0.15)  # "the same standard EKF" for all 11
        self.is_attention = (idx == 11)
        self.last_u = MASS * G  # hover thrust, sensible initial condition


class Arena2Drone:
    def __init__(self, cfg, idx):
        self.cfg = cfg
        self.idx = idx
        self.x = np.array([0.0, 0.0])
        self.pid = PID(**{k: v for k, v in ATTENTION_PID_BASE.items()})  # same standard PID for all 5
        if cfg["type"] == "ukf":
            self.est = UKF2D(cfg["r_lidar"], cfg["r_baro"])
        else:
            self.est = EKF2D(cfg["r_lidar"], cfg["r_baro"])
        self.is_attention = (cfg["type"] == "ekf_attn")
        self.last_u = MASS * G  # hover thrust, sensible initial condition


arena1_drones = [Arena1Drone(cfg, i + 1) for i, cfg in enumerate(PID_CONFIGS)]
arena1_drones.append(Arena1Drone({"name": "Attention-Gated PID", **ATTENTION_PID_BASE}, 11))

arena2_drones = [Arena2Drone(cfg, i + 1) for i, cfg in enumerate(EST_CONFIGS)]

env_engine = EnvironmentEngine(DEVICE)

# ----------------------------------------------------------------------------
# 8. CSV LOGGING
# ----------------------------------------------------------------------------
PID_LOG_PATH = "telemetry_pid.csv"
EST_LOG_PATH = "telemetry_est.csv"


def pid_header():
    cols = ["t", "target_altitude", "wind", "fog", "lidar_scatter", "baro_shift",
            "obstacle_ceiling", "obstacle_microburst", "obstacle_lidar_fault"]
    for d in arena1_drones:
        p = f"d{d.idx}_{d.cfg['name'].replace(' ', '_')}"
        cols += [f"{p}_z", f"{p}_v", f"{p}_a", f"{p}_u", f"{p}_up", f"{p}_ui", f"{p}_ud"]
    cols.append("d11_confidence_C")
    return cols


def est_header():
    cols = ["t", "target_altitude", "wind", "fog", "lidar_scatter", "baro_shift",
            "obstacle_ceiling", "obstacle_microburst", "obstacle_lidar_fault"]
    for d in arena2_drones:
        p = f"d{d.idx}_{d.cfg['name'].replace(' ', '_').replace('(', '').replace(')', '')}"
        cols += [f"{p}_z_true", f"{p}_v_true", f"{p}_z_est", f"{p}_v_est",
                 f"{p}_lidar_raw", f"{p}_baro_raw", f"{p}_P11", f"{p}_P22"]
    cols += ["d5_w_lidar", "d5_w_baro", "d5_innov_lidar", "d5_innov_baro"]
    return cols


class CsvBuffer:
    def __init__(self, path, header):
        self.path = path
        self.header = header
        self.rows = []
        self._file = open(path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(header)

    def add(self, row):
        self.rows.append(row)
        if len(self.rows) >= CSV_FLUSH_EVERY_N_ROWS:
            self.flush()

    def flush(self):
        if self.rows:
            self._writer.writerows(self.rows)
            self.rows = []
            self._file.flush()

    def close(self):
        self.flush()
        self._file.close()


pid_csv = CsvBuffer(PID_LOG_PATH, pid_header())
est_csv = CsvBuffer(EST_LOG_PATH, est_header())

# ----------------------------------------------------------------------------
# 9. MQTT — SESSION HANDSHAKE, COMMANDS, TELEMETRY
# ----------------------------------------------------------------------------
BROKER_HOST = "broker.hivemq.com"
BROKER_PORT = 1883
TOPIC_PREFIX = "uavsim"

sim_lock = threading.Lock()
sim_state = {
    "session": None,
    "running": False,
    "terminate": False,
}

client_id = f"uav-backend-{uuid.uuid4().hex[:8]}"
mqtt_client = mqtt.Client(client_id=client_id, protocol=mqtt.MQTTv311)


def on_connect(client, userdata, flags, rc):
    print(f"[mqtt] connected to {BROKER_HOST} (rc={rc})")
    client.subscribe(f"{TOPIC_PREFIX}/+/hello")
    client.subscribe(f"{TOPIC_PREFIX}/+/cmd")
    print("[mqtt] waiting for dashboard 'hello' on "
          f"{TOPIC_PREFIX}/<session_id>/hello ...")


def on_message(client, userdata, msg):
    parts = msg.topic.split("/")
    if len(parts) != 3:
        return
    _, session_id, kind = parts
    with sim_lock:
        if kind == "hello":
            if sim_state["session"] is None:
                sim_state["session"] = session_id
                sim_state["running"] = True
                print(f"[session] locked onto session '{session_id}', starting simulation")
                client.publish(f"{TOPIC_PREFIX}/{session_id}/status",
                                json.dumps({"status": "running", "session": session_id}),
                                qos=0, retain=True)
            elif session_id != sim_state["session"]:
                print(f"[session] ignoring hello from '{session_id}', "
                      f"already bound to '{sim_state['session']}'")
        elif kind == "cmd" and session_id == sim_state["session"]:
            try:
                payload = json.loads(msg.payload.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                return
            if payload.get("cmd") == "terminate":
                print("[cmd] terminate received from dashboard")
                sim_state["terminate"] = True


mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message
mqtt_client.connect(BROKER_HOST, BROKER_PORT, keepalive=30)
mqtt_client.loop_start()


def publish_telemetry(topic_kind, payload):
    session = sim_state["session"]
    if session is None:
        return
    try:
        mqtt_client.publish(f"{TOPIC_PREFIX}/{session}/telemetry/{topic_kind}",
                             json.dumps(payload), qos=0)
    except Exception as exc:  # never let a flaky broker connection kill the physics loop
        print(f"[mqtt] publish failed ({topic_kind}): {exc}")


# ----------------------------------------------------------------------------
# 10. CSV EXPORT + DOWNLOAD
# ----------------------------------------------------------------------------
def export_and_download():
    pid_csv.close()
    est_csv.close()
    zip_path = "uav_sim_telemetry.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(PID_LOG_PATH)
        zf.write(EST_LOG_PATH)
    print(f"[export] wrote {zip_path} "
          f"({os.path.getsize(zip_path)} bytes) containing {PID_LOG_PATH} and {EST_LOG_PATH}")
    try:
        from google.colab import files as colab_files
        colab_files.download(zip_path)
        print("[export] triggered browser download via google.colab.files")
    except ImportError:
        print("[export] not running in Colab — zip left on disk at "
              f"{os.path.abspath(zip_path)}")


# ----------------------------------------------------------------------------
# 11. MAIN SIMULATION LOOP (100 Hz, real-time paced)
# ----------------------------------------------------------------------------
def wait_for_session(poll_interval=0.5, timeout=None):
    start = time.time()
    while sim_state["session"] is None:
        time.sleep(poll_interval)
        if timeout is not None and (time.time() - start) > timeout:
            return False
    return True


def run_simulation():
    print("[sim] waiting for a dashboard connection before starting the physics loop...")
    if not wait_for_session(timeout=None):
        return

    t = 0.0
    step = 0
    next_tick = time.perf_counter()
    exported = False

    try:
        while True:
            if sim_state["terminate"]:
                break
            if (t > MAX_DURATION_SEC):
                print(f"[sim] reached MAX_DURATION_SEC={MAX_DURATION_SEC}s safety limit, auto-terminating")
                break

            target = target_altitude(t)
            obstacle = obstacle_state(t)
            env = env_engine.step(t)
            wind = env["wind"] + obstacle["microburst_accel"]

            # ---------------- Arena 1: controller comparison ----------------
            pid_row = [round(t, 3), target, env["wind"], env["fog"],
                       env["lidar_scatter"], env["baro_shift"],
                       obstacle["ceiling"] if obstacle["ceiling"] is not None else "",
                       int(obstacle["microburst_accel"] != 0.0),
                       int(obstacle["lidar_fault"])]
            confidence_c = None
            for d in arena1_drones:
                lidar_meas, baro_meas = sense(d.x[0], env, obstacle)
                d.ekf.predict(d.last_u, DT)  # propagate with the control actually applied last step
                result = d.ekf.update(lidar_meas, baro_meas, attention=d.is_attention)
                z_est = result["x"][0]
                error = target - z_est
                kp_override = None
                if d.is_attention:
                    confidence_c = result["confidence"]
                    kp_override = confidence_c * d.pid.kp
                u, up, ui, ud = d.pid.compute(error, DT, kp_override=kp_override)
                d.last_u = u
                d.x = rk4_step(d.x, u, wind, DT)
                d.x = apply_ceiling(d.x, obstacle["ceiling"])
                a_true = (1.0 / MASS) * (u - MASS * G - DRAG_C * d.x[1] * abs(d.x[1])) + wind
                pid_row += [d.x[0], d.x[1], a_true, u, up, ui, ud]
            pid_row.append(confidence_c if confidence_c is not None else "")
            pid_csv.add(pid_row)

            # ---------------- Arena 2: estimator comparison ----------------
            est_row = [round(t, 3), target, env["wind"], env["fog"],
                       env["lidar_scatter"], env["baro_shift"],
                       obstacle["ceiling"] if obstacle["ceiling"] is not None else "",
                       int(obstacle["microburst_accel"] != 0.0),
                       int(obstacle["lidar_fault"])]
            attn_extra = ("", "", "", "")
            for d in arena2_drones:
                lidar_meas, baro_meas = sense(d.x[0], env, obstacle)
                d.est.predict(d.last_u, DT)
                if isinstance(d.est, UKF2D):
                    r = d.est.update(lidar_meas, baro_meas)
                    w_lidar = w_baro = None
                else:
                    r = d.est.update(lidar_meas, baro_meas, attention=d.is_attention)
                    w_lidar, w_baro = r.get("w_lidar"), r.get("w_baro")
                z_est, v_est = r["x"][0], r["x"][1]
                error = target - z_est
                u, _, _, _ = d.pid.compute(error, DT)
                d.last_u = u
                d.x = rk4_step(d.x, u, wind, DT)
                d.x = apply_ceiling(d.x, obstacle["ceiling"])
                P11, P22 = r["P"][0, 0], r["P"][1, 1]
                est_row += [d.x[0], d.x[1], z_est, v_est, lidar_meas, baro_meas, P11, P22]
                if d.is_attention:
                    attn_extra = (w_lidar, w_baro, r["innov_lidar"], r["innov_baro"])
            est_row += list(attn_extra)
            est_csv.add(est_row)

            # ---------------- Throttled MQTT telemetry ----------------
            if step % TELEMETRY_EVERY_N_STEPS == 0:
                env_obstacle_payload = {
                    "env": env,
                    "obstacle": {
                        "ceiling": obstacle["ceiling"],
                        "microburst": obstacle["microburst_accel"] != 0.0,
                        "lidar_fault": obstacle["lidar_fault"],
                    },
                }
                pid_payload = {"t": round(t, 2), "target": target, **env_obstacle_payload,
                                "drones": [
                                    {"id": d.idx, "name": d.cfg["name"], "z": round(d.x[0], 4),
                                     "v": round(d.x[1], 4)}
                                    for d in arena1_drones
                                ]}
                if confidence_c is not None:
                    pid_payload["confidence_C"] = round(float(confidence_c), 4)
                publish_telemetry("pid", pid_payload)

                est_payload = {"t": round(t, 2), "target": target, **env_obstacle_payload,
                                "drones": [
                                    {"id": d.idx, "name": d.cfg["name"], "z": round(d.x[0], 4),
                                     "z_est": round(float(d.est.x[0]), 4),
                                     "P11": round(float(d.est.P[0, 0]), 5)}
                                    for d in arena2_drones
                                ]}
                publish_telemetry("est", est_payload)

            t += DT
            step += 1
            next_tick += DT
            sleep_for = next_tick - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_tick = time.perf_counter()  # we fell behind; resync pacing

    finally:
        if not exported:
            export_and_download()
            exported = True
        if sim_state["session"] is not None:
            mqtt_client.publish(f"{TOPIC_PREFIX}/{sim_state['session']}/status",
                                 json.dumps({"status": "terminated"}), qos=0, retain=True)
        print("[sim] simulation loop ended.")


run_simulation()
