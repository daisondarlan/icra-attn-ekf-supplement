from pathlib import Path

p = Path('index.html')
s = p.read_text()

old = '''  // Uses repository-local artifacts so the anonymous site does not depend on Google Drive.\n  document.addEventListener("DOMContentLoaded", () => {\n    const video = document.getElementById("field-video");\n    if (video) video.src = LOCAL_ARTIFACTS.videoFile;\n    const est = document.getElementById("drive-est-log");\n    if (est) est.href = LOCAL_ARTIFACTS.estimatorLog;\n    const pid = document.getElementById("drive-pid-log");\n    if (pid) pid.href = LOCAL_ARTIFACTS.controllerLog;\n  });\n'''
new = '''  function anonymousRepoId(){\n    if (location.hostname !== "anonymous.4open.science") return null;\n    const m = location.pathname.match(/^\\/w\\/([^/]+)/);\n    return m ? m[1] : null;\n  }\n\n  function artifactDownloadUrl(path){\n    const repoId = anonymousRepoId();\n    if (repoId) {\n      return "/api/repo/" + encodeURIComponent(repoId) + "/file/" +\n             path.split("/").map(encodeURIComponent).join("/") + "?download=true";\n    }\n    return path;\n  }\n\n  function downloadRepositoryFile(path, filename){\n    const a = document.createElement("a");\n    a.href = artifactDownloadUrl(path);\n    a.target = "_self";\n    if (!anonymousRepoId()) a.download = filename || path.split("/").pop();\n    document.body.appendChild(a);\n    a.click();\n    a.remove();\n  }\n\n  // Uses repository-local artifacts so the anonymous site does not depend on Google Drive.\n  document.addEventListener("DOMContentLoaded", () => {\n    const video = document.getElementById("field-video");\n    if (video) video.src = LOCAL_ARTIFACTS.videoFile;\n    const est = document.getElementById("drive-est-log");\n    if (est) {\n      est.href = artifactDownloadUrl(LOCAL_ARTIFACTS.estimatorLog);\n      est.target = "_self";\n      if (!anonymousRepoId()) est.setAttribute("download", "telemetry_est.csv");\n    }\n    const pid = document.getElementById("drive-pid-log");\n    if (pid) {\n      pid.href = artifactDownloadUrl(LOCAL_ARTIFACTS.controllerLog);\n      pid.target = "_self";\n      if (!anonymousRepoId()) pid.setAttribute("download", "telemetry_pid.csv");\n    }\n  });\n'''
if old not in s:
    raise SystemExit('artifact block not found')
s = s.replace(old, new)

old = '''  function triggerDownload(filename, content, mime){\n    try{\n      var blob = new Blob([content], { type: mime || "text/plain" });\n      var url = URL.createObjectURL(blob);\n      var a = document.createElement("a");\n      a.href = url; a.download = filename;\n      document.body.appendChild(a); a.click(); document.body.removeChild(a);\n      setTimeout(function(){ URL.revokeObjectURL(url); }, 4000);\n    }catch(e){ console.error("download failed", e); }\n  }\n\n  window.uavSim = {\n    downloadBackend: function(){ triggerDownload("uav_sim_backend.py", window.__SRC_BACKEND__ || "", "text/x-python"); },\n    downloadDashboard: function(){ triggerDownload("uav_sim_dashboard.html", window.__SRC_DASHBOARD__ || "", "text/html"); }\n  };\n'''
new = '''  window.uavSim = {\n    downloadBackend: function(){\n      downloadRepositoryFile("code/uav_sim_backend.py", "uav_sim_backend.py");\n    },\n    downloadDashboard: function(){\n      downloadRepositoryFile("code/uav_sim_dashboard.html", "uav_sim_dashboard.html");\n    }\n  };\n'''
if old not in s:
    raise SystemExit('download block not found')
s = s.replace(old, new)

old = '''    if(newTabBtn && window.__SRC_DASHBOARD__) {\n      newTabBtn.onclick = function() {\n        var blob = new Blob([window.__SRC_DASHBOARD__], { type: "text/html" });\n        window.open(URL.createObjectURL(blob), "_blank");\n      };\n    }\n'''
new = '''    if(newTabBtn) {\n      newTabBtn.onclick = function() {\n        window.open("code/uav_sim_dashboard.html", "_blank", "noopener");\n      };\n    }\n'''
if old not in s:
    raise SystemExit('dashboard block not found')
s = s.replace(old, new)

p.write_text(s)
print('patched index.html')
