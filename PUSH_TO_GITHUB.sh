#!/usr/bin/env bash
set -euo pipefail
REPO_URL="https://github.com/daisondarlan/icra-attn-ekf-supplement.git"

echo "Preparing anonymous supplementary repository..."
git init -q
git config user.name "Anonymous Authors"
git config user.email "anonymous@example.invalid"
git add index.html .nojekyll .gitignore code data media REPOSITORY_NOTES.txt 2>/dev/null || git add .
git commit -m "Add anonymous supplementary material" --allow-empty

git branch -M main
if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin "$REPO_URL"
else
  git remote add origin "$REPO_URL"
fi

echo "Pushing to GitHub. Git may ask you to authenticate in the browser or with your configured credential helper."
git push -u origin main --force

echo
echo "Done: https://github.com/daisondarlan/icra-attn-ekf-supplement"
