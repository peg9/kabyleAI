#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail

PROJECT="$HOME/projects/KabyleAI"

cd "$PROJECT" || exit 1

echo "=================================="
echo "   ENVOI KABYLEAI VERS GITHUB"
echo "=================================="

# Vérifier Git
if ! command -v git >/dev/null 2>&1; then
    echo "Installation de Git..."
    pkg install git -y
fi

# Vérifier l'identité Git
if ! git config user.name >/dev/null 2>&1; then
    read -r -p "Nom Git : " GIT_NAME
    git config --global user.name "$GIT_NAME"
fi

if ! git config user.email >/dev/null 2>&1; then
    read -r -p "Email Git : " GIT_EMAIL
    git config --global user.email "$GIT_EMAIL"
fi

# Créer ou réutiliser le dépôt local
if [ ! -d .git ]; then
    git init
    git branch -M main
fi

# Ajouter les exclusions sans écraser le .gitignore existant
touch .gitignore

if ! grep -qF -- '# --- Fichiers locaux et sensibles ---' .gitignore; then
cat >> .gitignore <<'IGNORE'

# --- Fichiers locaux et sensibles ---
.env
.env.*
!.env.example
*.pem
*.key
*.p12
*.pfx
*.keystore
*.jks
local.properties
credentials.json
secrets.json

# --- Android / Gradle ---
.gradle/
build/
**/build/
local/
captures/
.externalNativeBuild/
.cxx/

# --- Python ---
__pycache__/
*.py[cod]
.venv/
venv/
.ipynb_checkpoints/

# --- Modèles volumineux ---
*.safetensors
*.pt
*.pth
*.ckpt
*.bin
*.onnx
*.onnx.data
*.tflite

# --- APK et archives générées ---
*.apk
*.aab
*.zip
*.tar
*.tar.gz

# --- Logs ---
*.log
IGNORE
fi

echo
echo "Fichiers exclus selon .gitignore."
echo "Les sources Kotlin, Python et Gradle restent incluses."
echo

# Demander l'URL du dépôt
read -r -p "URL HTTPS du dépôt GitHub : " REPO_URL

if [[ ! "$REPO_URL" =~ ^https://github\.com/[^/]+/[^/]+(\.git)?$ ]]; then
    echo "ERREUR : URL GitHub HTTPS invalide."
    exit 1
fi

# Configurer le remote
if git remote get-url origin >/dev/null 2>&1; then
    git remote set-url origin "$REPO_URL"
else
    git remote add origin "$REPO_URL"
fi

# Ajouter les sources
git add .

echo
echo "=== FICHIERS À ENVOYER ==="
git status --short

echo
read -r -p "Confirmer le commit et l'envoi ? (oui/non) : " CONFIRM

if [[ "$CONFIRM" != "oui" ]]; then
    echo "Annulé. Aucun push effectué."
    exit 0
fi

# Commit
if ! git diff --cached --quiet; then
    git commit -m "Sauvegarde du projet KabyleAI"
else
    echo "Aucune nouvelle modification à committer."
fi

# Envoyer sur GitHub
git push -u origin HEAD:main

echo
echo "=================================="
echo "ENVOI GITHUB TERMINÉ"
echo "=================================="
echo "$REPO_URL"
