#!/data/data/com.termux/files/usr/bin/bash
# Compile, installe et lance KabyleAI.
# Usage : ./build_install.sh [modele.onnx]
#   Avec un modèle, il est copié dans l'application sous le nom
#   matoub_82m.onnx, avec son fichier .onnx.data s'il existe.

set -euo pipefail

PROJECT="$HOME/projects/KabyleAI"
APK="$PROJECT/app/build/outputs/apk/debug/app-debug.apk"
PKG="com.kabyleai.app"
APP_DIR="/data/data/$PKG/files/matoub"
MODEL="${1:-}"

cd "$PROJECT"

echo "[1/5] Compilation..."
if ! gradle :app:assembleDebug; then
    echo "ECHEC DE LA COMPILATION : copiez l'erreur ci-dessus."
    exit 1
fi
ls -lh "$APK"

echo "[2/5] Installation..."
if su -c true 2>/dev/null; then
    su -c "pm install -r '$APK'"
else
    echo "Pas de root : ouverture de l'installateur Android."
    termux-open "$APK"
    exit 0
fi

echo "[3/5] Modèle..."
if [ -n "$MODEL" ]; then
    if [ ! -f "$MODEL" ]; then
        echo "Modèle introuvable : $MODEL"
        exit 1
    fi

    su -c "mkdir -p '$APP_DIR'"
    su -c "cp '$MODEL' '$APP_DIR/matoub_82m.onnx'"

    # Le .onnx référence son fichier de poids par son nom d'origine :
    # on le copie sans le renommer.
    if [ -f "$MODEL.data" ]; then
        su -c "cp '$MODEL.data' '$APP_DIR/'"
        echo "Poids copiés : $(basename "$MODEL.data")"
    fi

    APP_UID=$(su -c "stat -c %u /data/data/$PKG")
    su -c "chown -R $APP_UID:$APP_UID '$APP_DIR'"
    su -c "restorecon -R '$APP_DIR'" 2>/dev/null || true
    su -c "ls -lh '$APP_DIR'"
else
    echo "Aucun modèle fourni : modèle actuel conservé."
fi

echo "[4/5] Lancement..."
su -c "am force-stop $PKG"
su -c "logcat -c"
su -c "am start -n $PKG/.MainActivity" >/dev/null

echo "[5/5] Journal (Ctrl+C pour quitter)..."
echo "Lancez une synthèse dans l'application."
su -c "logcat -v brief" | grep --line-buffered -iE "matoub|onnx|AndroidRuntime|kabyleai"
