#!/data/data/com.termux/files/usr/bin/bash
# Compile, installe et lance KabyleAI.
# Usage : ./build_install.sh [dossier_modele]
#   Le dossier contient matoub_front.onnx et matoub_back.onnx (produits
#   par export_matoub_onnx.py) ; ils sont copiés dans l'application.

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
    if [ ! -d "$MODEL" ]; then
        echo "Il faut un dossier contenant matoub_front.onnx et matoub_back.onnx : $MODEL"
        exit 1
    fi

    for f in matoub_front.onnx matoub_back.onnx; do
        if [ ! -f "$MODEL/$f" ]; then
            echo "Fichier manquant : $MODEL/$f"
            exit 1
        fi
    done

    su -c "mkdir -p '$APP_DIR'"
    su -c "rm -f '$APP_DIR/matoub_82m.onnx' '$APP_DIR/matoub_82m.onnx.data'"

    for f in "$MODEL"/matoub_front.onnx "$MODEL"/matoub_back.onnx \
             "$MODEL"/matoub_front.onnx.data "$MODEL"/matoub_back.onnx.data; do
        if [ -f "$f" ]; then
            su -c "cp '$f' '$APP_DIR/'"
        fi
    done

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
