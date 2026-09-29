#!/data/data/com.termux/files/usr/bin/bash
# Compile, installe et lance KabyleAI.
# Usage : ./build_install.sh [dossier_matoub [dossier_nllb]]
#   dossier_matoub : matoub_front.onnx et matoub_back.onnx (produits par
#                    export_matoub_onnx.py), copiés dans l'application.
#   dossier_nllb   : fichiers nllb_* produits par export_nllb_onnx.py
#                    (traduction français -> kabyle).
#   Avec SKIP_BUILD=1, l'application n'est pas recompilée : seuls les
#   modèles sont copiés.

set -euo pipefail

PROJECT="$HOME/projects/KabyleAI"
APK="$PROJECT/app/build/outputs/apk/debug/app-debug.apk"
PKG="com.kabyleai.app"
APP_DIR="/data/data/$PKG/files/matoub"
MODEL="${1:-}"
NLLB_DIR_SRC="${2:-}"
NLLB_APP_DIR="/data/data/$PKG/files/nllb"

cd "$PROJECT"

if [ "${SKIP_BUILD:-0}" != "1" ]; then
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
else
    echo "[1-2/5] Compilation et installation ignorées (SKIP_BUILD=1)."
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

echo "[3b/5] Traduction NLLB..."
if [ -n "$NLLB_DIR_SRC" ]; then
    for f in nllb_config.txt nllb_vocab.tsv nllb_embed.i8 nllb_embed.scales \
             nllb_encoder.onnx nllb_decoder.onnx nllb_head.onnx; do
        if [ ! -f "$NLLB_DIR_SRC/$f" ]; then
            echo "Fichier manquant : $NLLB_DIR_SRC/$f"
            exit 1
        fi
    done

    su -c "mkdir -p '$NLLB_APP_DIR'"
    for f in "$NLLB_DIR_SRC"/nllb_*; do
        su -c "cp '$f' '$NLLB_APP_DIR/'"
    done

    APP_UID=$(su -c "stat -c %u /data/data/$PKG")
    su -c "chown -R $APP_UID:$APP_UID '$NLLB_APP_DIR'"
    su -c "restorecon -R '$NLLB_APP_DIR'" 2>/dev/null || true
    su -c "ls -lh '$NLLB_APP_DIR'"
else
    echo "Aucun dossier NLLB fourni : modèle actuel conservé."
fi

echo "[4/5] Lancement..."
su -c "am force-stop $PKG"
su -c "logcat -c"
su -c "am start -n $PKG/.MainActivity" >/dev/null

echo "[5/5] Journal (Ctrl+C pour quitter)..."
echo "Lancez une synthèse dans l'application."
su -c "logcat -v brief" | grep --line-buffered -iE "matoub|onnx|AndroidRuntime|kabyleai"
