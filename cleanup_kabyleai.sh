#!/data/data/com.termux/files/usr/bin/bash

set -eu

PROJECT="$HOME/projects/KabyleAI"
BACKUP="$HOME/projects/KabyleAI_cleanup_$(date +%Y%m%d_%H%M%S)"

cd "$PROJECT" || exit 1

echo "======================================"
echo " NETTOYAGE KABYLEAI"
echo "======================================"

echo
echo "Projet : $PROJECT"
echo "Sauvegarde : $BACKUP"
echo

# 1. Sauvegarder les scripts de diagnostic et d'export
mkdir -p "$BACKUP/scripts"

for f in \
    export_legacy.py \
    fix_matoub_export.sh \
    fix_matoub_export.py \
    test_matoub_real_stft.sh \
    test_matoub_patched_generator.sh \
    diagnose_matoub_original.py
do
    if [ -f "$f" ]; then
        cp -p "$f" "$BACKUP/scripts/"
        echo "Sauvegardé : $f"
    fi
done

# 2. Sauvegarder les résultats de diagnostic éventuels
if [ -d "$HOME/kabyle-models/matoub_stft_test" ]; then
    cp -a "$HOME/kabyle-models/matoub_stft_test" "$BACKUP/"
fi

if [ -d "$HOME/kabyle-models/matoub_original_diagnostic" ]; then
    cp -a "$HOME/kabyle-models/matoub_original_diagnostic" "$BACKUP/"
fi

# 3. Supprimer uniquement les scripts temporaires identifiés
for f in \
    export_legacy.py \
    fix_matoub_export.sh \
    fix_matoub_export.py \
    test_matoub_real_stft.sh \
    test_matoub_patched_generator.sh \
    diagnose_matoub_original.py
do
    if [ -f "$f" ]; then
        rm -f -- "$f"
        echo "Supprimé : $f"
    fi
done

# 4. Supprimer uniquement les caches de compilation Gradle
if [ -d "$PROJECT/.gradle" ]; then
    rm -rf -- "$PROJECT/.gradle"
    echo "Cache Gradle local supprimé."
fi

# 5. Nettoyage des fichiers Python temporaires dans le projet
find "$PROJECT" \
    -type d -name "__pycache__" \
    -prune -exec rm -rf -- {} +

find "$PROJECT" \
    -type f -name "*.pyc" \
    -delete

echo
echo "======================================"
echo " NETTOYAGE TERMINE"
echo "======================================"

echo
echo "Sauvegarde : $BACKUP"
echo
echo "Les fichiers suivants n'ont pas été supprimés :"
echo "- Sources Kotlin"
echo "- Fichiers Gradle"
echo "- Modèle Matoub original"
echo "- Modèle Fadhma"
echo "- Assets ONNX existants"
echo "- AndroidManifest.xml"
echo "- APK et fichiers de configuration"
echo
echo "Aucune compilation n'a été lancée."
