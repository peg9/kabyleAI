#!/bin/bash
set -euo pipefail

# Sessions cloud uniquement
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "$CLAUDE_PROJECT_DIR"

# Outils Python : linter (ruff) et paquets utilisés par les scripts d'export et de test.
# torch et transformers (export des modèles) ne sont pas installés : trop lourds pour un démarrage de session.
python3 -m pip install --quiet --upgrade ruff numpy onnx onnxruntime

# Les tokenizers Java sont du Java pur : on les compile avec javac, sans SDK Android.
# (dl.google.com est bloqué par la politique réseau par défaut, donc pas de SDK ni de Gradle Android ici.)
mkdir -p "${TMPDIR:-/tmp}/kabyleai-tokenizer-build"
javac -d "${TMPDIR:-/tmp}/kabyleai-tokenizer-build" \
  app/src/main/java/com/kabyleai/app/Tokenizer.java \
  app/src/main/java/com/kabyleai/app/BpeTokenizer.java \
  app/src/main/java/com/kabyleai/app/UnigramTokenizer.java \
  tools/nllb/TokenizerCheck.java

echo "export KABYLEAI_TOKENIZER_CLASSES=\"${TMPDIR:-/tmp}/kabyleai-tokenizer-build\"" >> "${CLAUDE_ENV_FILE:-/dev/null}"
