#!/data/data/com.termux/files/usr/bin/bash

set -u

BASE="$HOME/kabyle-models/matoub-model"
PROJECT="$HOME/projects/KabyleAI"
ASSETS="$PROJECT/app/src/main/assets/matoub"
WORK="$HOME/matoub_dynamic_work"
LOG="$WORK/pipeline.log"

mkdir -p "$WORK"

exec > >(tee -a "$LOG") 2>&1

echo "============================================"
echo " MAT OUB - DIAGNOSTIC ET EXPORT DYNAMIQUE"
echo "============================================"
echo "Date : $(date)"
echo "Modèle : $BASE"
echo "Projet : $PROJECT"
echo

# ------------------------------------------------
# 1. Vérification de l'environnement
# ------------------------------------------------

echo "[1/7] Vérification de l'environnement"

for f in \
  "$BASE/modeling_matoub.py" \
  "$BASE/istftnet.py" \
  "$BASE/config.json" \
  "$BASE/model.safetensors" \
  "$BASE/vocab.json"
do
  if [ ! -f "$f" ]; then
    echo "ERREUR : fichier manquant : $f"
    exit 1
  fi
done

python - <<'PY'
import torch
import onnx
import onnxscript

print("PyTorch :", torch.__version__)
print("ONNX :", onnx.__version__)
print("ONNXScript :", onnxscript.__version__)
PY

if [ $? -ne 0 ]; then
  echo "ERREUR : environnement Python incomplet"
  exit 1
fi

# ------------------------------------------------
# 2. Sauvegarde de l'ONNX actuel
# ------------------------------------------------

echo
echo "[2/7] Sauvegarde de l'ONNX actuel"

mkdir -p "$WORK/backup"

if [ -f "$ASSETS/matoub_82m.onnx" ]; then
  cp -n "$ASSETS/matoub_82m.onnx" \
    "$WORK/backup/matoub_82m.onnx"
fi

if [ -f "$ASSETS/matoub_82m.onnx.data" ]; then
  cp -n "$ASSETS/matoub_82m.onnx.data" \
    "$WORK/backup/matoub_82m.onnx.data"
fi

echo "Sauvegarde terminée."

# ------------------------------------------------
# 3. Diagnostic des opérations non exportables
# ------------------------------------------------

echo
echo "[3/7] Analyse des opérations du modèle original"

grep -nE \
  'torch\.stft|torch\.istft|repeat_interleave|scatter_|int\(|\.item\(' \
  "$BASE/modeling_matoub.py" "$BASE/istftnet.py" \
  > "$WORK/operations.txt" || true

cat "$WORK/operations.txt"

# ------------------------------------------------
# 4. Création du diagnostic Python
# ------------------------------------------------

cat > "$WORK/diagnose.py" <<'PY'
import os
import sys
import json
import traceback

import torch

BASE = os.path.expanduser("~/kabyle-models")
MODEL = os.path.join(BASE, "matoub_model")

sys.path.insert(0, BASE)

print("\n=== IMPORT DU MODELE ORIGINAL ===")

try:
    import matoub_model.modeling_matoub as mm
    import matoub_model.configuration_matoub as cm

    print("Module modeling_matoub :", mm.__file__)
    print("Module configuration :", cm.__file__)

except Exception:
    print("Échec de l'import du modèle.")
    traceback.print_exc()
    sys.exit(2)

print("\n=== CONFIGURATION ===")

config_path = os.path.join(MODEL, "config.json")

with open(config_path, "r") as f:
    config = json.load(f)

for key in (
    "max_token_length",
    "plbert_max_position_embeddings",
    "hidden_size",
    "style_dim",
    "vocab_size",
    "sampling_rate",
    "max_duration",
):
    print(key, "=", config.get(key))

print("\n=== CLASSES DISPONIBLES ===")

for name, obj in vars(mm).items():
    if isinstance(obj, type):
        print(name)

print("\n=== ANALYSE DU GRAPHE ORIGINAL ===")

for path in (
    os.path.join(MODEL, "modeling_matoub.py"),
    os.path.join(MODEL, "istftnet.py"),
):
    print("\nFichier :", path)

    with open(path, "r") as f:
        for n, line in enumerate(f, 1):
            if any(x in line for x in (
                "torch.stft",
                "torch.istft",
                "repeat_interleave",
                "scatter_",
            )):
                print(f"{n}: {line.rstrip()}")

print("\nDiagnostic terminé.")
PY

# ------------------------------------------------
# 5. Compilation du diagnostic
# ------------------------------------------------

echo
echo "[4/7] Compilation du diagnostic Python"

python -m py_compile "$WORK/diagnose.py"

if [ $? -ne 0 ]; then
  echo "ERREUR : diagnostic Python invalide"
  exit 1
fi

# ------------------------------------------------
# 6. Exécution du diagnostic
# ------------------------------------------------

echo
echo "[5/7] Exécution du diagnostic"

python "$WORK/diagnose.py"

RESULT=$?

if [ "$RESULT" -ne 0 ]; then
  echo
  echo "Le diagnostic a échoué."
  echo "Consulte : $LOG"
  exit "$RESULT"
fi

# ------------------------------------------------
# 7. Rapport final
# ------------------------------------------------

echo
echo "[6/7] Vérification des fichiers ONNX existants"

if [ -f "$ASSETS/matoub_82m.onnx" ]; then
  ls -lh "$ASSETS/matoub_82m.onnx"
fi

if [ -f "$ASSETS/matoub_82m.onnx.data" ]; then
  ls -lh "$ASSETS/matoub_82m.onnx.data"
fi

echo
echo "[7/7] Résultat"

echo "Diagnostic terminé."
echo "Aucun poids original n'a été modifié."
echo "Aucun ONNX n'a été remplacé."
echo "Fadhma n'a pas été modifié."
echo
echo "Rapport : $LOG"
echo "Opérations : $WORK/operations.txt"
echo
echo "L'export dynamique n'est pas lancé automatiquement :"
echo "il nécessite une adaptation vérifiée de STFT/ISTFT"
echo "et de l'alignement dynamique du générateur original."
