#!/data/data/com.termux/files/usr/bin/bash
set -u

BASE="$HOME/kabyle-models/matoub_model"
WORK="$HOME/matoub_dynamic_work"
mkdir -p "$WORK"

LOG="$WORK/full_diagnostic.log"
exec > >(tee "$LOG") 2>&1

echo "======================================"
echo " DIAGNOSTIC MAToub ORIGINAL COMPLET"
echo "======================================"
date

echo
echo "[1] Vérification des fichiers"

for f in \
    "$BASE/config.json" \
    "$BASE/model.safetensors" \
    "$BASE/modeling_matoub.py" \
    "$BASE/istftnet.py" \
    "$BASE/vocab.json"
do
    if [ ! -f "$f" ]; then
        echo "FICHIER MANQUANT : $f"
        exit 1
    fi
    ls -lh "$f"
done

echo
echo "[2] Création du diagnostic Python"

cat > "$WORK/full_diagnostic.py" <<'PY'
import os
import sys
import json
import inspect
import traceback
import torch

from safetensors.torch import load_file

BASE = os.path.expanduser("~/kabyle-models")
MODEL = os.path.join(BASE, "matoub_model")

sys.path.insert(0, BASE)

from matoub_model.configuration_matoub import MatoubConfig
from matoub_model.modeling_matoub import MatoubForTextToWaveform

torch.set_num_threads(2)
torch.manual_seed(1234)

print("\n=== CONFIGURATION ===")

config = MatoubConfig.from_pretrained(MODEL)

print(config)

print("\n=== SIGNATURES ===")

print("forward:", inspect.signature(
    MatoubForTextToWaveform.forward
))

print("_synthesise:", inspect.signature(
    MatoubForTextToWaveform._synthesise
))

print("\n=== CHARGEMENT DES POIDS ORIGINAUX ===")

try:
    model = MatoubForTextToWaveform.from_pretrained(
        MODEL,
        config=config,
        local_files_only=True,
        use_safetensors=True,
    )

    model.eval()
    print("Chargement from_pretrained réussi.")

except Exception as e:
    print("Échec from_pretrained :", repr(e))
    print("Tentative de chargement direct du state_dict.")

    model = MatoubForTextToWaveform(config)

    weights_path = os.path.join(MODEL, "model.safetensors")
    state = load_file(weights_path, device="cpu")

    print("Nombre de tenseurs :", len(state))

    result = model.load_state_dict(state, strict=False)

    print("Missing keys :", result.missing_keys[:30])
    print("Unexpected keys :", result.unexpected_keys[:30])

    if result.missing_keys or result.unexpected_keys:
        print("ATTENTION : correspondance des poids non complète.")

    model.eval()

print("\n=== PARAMÈTRES ===")

print("Nombre de paramètres :", sum(
    p.numel() for p in model.parameters()
))

print("Appareil :", next(model.parameters()).device)

print("\n=== VOCABULAIRE ===")

with open(os.path.join(MODEL, "vocab.json")) as f:
    vocab = json.load(f)

print("Taille du vocabulaire :", len(vocab))

print("\n=== TESTS SYNTHÈSE ===")

# Les tests utilisent uniquement des IDs valides.
# Ils vérifient le fonctionnement du modèle PyTorch original,
# sans imposer de longueur ONNX fixe.

vocab_ids = sorted(set(
    int(v) for v in vocab.values()
    if isinstance(v, int) and int(v) >= 0
))

if not vocab_ids:
    print("Aucun ID valide trouvé.")
    sys.exit(2)

# ID valide, non réservé si possible.
token = vocab_ids[min(5, len(vocab_ids)-1)]

print("Token utilisé :", token)

lengths = [28, 36, 40]

for n in lengths:
    print(f"\n--- Test {n} tokens ---")

    ids = torch.full(
        (1, n),
        token,
        dtype=torch.long
    )

    mask = torch.ones_like(ids)

    try:
        with torch.no_grad():
            output = model(
                input_ids=ids,
                attention_mask=mask,
                return_dict=True,
            )

        print("Type sortie :", type(output))

        if hasattr(output, "keys"):
            print("Clés :", list(output.keys()))

        if hasattr(output, "waveform"):
            wav = output.waveform
        elif hasattr(output, "audio"):
            wav = output.audio
        elif isinstance(output, (tuple, list)):
            wav = output[0]
        else:
            wav = output

        if isinstance(wav, torch.Tensor):
            print("Shape waveform :", tuple(wav.shape))
            print("Dtype :", wav.dtype)
            print("Finite :", bool(torch.isfinite(wav).all()))
            print("Min :", float(wav.min()))
            print("Max :", float(wav.max()))

        else:
            print("Sortie non tensor :", type(wav))

    except Exception as e:
        print("ÉCHEC :", repr(e))
        traceback.print_exc()

print("\n=== TEST TERMINÉ ===")
PY

echo
echo "[3] Vérification syntaxe"

python -m py_compile "$WORK/full_diagnostic.py" || exit 1

echo
echo "[4] Exécution"

python "$WORK/full_diagnostic.py"

STATUS=$?

echo
echo "======================================"
echo "FIN DU DIAGNOSTIC : $STATUS"
echo "Rapport : $LOG"
echo "======================================"

exit "$STATUS"
