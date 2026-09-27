#!/data/data/com.termux/files/usr/bin/bash

set -u

PROJECT="$HOME/projects/KabyleAI"
MODEL="$HOME/kabyle-models/matoub-model"
ASSETS="$PROJECT/app/src/main/assets/matoub"
PY="$PREFIX/bin/python"

cd "$PROJECT" || exit 1

echo "=========================================="
echo " KABYLE AI - MAToub DYNAMIC ONNX"
echo "=========================================="

# 1. Vérifications
for f in \
    "$MODEL/model.safetensors" \
    "$MODEL/config.json" \
    "$MODEL/vocab.json" \
    "$PROJECT/app/src/main/java/com/kabyleai/app/MatoubTts.kt" \
    "$PROJECT/app/src/main/java/com/kabyleai/app/MatoubTokenizer.kt"
do
    if [ ! -f "$f" ]; then
        echo "ERREUR : fichier absent : $f"
        exit 1
    fi
done

# 2. Sauvegarde
BACKUP="$PROJECT/backup_matoub_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP"

cp "$PROJECT/app/src/main/java/com/kabyleai/app/MatoubTts.kt" "$BACKUP/"
cp "$PROJECT/app/src/main/java/com/kabyleai/app/MatoubTokenizer.kt" "$BACKUP/"
cp -r "$ASSETS" "$BACKUP/assets"

echo "Sauvegarde : $BACKUP"

# 3. Vérification des dépendances
"$PY" - <<'PY'
import torch, onnx, onnxruntime
print("PyTorch :", torch.__version__)
print("ONNX :", onnx.__version__)
print("ONNX Runtime :", onnxruntime.__version__)
PY

if [ $? -ne 0 ]; then
    echo "ERREUR : dépendances Python manquantes."
    exit 1
fi

# 4. Génération du script de diagnostic et d'export
cat > "$PROJECT/export_matoub_dynamic.py" <<'PY'
import os
import sys
import json
import traceback
import inspect
import torch
import onnx
import onnxruntime as ort

MODEL = os.path.expanduser("~/kabyle-models/matoub-model")
OUT = os.path.expanduser("~/kabyle-models/matoub_dynamic.onnx")

sys.path.insert(0, os.path.expanduser("~/kabyle-models"))

print("\n=== CHARGEMENT DU MODELE ===")

try:
    from matoub_model.configuration_matoub import MatoubConfig
    from matoub_model.modeling_matoub import MatoubForTextToWaveform

    config = MatoubConfig.from_pretrained(MODEL)
    model = MatoubForTextToWaveform(config)

    from safetensors.torch import load_file

    weights = load_file(os.path.join(MODEL, "model.safetensors"))
    result = model.load_state_dict(weights, strict=True)

    model.eval()
    model.to("cpu")

    print("Poids chargés strictement.")
    print("Paramètres :", sum(p.numel() for p in model.parameters()))
    print("Forward :", inspect.signature(model.forward))

except Exception:
    traceback.print_exc()
    sys.exit(2)

# Tester plusieurs longueurs en PyTorch.
print("\n=== TESTS PYTORCH ===")

lengths = [6, 12, 20, 36, 50]

valid = []

for n in lengths:
    try:
        ids = torch.zeros((1, n), dtype=torch.long)
        with torch.no_grad():
            result = model(input_ids=ids)

        if hasattr(result, "waveform"):
            audio = result.waveform
        elif isinstance(result, (tuple, list)):
            audio = result[0]
        else:
            audio = result

        print(
            "Tokens:", n,
            "Shape:", tuple(audio.shape),
            "Finite:", bool(torch.isfinite(audio).all())
        )

        valid.append(n)

    except Exception as e:
        print("ECHEC longueur", n, ":", repr(e))

if len(valid) < 2:
    print("Le modèle ne passe pas les tests de longueurs variables.")
    sys.exit(3)

# Export dynamique : tentative avec le nouvel exporteur.
print("\n=== EXPORT ONNX DYNAMIQUE ===")

example = torch.zeros((1, 36), dtype=torch.long)

class Wrapper(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.model = m

    def forward(self, input_ids):
        result = self.model(input_ids=input_ids)

        if hasattr(result, "waveform"):
            return result.waveform

        if isinstance(result, (tuple, list)):
            return result[0]

        return result

wrapper = Wrapper(model).eval()

try:
    torch.onnx.export(
        wrapper,
        (example,),
        OUT,
        input_names=["input_ids"],
        output_names=["waveform"],
        dynamic_shapes={
            "input_ids": {
                0: None,
                1: torch.export.Dim("tokens", min=2, max=510)
            }
        },
        dynamo=True,
        opset_version=20,
        external_data=True
    )

    print("Export terminé :", OUT)

except Exception:
    print("\nECHEC de l'export dynamique.")
    traceback.print_exc()
    print("Le modèle Android existant est conservé.")
    sys.exit(4)

# Vérification ONNX
print("\n=== VERIFICATION ONNX ===")

try:
    m = onnx.load(OUT, load_external_data=False)
    onnx.checker.check_model(m)

    for x in m.graph.input:
        print("INPUT", x.name, [
            d.dim_param or d.dim_value
            for d in x.type.tensor_type.shape.dim
        ])

    for x in m.graph.output:
        print("OUTPUT", x.name, [
            d.dim_param or d.dim_value
            for d in x.type.tensor_type.shape.dim
        ])

except Exception:
    traceback.print_exc()
    sys.exit(5)

# Test ONNX Runtime sur les différentes longueurs
print("\n=== TESTS ONNX RUNTIME ===")

try:
    session = ort.InferenceSession(
        OUT,
        providers=["CPUExecutionProvider"]
    )

    for n in [6, 12, 20, 36, 50]:
        x = torch.zeros((1, n), dtype=torch.long).numpy()

        try:
            y = session.run(None, {"input_ids": x})[0]
            print("Tokens:", n, "Output:", y.shape)
        except Exception as e:
            print("ECHEC ORT longueur", n, repr(e))
            sys.exit(6)

except Exception:
    traceback.print_exc()
    sys.exit(7)

print("\nEXPORT ET TESTS TERMINES.")
print("Le modèle Android n'a pas été remplacé.")
print("Il reste à adapter MatoubTts.kt au graphe validé.")
PY

# 5. Vérification syntaxique obligatoire
echo
echo "=== COMPILATION PYTHON ==="

"$PY" -m py_compile "$PROJECT/export_matoub_dynamic.py" || exit 1

# 6. Exécution export
echo
echo "=== EXPORT DYNAMIQUE ==="

"$PY" "$PROJECT/export_matoub_dynamic.py"
STATUS=$?

if [ "$STATUS" -ne 0 ]; then
    echo
    echo "L'export dynamique a échoué (code $STATUS)."
    echo "Les fichiers Android originaux sont conservés."
    echo "Sauvegarde : $BACKUP"
    exit "$STATUS"
fi

echo
echo "Export validé techniquement."
echo "Le modèle Android n'a pas été remplacé automatiquement."
echo "Il faut adapter le runtime Kotlin aux dimensions ONNX validées."
