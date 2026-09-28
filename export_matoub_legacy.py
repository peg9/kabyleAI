import os
import sys
import traceback
import inspect
import torch
import onnx
import onnxruntime as ort
import numpy as np

MODEL = os.path.expanduser("~/kabyle-models/matoub-model")
OUT = os.path.expanduser("~/kabyle-models/matoub_legacy_dynamic.onnx")

sys.path.insert(0, os.path.expanduser("~/kabyle-models"))

os.environ["PYTHONFAULTHANDLER"] = "1"

torch.set_num_threads(1)

print("=== CHARGEMENT DU MODELE ===", flush=True)

try:
    from matoub_model.configuration_matoub import MatoubConfig
    from matoub_model.modeling_matoub import MatoubForTextToWaveform
    from safetensors.torch import load_file

    config = MatoubConfig.from_pretrained(MODEL)
    model = MatoubForTextToWaveform(config)

    weights = load_file(os.path.join(MODEL, "model.safetensors"))
    model.load_state_dict(weights, strict=True)

    model.eval()
    model.to("cpu")

    print("Poids chargés.", flush=True)
    print("Forward :", inspect.signature(model.forward), flush=True)

except Exception:
    traceback.print_exc()
    sys.exit(2)


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

print("\n=== TEST PYTORCH ===", flush=True)

for n in [6, 12, 20, 36, 50]:
    ids = torch.zeros((1, n), dtype=torch.long)

    with torch.no_grad():
        audio = wrapper(ids)

    print(
        "Tokens:", n,
        "Sortie:", tuple(audio.shape),
        "Finite:", bool(torch.isfinite(audio).all()),
        flush=True
    )


print("\n=== EXPORT LEGACY DYNAMIQUE ===", flush=True)

example = torch.zeros((1, 36), dtype=torch.long)

try:
    torch.onnx.export(
        wrapper,
        (example,),
        OUT,
        input_names=["input_ids"],
        output_names=["waveform"],
        dynamic_axes={
            "input_ids": {
                0: "batch",
                1: "tokens"
            },
            "waveform": {
                0: "batch",
                1: "audio_length"
            }
        },
        opset_version=17,
        dynamo=False,
        do_constant_folding=False,
        export_params=True,
        external_data=True,
        verbose=False
    )

    print("Export terminé :", OUT, flush=True)

except Exception:
    print("ECHEC EXPORT LEGACY", flush=True)
    traceback.print_exc()
    sys.exit(4)


print("\n=== VERIFICATION ONNX ===", flush=True)

try:
    graph = onnx.load(OUT, load_external_data=False)
    onnx.checker.check_model(graph)

    for x in graph.graph.input:
        print("INPUT", x.name, flush=True)
        print(x.type.tensor_type.shape, flush=True)

    for x in graph.graph.output:
        print("OUTPUT", x.name, flush=True)
        print(x.type.tensor_type.shape, flush=True)

except Exception:
    traceback.print_exc()
    sys.exit(5)


print("\n=== TEST ONNX RUNTIME ===", flush=True)

try:
    session = ort.InferenceSession(
        OUT,
        providers=["CPUExecutionProvider"]
    )

    for n in [6, 12, 20, 36, 50]:
        ids = np.zeros((1, n), dtype=np.int64)

        outputs = session.run(
            ["waveform"],
            {"input_ids": ids}
        )

        audio = outputs[0]

        print(
            "Tokens:", n,
            "Sortie:", audio.shape,
            "Finite:", bool(np.isfinite(audio).all()),
            flush=True
        )

except Exception:
    print("ECHEC ONNX RUNTIME", flush=True)
    traceback.print_exc()
    sys.exit(6)


print("\n=== EXPORT ET TESTS TERMINES ===", flush=True)
print("Fichier candidat :", OUT, flush=True)
