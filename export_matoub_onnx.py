"""Export ONNX de Matoub-82M à longueur variable, pour Android.

Deux obstacles empêchaient l'export dynamique :

1. TorchSTFT (istftnet.py) utilise torch.stft / torch.istft avec des
   nombres complexes, que l'exporteur ONNX refuse
   ("Unknown number type: complex"). On le remplace par RealSTFT, qui
   fait le même calcul avec des convolutions réelles.

2. MatoubForTextToWaveform.forward boucle en Python sur le batch et
   convertit les longueurs en int() : la trace fige alors la durée de
   l'audio sur celle de l'exemple (d'où les modèles "fixed161"). On
   appelle directement _synthesise(), qui construit l'alignement avec
   repeat_interleave et reste dynamique.

Entrée ONNX : input_ids int64 [1, tokens]
Sortie ONNX : waveform float32 [1, samples], 24 kHz
"""

import faulthandler
import json
import math
import os
import random
import sys
import traceback

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL = os.path.expanduser(
    os.environ.get("MATOUB_MODEL", "~/kabyle-models/matoub-model")
)
OUT = os.path.expanduser(
    os.environ.get("MATOUB_OUT", "~/kabyle-models/matoub_dynamic_real.onnx")
)

sys.path.insert(0, os.path.dirname(MODEL))

torch.set_num_threads(1)
faulthandler.enable()


class RealSTFT(nn.Module):
    """STFT / iSTFT sans nombres complexes.

    Même interface que TorchSTFT : transform() rend (magnitude, phase),
    inverse() rend [batch, 1, samples]. Équivalent à torch.stft /
    torch.istft avec center=True, pad_mode="reflect" et une fenêtre de
    Hann périodique.
    """

    def __init__(self, n_fft: int, hop: int) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop = hop

        bins = n_fft // 2 + 1
        n = torch.arange(n_fft, dtype=torch.float32)
        k = torch.arange(bins, dtype=torch.float32)
        angle = 2 * math.pi * k[:, None] * n[None, :] / n_fft
        window = torch.hann_window(n_fft, periodic=True)

        # Analyse : une convolution par fréquence.
        self.register_buffer("fwd_cos", (torch.cos(angle) * window).unsqueeze(1))
        self.register_buffer("fwd_sin", (-torch.sin(angle) * window).unsqueeze(1))

        # Synthèse : DFT inverse réelle (DC et Nyquist comptés une fois),
        # fenêtrage et overlap-add en une seule conv_transpose1d.
        scale = torch.full((bins,), 2.0)
        scale[0] = 1.0
        scale[-1] = 1.0
        inv_cos = torch.cos(angle) * scale[:, None] / n_fft * window
        inv_sin = -torch.sin(angle) * scale[:, None] / n_fft * window
        self.register_buffer("inv_cos", inv_cos.unsqueeze(1))
        self.register_buffer("inv_sin", inv_sin.unsqueeze(1))
        self.register_buffer("win_sq", (window * window).view(1, 1, n_fft))

    def transform(self, waveform):
        x = waveform.unsqueeze(1)
        pad = self.n_fft // 2
        x = F.pad(x, (pad, pad), mode="reflect")
        real = F.conv1d(x, self.fwd_cos, stride=self.hop)
        imag = F.conv1d(x, self.fwd_sin, stride=self.hop)
        magnitude = torch.sqrt(real * real + imag * imag)
        phase = torch.atan2(imag, real)
        return magnitude, phase

    def inverse(self, magnitude, phase):
        real = magnitude * torch.cos(phase)
        imag = magnitude * torch.sin(phase)
        audio = (
            F.conv_transpose1d(real, self.inv_cos, stride=self.hop)
            + F.conv_transpose1d(imag, self.inv_sin, stride=self.hop)
        )
        envelope = F.conv_transpose1d(
            torch.ones_like(real[:, :1, :]), self.win_sq, stride=self.hop
        )
        audio = audio / torch.clamp(envelope, min=1e-11)
        pad = self.n_fft // 2
        return audio[:, :, pad:-pad]


class Wrapper(nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        style = self.model.voice.reshape(1, -1)
        waveform, _ = self.model._synthesise(input_ids, style, 1.0)
        return waveform.reshape(1, -1)


def step(title):
    print(f"\n=== {title} ===", flush=True)


def fail(code):
    traceback.print_exc()
    print(f"\nECHEC (code {code})", flush=True)
    sys.exit(code)


step("CHARGEMENT DU MODELE")

try:
    from matoub_model.configuration_matoub import MatoubConfig
    from matoub_model.modeling_matoub import MatoubForTextToWaveform
    from safetensors.torch import load_file

    config = MatoubConfig.from_pretrained(MODEL)
    model = MatoubForTextToWaveform(config)
    model.load_state_dict(
        load_file(os.path.join(MODEL, "model.safetensors")), strict=True
    )
    model.eval()

    with open(os.path.join(MODEL, "vocab.json"), encoding="utf-8") as f:
        vocab = json.load(f)

    print("Poids chargés.", flush=True)
except Exception:
    fail(2)


step("REMPLACEMENT DE TorchSTFT")

try:
    n_fft = config.gen_istft_n_fft
    hop = config.gen_istft_hop_size

    replaced = 0
    for name, module in list(model.named_modules()):
        if type(module).__name__ != "TorchSTFT":
            continue

        real = RealSTFT(n_fft, hop)

        # Vérification numérique contre l'original avant remplacement.
        x = torch.randn(1, 4000)
        mag, ph = module.transform(x)
        mag2, ph2 = real.transform(x)
        err_fwd = (torch.polar(mag, ph) - torch.polar(mag2, ph2)).abs().max()

        mag = torch.rand(1, n_fft // 2 + 1, 300) + 0.1
        ph = torch.rand(1, n_fft // 2 + 1, 300) * 6 - 3
        ref, out = module.inverse(mag, ph), real.inverse(mag, ph)
        if ref.shape != out.shape:
            raise RuntimeError(
                f"Forme iSTFT différente : {tuple(ref.shape)} "
                f"au lieu de {tuple(out.shape)}"
            )
        err_inv = (ref - out).abs().max()

        print(
            f"{name} : n_fft={n_fft} hop={hop} "
            f"erreur stft={err_fwd:.2e} istft={err_inv:.2e}",
            flush=True,
        )
        if err_fwd > 1e-3 or err_inv > 1e-3:
            raise RuntimeError("RealSTFT ne reproduit pas TorchSTFT")

        parent_name, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, attr, real)
        replaced += 1

    if replaced == 0:
        raise RuntimeError("Aucun module TorchSTFT trouvé")
except Exception:
    fail(3)


step("TEST PYTORCH")

wrapper = Wrapper(model).eval()

boundary = vocab["$"]
symbols = [v for k, v in vocab.items() if k not in ("$", " ")]
random.seed(1234)


def sample_ids(n):
    body = [random.choice(symbols) for _ in range(n - 2)]
    return np.array([[boundary] + body + [boundary]], dtype=np.int64)


LENGTHS = [6, 12, 20, 36, 43, 80, 160]
cases = {n: sample_ids(n) for n in LENGTHS}
expected = {}

try:
    with torch.no_grad():
        for n, ids in cases.items():
            audio = wrapper(torch.from_numpy(ids))
            expected[n] = audio.shape[-1]
            print(
                f"Tokens: {n} Sortie: {tuple(audio.shape)} "
                f"Finite: {bool(torch.isfinite(audio).all())}",
                flush=True,
            )
except Exception:
    fail(4)


step("EXPORT ONNX")


# Un exemple court garde les tenseurs intermédiaires petits pendant la trace.
EXAMPLE = torch.from_numpy(cases[12])


def export_dynamo():
    program = torch.onnx.export(
        wrapper,
        (EXAMPLE,),
        input_names=["input_ids"],
        output_names=["waveform"],
        dynamic_shapes={"input_ids": {1: torch.export.Dim("tokens", min=2, max=510)}},
        dynamo=True,
        opset_version=18,
    )
    program.save(OUT)


def export_torchscript():
    torch.onnx.export(
        wrapper,
        (EXAMPLE,),
        OUT,
        input_names=["input_ids"],
        output_names=["waveform"],
        dynamic_axes={"input_ids": {1: "tokens"}, "waveform": {1: "samples"}},
        opset_version=17,
        dynamo=False,
        do_constant_folding=True,
    )


# Le PyTorch de Termux plante dans l'exporteur TorchScript ("attribute
# has the wrong type") : on essaie d'abord l'exporteur torch.export.
exported = None
with torch.no_grad():
    exporters = (("dynamo", export_dynamo), ("torchscript", export_torchscript))
    only = os.environ.get("MATOUB_EXPORTER")
    for name, fn in exporters:
        if only and name != only:
            continue
        print(f"Exporteur {name}...", flush=True)
        try:
            if os.path.exists(OUT):
                os.remove(OUT)
            fn()
            exported = name
            break
        except Exception:
            print(f"\nExporteur {name} : ECHEC", flush=True)
            traceback.print_exc(limit=4)

if exported is None:
    print("\nECHEC (code 5) : aucun exporteur n'a abouti", flush=True)
    sys.exit(5)

try:
    onnx.checker.check_model(OUT)
    size = os.path.getsize(OUT) / 1e6
    print(f"Export terminé avec {exported} : {OUT} ({size:.0f} Mo)", flush=True)
except Exception:
    fail(5)


step("TEST ONNX RUNTIME")

# L'audio varie d'une exécution à l'autre (bruit de la source harmonique) :
# on compare la longueur, qui dépend seulement des durées prédites.
try:
    session = ort.InferenceSession(OUT, providers=["CPUExecutionProvider"])
    ok = True

    for n, ids in cases.items():
        audio = session.run(["waveform"], {"input_ids": ids})[0]
        same = audio.shape[-1] == expected[n]
        finite = bool(np.isfinite(audio).all())
        ok = ok and same and finite
        print(
            f"Tokens: {n} Sortie: {audio.shape} "
            f"PyTorch: {expected[n]} "
            f"{'OK' if same else 'LONGUEUR DIFFERENTE'} Finite: {finite}",
            flush=True,
        )

    if len(set(expected.values())) == 1:
        print("Toutes les sorties ont la même longueur : modèle figé.")
        ok = False

    if not ok:
        print("\nECHEC : le modèle ONNX ne suit pas PyTorch.", flush=True)
        sys.exit(6)
except SystemExit:
    raise
except Exception:
    fail(6)


print("\n=== EXPORT ET TESTS TERMINES ===", flush=True)
print("Installation : ./build_install.sh " + OUT, flush=True)
