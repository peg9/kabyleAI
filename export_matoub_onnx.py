"""Export ONNX de Matoub-82M à longueur variable, pour Android.

Trois obstacles empêchaient l'export dynamique :

1. TorchSTFT (istftnet.py) utilise torch.stft / torch.istft avec des
   nombres complexes, que l'exporteur ONNX refuse
   ("Unknown number type: complex"). On le remplace par RealSTFT, qui
   fait le même calcul avec des convolutions réelles.

2. MatoubForTextToWaveform.forward boucle en Python sur le batch et
   convertit les longueurs en int() : la trace fige alors la durée de
   l'audio sur celle de l'exemple (d'où les modèles "fixed161").

3. Le nombre total de trames (somme des durées prédites) n'est connu
   qu'à l'exécution : torch.export ne peut pas s'en servir comme
   dimension ("Could not guard on data-dependent expression u0 >= 1").

Le modèle est donc exporté en deux graphes, et l'alignement (durées ->
matrice tokens x trames) se fait entre les deux, dans l'application :

  matoub_front.onnx
    entrée  input_ids  int64   [1, tokens]
    sorties d          float32 [1, tokens, C]
            frames     int64   [tokens]        trames par token

  matoub_back.onnx
    entrées input_ids  int64   [1, tokens]
            d          float32 [1, tokens, C]
            alignment  float32 [1, tokens, N]  N = somme de frames
    sortie  waveform   float32 [1, samples]    24 kHz
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
OUT_DIR = os.path.expanduser(
    os.environ.get("MATOUB_OUT_DIR", "~/kabyle-models/matoub_dynamic")
)
FRONT = os.path.join(OUT_DIR, "matoub_front.onnx")
BACK = os.path.join(OUT_DIR, "matoub_back.onnx")

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


class Front(nn.Module):
    """Début de MatoubForTextToWaveform._synthesise (speed=1) : encodage du
    texte et durées prédites."""

    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        m = self.model
        style = m.voice.reshape(1, -1)
        prosody_style = style[:, m.config.style_dim:]

        attention = torch.ones_like(input_ids)
        bert_dur = m.bert(input_ids, attention_mask=attention).last_hidden_state
        d_en = m.bert_encoder(bert_dur).transpose(-1, -2)

        d = m.predictor.text_encoder(d_en, prosody_style)
        x, _ = m.predictor.lstm(d)
        duration = torch.sigmoid(m.predictor.duration_proj(x)).sum(dim=-1)
        frames = torch.round(duration).clamp(min=1).long().squeeze(0)
        return d, frames


class Back(nn.Module):
    """Fin de _synthesise : prosodie, décodeur et vocodeur, à partir de
    l'alignement fourni."""

    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids, d, alignment):
        m = self.model
        style_dim = m.config.style_dim
        style = m.voice.reshape(1, -1)
        prosody_style = style[:, style_dim:]
        acoustic_style = style[:, :style_dim]

        pitch, energy = m.predictor.contours(
            d.transpose(-1, -2) @ alignment, prosody_style
        )
        asr = m.text_encoder(input_ids) @ alignment
        waveform = m.decoder(asr, pitch, energy, acoustic_style)
        return waveform.squeeze(1).squeeze(0).reshape(1, -1)


def refresh_shapes(path):
    """Efface les formes enregistrées dans le graphe et les recalcule.

    Le LSTM exporté garde la forme de l'exemple (12 tokens) sur sa sortie,
    et tout ce qui en dépend hérite de ce 12 ; ONNX Runtime s'y fie pour
    fusionner des opérations et échoue dès que la longueur change.
    """
    graph = onnx.load(path)
    del graph.graph.value_info[:]
    for value in graph.graph.output:
        value.type.tensor_type.ClearField("shape")
    # torch.export note certaines dimensions comme figées (le LSTM est
    # décomposé avec la taille de l'exemple) alors que le graphe ne dépend
    # pas de ces valeurs : on les remet symboliques.
    symbolic = {
        "input_ids": {1: "tokens"},
        "d": {1: "tokens"},
        "alignment": {1: "tokens", 2: "frames"},
    }
    for value in graph.graph.input:
        for index, name in symbolic.get(value.name, {}).items():
            dim = value.type.tensor_type.shape.dim[index]
            dim.Clear()
            dim.dim_param = name
    graph = onnx.shape_inference.infer_shapes(graph)
    onnx.save(graph, path)


def make_alignment(frames, tokens):
    """Matrice [1, tokens, N] : 1 pour les trames appartenant au token."""
    total = int(frames.sum())
    alignment = np.zeros((1, tokens, total), dtype=np.float32)
    start = 0
    for t, count in enumerate(frames):
        alignment[0, t, start:start + int(count)] = 1.0
        start += int(count)
    return alignment


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

front = Front(model).eval()
back = Back(model).eval()

boundary = vocab["$"]
symbols = [v for k, v in vocab.items() if k not in ("$", " ")]
random.seed(1234)


def sample_ids(n):
    body = [random.choice(symbols) for _ in range(n - 2)]
    return np.array([[boundary] + body + [boundary]], dtype=np.int64)


LENGTHS = [6, 12, 20, 36, 43, 80, 160]
cases = {n: sample_ids(n) for n in LENGTHS}
expected = {}
expected_frames = {}

try:
    with torch.no_grad():
        for n, ids in cases.items():
            tokens = torch.from_numpy(ids)
            reference, ref_frames = model._synthesise(
                tokens, model.voice.reshape(1, -1), 1.0
            )
            d, frames = front(tokens)
            if not torch.equal(frames, ref_frames):
                raise RuntimeError(f"Durées différentes du modèle ({n} tokens)")

            alignment = torch.from_numpy(make_alignment(frames.numpy(), n))
            audio = back(tokens, d, alignment)
            expected[n] = reference.shape[-1]
            expected_frames[n] = frames.numpy()
            if audio.shape[-1] != expected[n]:
                raise RuntimeError(
                    f"Longueur différente du modèle ({n} tokens) : "
                    f"{audio.shape[-1]} au lieu de {expected[n]}"
                )
            print(
                f"Tokens: {n} Trames: {int(frames.sum())} "
                f"Sortie: {tuple(audio.shape)} "
                f"Finite: {bool(torch.isfinite(audio).all())}",
                flush=True,
            )
except Exception:
    fail(4)


step("EXPORT ONNX")

os.makedirs(OUT_DIR, exist_ok=True)
EX = cases[12]
tokens_dim = torch.export.Dim("tokens", min=2, max=510)
frames_dim = torch.export.Dim("frames", min=2, max=20000)

try:
    with torch.no_grad():
        ex_ids = torch.from_numpy(EX)
        ex_d, ex_frames = front(ex_ids)
        ex_align = torch.from_numpy(make_alignment(ex_frames.numpy(), EX.shape[1]))

        print("Export du graphe avant...", flush=True)
        torch.onnx.export(
            front,
            (ex_ids,),
            input_names=["input_ids"],
            output_names=["d", "frames"],
            dynamic_shapes={"input_ids": {1: tokens_dim}},
            dynamo=True,
            opset_version=18,
            optimize=False,
        ).save(FRONT)

        print("Export du graphe arrière...", flush=True)
        torch.onnx.export(
            back,
            (ex_ids, ex_d, ex_align),
            input_names=["input_ids", "d", "alignment"],
            output_names=["waveform"],
            dynamic_shapes={
                "input_ids": {1: tokens_dim},
                "d": {1: tokens_dim},
                "alignment": {1: tokens_dim, 2: frames_dim},
            },
            dynamo=True,
            opset_version=18,
            optimize=False,
        ).save(BACK)

    for path in (FRONT, BACK):
        refresh_shapes(path)
        onnx.checker.check_model(path)
        print(f"{path} ({os.path.getsize(path) / 1e6:.0f} Mo)", flush=True)
except Exception:
    fail(5)


step("TEST ONNX RUNTIME")

# L'audio varie d'une exécution à l'autre (bruit de la source harmonique) :
# on compare les durées, exactes, et la longueur de l'audio.
try:
    front_session = ort.InferenceSession(FRONT, providers=["CPUExecutionProvider"])
    back_session = ort.InferenceSession(BACK, providers=["CPUExecutionProvider"])
    for label, session in (("avant", front_session), ("arrière", back_session)):
        print(
            f"Entrées {label} : "
            + ", ".join(f"{i.name}{i.shape}" for i in session.get_inputs()),
            flush=True,
        )
    ok = True

    for n, ids in cases.items():
        d, frames = front_session.run(["d", "frames"], {"input_ids": ids})
        same_frames = np.array_equal(frames, expected_frames[n])
        alignment = make_alignment(frames, n)
        try:
            audio = back_session.run(
                ["waveform"],
                {"input_ids": ids, "d": d, "alignment": alignment},
            )[0]
        except Exception:
            print(
                f"Graphe arrière refusé pour {n} tokens : "
                f"d{d.shape} alignment{alignment.shape}",
                flush=True,
            )
            raise
        same = audio.shape[-1] == expected[n]
        finite = bool(np.isfinite(audio).all())
        ok = ok and same and finite and same_frames
        print(
            f"Tokens: {n} Sortie: {audio.shape} PyTorch: {expected[n]} "
            f"{'OK' if same and same_frames else 'DIFFERENT'} Finite: {finite}",
            flush=True,
        )

    if len(set(expected.values())) == 1:
        print("Toutes les sorties ont la même longueur : modèle figé.")
        ok = False

    if not ok:
        print("\nECHEC : les modèles ONNX ne suivent pas PyTorch.", flush=True)
        sys.exit(6)
except SystemExit:
    raise
except Exception:
    fail(6)


print("\n=== EXPORT ET TESTS TERMINES ===", flush=True)
print("Installation : ./build_install.sh " + OUT_DIR, flush=True)
