import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path.home() / "kabyle-models"))

from matoub_model.istftnet import TorchSTFT

torch.manual_seed(1234)

n_fft = 20
hop = 5
win = 20

stft = TorchSTFT(
    filter_length=n_fft,
    hop_length=hop,
    win_length=win
).eval()

print("=== TEST STFT / ISTFT MAToub ===")
print("PyTorch :", torch.__version__)

for length in (240, 1000, 4096):
    print(f"\n--- Signal {length} échantillons ---")

    x = torch.randn(1, length)

    try:
        with torch.no_grad():
            magnitude, phase = stft.transform(x)
            y = stft.inverse(magnitude, phase)

        print("Entrée :", tuple(x.shape))
        print("Magnitude :", tuple(magnitude.shape))
        print("Phase :", tuple(phase.shape))
        print("Reconstruction :", tuple(y.shape))
        print("Valeurs finies :", bool(torch.isfinite(y).all()))

        y = y.squeeze(-2)

        if y.shape == x.shape:
            error = (x - y).abs().max().item()
            print("Erreur absolue max :", error)
        else:
            print("ATTENTION : longueurs différentes")

    except Exception as e:
        print("ERREUR :", repr(e))
        raise

print("\n=== FIN DU TEST ===")
