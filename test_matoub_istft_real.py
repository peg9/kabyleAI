import torch
import math

torch.set_num_threads(1)

N = 20
HOP = 5
FRAMES = 100

# Fenetre de Hann periodique, comme torch.istft
window = torch.hann_window(N, periodic=True)

# Matrices de reconstruction inverse DFT, uniquement reelles
n = torch.arange(N, dtype=torch.float32)
k = torch.arange(N // 2 + 1, dtype=torch.float32)

angles = 2 * math.pi * k[:, None] * n[None, :] / N

cos_matrix = torch.cos(angles)
sin_matrix = torch.sin(angles)

# Spectre aleatoire : partie reelle et imaginaire
real = torch.randn(1, N // 2 + 1, FRAMES)
imag = torch.randn(1, N // 2 + 1, FRAMES)

# Reference PyTorch
spectrum = torch.complex(real, imag)

reference = torch.istft(
    spectrum,
    n_fft=N,
    hop_length=HOP,
    win_length=N,
    window=window,
    center=True,
    normalized=False,
    onesided=True,
    return_complex=False,
)

# Reconstruction inverse DFT manuelle
# X0 + XNyquist + 2 * somme des frequences intermediaires
frames = (
    real[:, 0:1, :]
    + real[:, N // 2:N // 2 + 1, :]
       * torch.cos(math.pi * n)[None, :, None]
    + 2 * (
        real[:, 1:N // 2, :].transpose(1, 2)
        @ cos_matrix[1:N // 2, :]
        - imag[:, 1:N // 2, :].transpose(1, 2)
        @ sin_matrix[1:N // 2, :]
    ).transpose(1, 2)
) / N

# Application de la fenetre
frames = frames * window[None, :, None]

# Overlap-add
total_length = (FRAMES - 1) * HOP + N

audio = torch.zeros(1, total_length)
envelope = torch.zeros(total_length)

for i in range(FRAMES):
    start = i * HOP
    audio[:, start:start + N] += frames[:, :, i]
    envelope[start:start + N] += window * window

# Normalisation et retrait du padding central
audio = audio / envelope.clamp(min=1e-8)[None, :]

manual = audio[:, N // 2:-N // 2]

# Comparaison
min_length = min(reference.shape[-1], manual.shape[-1])
reference = reference[:, :min_length]
manual = manual[:, :min_length]

diff = (reference - manual).abs()

print("PyTorch shape :", tuple(reference.shape))
print("Manuel shape  :", tuple(manual.shape))
print("Erreur max    :", diff.max().item())
print("Erreur moyenne:", diff.mean().item())

if diff.max().item() < 1e-4:
    print("RESULTAT : ISTFT reelle validee")
else:
    print("RESULTAT : ISTFT a corriger")
