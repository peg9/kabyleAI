import torch

torch.manual_seed(1234)

n_fft = 20
hop = 5
win = 20

window = torch.hann_window(win, periodic=True)

print("=== TEST ISTFT DU GENERATEUR MAToub ===")

for frames in (48, 100, 200, 1000):
    magnitude = torch.rand(1, 11, frames) + 0.1
    phase = torch.rand(1, 11, frames) * 2 - 1

    spectrum = magnitude * torch.exp(phase * 1j)

    with torch.no_grad():
        y = torch.istft(
            spectrum,
            n_fft,
            hop,
            win,
            window=window
        )

    expected = (frames - 1) * hop

    print(
        f"Frames={frames}, "
        f"sortie={y.shape[-1]}, "
        f"longueur attendue approximative={expected}"
    )

    assert torch.isfinite(y).all()

print("=== FIN ===")
