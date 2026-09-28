import torch
import torch.nn.functional as F

torch.set_num_threads(1)

print("PyTorch :", torch.__version__)

N_FFT = 20
HOP = 5

class TestISTFT(torch.nn.Module):
    def forward(self, spectrum, phase):
        real = spectrum * torch.cos(phase)
        imag = spectrum * torch.sin(phase)

        complex_spec = torch.complex(real, imag)

        window = torch.hann_window(
            N_FFT,
            periodic=True,
            dtype=spectrum.dtype,
            device=spectrum.device,
        )

        return torch.istft(
            complex_spec,
            n_fft=N_FFT,
            hop_length=HOP,
            win_length=N_FFT,
            window=window,
        )

model = TestISTFT().eval()

spectrum = torch.rand(1, 11, 100)
phase = torch.rand(1, 11, 100)

print("Spectrum :", spectrum.shape)
print("Phase    :", phase.shape)

with torch.inference_mode():
    try:
        audio = model(spectrum, phase)
        print("Audio    :", audio.shape)
        print("Min      :", audio.min().item())
        print("Max      :", audio.max().item())
        print("ISTFT PyTorch : OK")
    except Exception as e:
        print("ISTFT PyTorch : ERREUR")
        print(type(e).__name__, str(e))

print()
print("===== TEST EXPORT ONNX =====")

try:
    torch.onnx.export(
        model,
        (spectrum, phase),
        "/data/data/com.termux/files/home/projects/KabyleAI/test_istft.onnx",
        input_names=["spectrum", "phase"],
        output_names=["audio"],
        opset_version=18,
        dynamo=False,
    )
    print("Export ISTFT : OK")
except Exception as e:
    print("Export ISTFT : ECHEC")
    print(type(e).__name__, str(e))
