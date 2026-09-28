import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
import onnxruntime as ort
import numpy as np
import math

torch.set_num_threads(1)

N = 20
HOP = 5
K = N // 2 + 1

class DynamicISTFT(nn.Module):
    def __init__(self):
        super().__init__()

        n = torch.arange(N, dtype=torch.float32)
        k = torch.arange(K, dtype=torch.float32)
        angle = 2 * math.pi * k[:, None] * n[None, :] / N

        cos = torch.cos(angle)
        sin = torch.sin(angle)

        # Poids de la transformee inverse reelle
        weights = torch.ones(K)
        weights[1:-1] = 2.0

        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)
        self.register_buffer("weights", weights)
        self.register_buffer(
            "window",
            torch.hann_window(N, periodic=True)
        )

        # Overlap-add avec ConvTranspose1d
        overlap = torch.zeros(N, 1, N)
        for i in range(N):
            overlap[i, 0, i] = 1.0

        self.register_buffer("overlap", overlap)

    def forward(self, magnitude, phase):
        # magnitude, phase : [batch, 11, frames]

        real = magnitude * torch.cos(phase)
        imag = magnitude * torch.sin(phase)

        real = real.transpose(1, 2)
        imag = imag.transpose(1, 2)

        # Reconstruction inverse DFT reelle
        frames = (
            torch.matmul(real * self.weights, self.cos)
            - torch.matmul(imag * self.weights, self.sin)
        ) / N

        frames = frames.transpose(1, 2)
        frames = frames * self.window.view(1, N, 1)

        # Overlap-add avec un pas de 5 echantillons
        audio = F.conv_transpose1d(
            frames,
            self.overlap,
            stride=HOP
        )

        # Normalisation par la somme des fenetres
        norm_input = self.window.square().view(1, 1, N)
        norm = F.conv_transpose1d(
            torch.ones_like(frames[:, :1, :]),
            norm_input,
            stride=HOP
        )

        audio = audio / torch.clamp(norm, min=1e-8)

        # Retrait du padding central (N // 2)
        audio = audio[:, :, N // 2:-N // 2]

        return audio.squeeze(1)


model = DynamicISTFT().eval()

try:
    torch.onnx.export(
        model,
        (
            torch.rand(1, K, 100),
            torch.rand(1, K, 100)
        ),
        "test_matoub_istft_dynamic.onnx",
        input_names=["magnitude", "phase"],
        output_names=["audio"],
        dynamic_shapes=(
            {2: torch.export.Dim("frames", min=20, max=500)},
            {2: torch.export.Dim("frames", min=20, max=500)}
        ),
        opset_version=18,
        dynamo=True,
        external_data=False
    )

    print("Export ONNX : OK")

    onnx_model = onnx.load(
        "test_matoub_istft_dynamic.onnx"
    )
    onnx.checker.check_model(onnx_model)
    print("Verification ONNX : OK")

    session = ort.InferenceSession(
        "test_matoub_istft_dynamic.onnx",
        providers=["CPUExecutionProvider"]
    )

    for frames in [50, 100, 150]:
        mag = np.random.rand(1, K, frames).astype(np.float32)
        phase = np.random.rand(1, K, frames).astype(np.float32)

        result = session.run(
            ["audio"],
            {"magnitude": mag, "phase": phase}
        )[0]

        expected = (frames - 1) * HOP

        print(
            f"Trames={frames} | "
            f"Sortie={result.shape} | "
            f"Attendu={expected}"
        )

    print("RESULTAT : TEST DYNAMIQUE TERMINE")

except Exception as e:
    print("ECHEC :", type(e).__name__, str(e))
