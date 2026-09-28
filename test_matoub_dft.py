import torch
import onnx
import onnxruntime as ort
import numpy as np

torch.set_num_threads(1)

class TestDFT(torch.nn.Module):
    def forward(self, x):
        return torch.fft.rfft(x, dim=-1)

x = torch.randn(1, 256)

try:
    torch.onnx.export(
        TestDFT(),
        (x,),
        "test_matoub_dft.onnx",
        input_names=["audio"],
        output_names=["spectrum"],
        opset_version=18,
        dynamo=False,
    )
    print("Export DFT : OK")

    model = onnx.load("test_matoub_dft.onnx")
    onnx.checker.check_model(model)

    print("Opérateurs ONNX :",
          [node.op_type for node in model.graph.node])

except Exception as e:
    print("Export DFT : ECHEC")
    print(type(e).__name__, str(e))
