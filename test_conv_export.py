import torch
import torch.nn as nn
import onnx
import onnxscript

class TestConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("weight", torch.eye(20).unsqueeze(1))

    def forward(self, x):
        return torch.nn.functional.conv_transpose1d(
            x, self.weight, stride=5
        )

model = TestConv().eval()
x = torch.randn(1, 20, 100)

torch.onnx.export(
    model,
    (x,),
    "test_conv_export.onnx",
    input_names=["x"],
    output_names=["audio"],
    dynamic_shapes=({0: torch.export.Dim("batch"), 2: torch.export.Dim("frames", min=20, max=500)},),
    dynamo=True,
    opset_version=18,
    external_data=False
)

onnx.checker.check_model("test_conv_export.onnx")
print("EXPORT CONV OK")
