import torch
import torch.nn as nn
import onnx

original = torch._C._jit_pass_onnx_node_shape_type_inference

torch._C._jit_pass_onnx_node_shape_type_inference = (
    lambda *args, **kwargs: None
)

class TestModel(nn.Module):
    def forward(self, x):
        return x.transpose(0, 1)

model = TestModel().eval()
x = torch.randn(2, 3)

try:
    torch.onnx.export(
        model,
        (x,),
        "test_onnx_patch.onnx",
        opset_version=17,
        dynamo=False,
        input_names=["input"],
        output_names=["output"],
    )
    onnx.checker.check_model("test_onnx_patch.onnx")
    print("EXPORT ET VERIFICATION OK")
finally:
    torch._C._jit_pass_onnx_node_shape_type_inference = original
