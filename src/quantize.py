from onnxruntime.quantization import quantize_dynamic
from onnxruntime.quantization import QuantType


quantize_dynamic(
    model_input="onnx/student/model.onnx",
    model_output="onnx/student/model-int8.onnx",
    weight_type=QuantType.QInt8,
)