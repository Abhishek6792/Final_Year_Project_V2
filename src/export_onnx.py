from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer


model = ORTModelForSequenceClassification.from_pretrained(
    "checkpoints/student"
)

model.save_pretrained("onnx/student")


tokenizer = AutoTokenizer.from_pretrained("checkpoints/student")

tokenizer.save_pretrained("onnx/student")