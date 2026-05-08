import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


LABELS = [
    "attention_dysregulation",
    "anxious_affect",
    "autistic_trait_discussion",
    "emotional_instability",
    "depressive_affect",
    "trauma_stress",
    "crisis_self_harm",
    "no_clear_signal",
]


tokenizer = AutoTokenizer.from_pretrained("onnx/student")

session = ort.InferenceSession(
    "onnx/student/model-int8.onnx"
)


THRESHOLDS = [0.5] * len(LABELS)


def predict(text):
    encoded = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="np"
    )

    outputs = session.run(
        None,
        {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }
    )

    logits = outputs[0]

    probs = 1 / (1 + np.exp(-logits))

    labels = {}

    for i, label in enumerate(LABELS):
        if probs[0][i] > THRESHOLDS[i]:
            labels[label] = float(probs[0][i])

    return labels


if __name__ == "__main__":
    text = "I feel exhausted and hopeless lately"

    print(predict(text))