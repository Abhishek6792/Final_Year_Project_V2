# Research-Grade Mental Health Signal Analysis System — Full Training Codebase (v2.0)

## Project Structure

```text
project/
│
├── data/
│   ├── labeled.csv
│   ├── unlabeled.csv
│   └── processed/
│
├── checkpoints/
├── outputs/
├── onnx/
│
├── configs/
│   └── config.yaml
│
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── losses.py
│   ├── train_teacher.py
│   ├── train_dapt.py
│   ├── distill_student.py
│   ├── calibrate.py
│   ├── export_onnx.py
│   ├── inference.py
│   ├── metrics.py
│   └── utils.py
│
└── requirements.txt
```

---

# requirements.txt

```txt
transformers>=4.40.0
accelerate>=0.30.0
datasets>=2.19.0
torch>=2.2.0
scikit-learn>=1.4.0
pandas>=2.2.0
numpy>=1.26.0
iterative-stratification>=0.1.9
onnxruntime>=1.17.0
optimum>=1.18.0
scipy>=1.12.0
```

---

# configs/config.yaml

```yaml
teacher_model: microsoft/deberta-v3-base
student_model: distilroberta-base

max_length: 512
batch_size: 8
lr: 2e-5
epochs: 5
weight_decay: 0.01
warmup_ratio: 0.1

num_labels: 8
num_severity: 4

mixed_precision: true

label_names:
  - attention_dysregulation
  - anxious_affect
  - autistic_trait_discussion
  - emotional_instability
  - depressive_affect
  - trauma_stress
  - crisis_self_harm
  - no_clear_signal
```

---

# src/dataset.py

```python
import torch
from torch.utils.data import Dataset


class MentalHealthDataset(Dataset):
    def __init__(self, df, tokenizer, label_cols, max_length=512):
        self.df = df
        self.tokenizer = tokenizer
        self.label_cols = label_cols
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        encoding = self.tokenizer(
            str(row["text"]),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        labels = torch.tensor(
            row[self.label_cols].values.astype(float),
            dtype=torch.float
        )

        severity = torch.tensor(int(row["severity"]), dtype=torch.long)

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": labels,
            "severity": severity,
        }

        if "platform" in row:
            item["platform"] = torch.tensor(int(row["platform"]))

        return item
```

---

# src/model.py

```python
import torch
import torch.nn as nn
from transformers import AutoModel


class MentalHealthModel(nn.Module):
    def __init__(
        self,
        model_name,
        num_labels=8,
        num_severity=4,
        num_platforms=None,
        dropout=0.1,
    ):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)

        hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(dropout)

        self.signal_head = nn.Linear(hidden_size, num_labels)

        self.severity_head = nn.Linear(hidden_size, num_severity)

        self.platform_head = None
        if num_platforms is not None:
            self.platform_head = nn.Linear(hidden_size, num_platforms)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        pooled = outputs.last_hidden_state[:, 0]

        pooled = self.dropout(pooled)

        signal_logits = self.signal_head(pooled)

        severity_logits = self.severity_head(pooled)

        platform_logits = None
        if self.platform_head is not None:
            platform_logits = self.platform_head(pooled)

        return {
            "signal_logits": signal_logits,
            "severity_logits": severity_logits,
            "platform_logits": platform_logits,
        }
```

---

# src/losses.py

```python
import torch
import torch.nn as nn


class FocalBCELoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none"
        )

        probs = torch.sigmoid(logits)

        pt = torch.where(targets == 1, probs, 1 - probs)

        focal = self.alpha * ((1 - pt) ** self.gamma) * bce

        return focal.mean()
```

---

# src/metrics.py

```python
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


def compute_metrics(y_true, y_pred):
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "micro_f1": f1_score(y_true, y_pred, average="micro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
    }


def optimize_thresholds(y_true, y_probs):
    thresholds = []

    for i in range(y_true.shape[1]):
        best_thresh = 0.5
        best_f1 = 0

        for t in np.arange(0.1, 0.9, 0.05):
            preds = (y_probs[:, i] > t).astype(int)
            score = f1_score(y_true[:, i], preds)

            if score > best_f1:
                best_f1 = score
                best_thresh = t

        thresholds.append(best_thresh)

    return thresholds
```

---

# src/train_dapt.py

```python
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

from datasets import load_dataset


MODEL_NAME = "microsoft/deberta-v3-base"


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)


dataset = load_dataset(
    "csv",
    data_files={"train": "data/unlabeled.csv"}
)


def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=512,
    )


encoded = dataset.map(tokenize, batched=True)

collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15,
)

args = TrainingArguments(
    output_dir="checkpoints/dapt",
    per_device_train_batch_size=8,
    learning_rate=5e-5,
    num_train_epochs=5,
    weight_decay=0.01,
    fp16=True,
    save_strategy="epoch",
    logging_steps=100,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=encoded["train"],
    data_collator=collator,
)

trainer.train()

trainer.save_model("checkpoints/dapt-final")
```

---

# src/train_teacher.py

```python
import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tqdm import tqdm

from model import MentalHealthModel
from dataset import MentalHealthDataset
from losses import FocalBCELoss


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

MODEL_NAME = "checkpoints/dapt-final"

BATCH_SIZE = 8
EPOCHS = 5
LR = 2e-5


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


df = pd.read_csv("data/labeled.csv")

train_df, val_df = train_test_split(
    df,
    test_size=0.1,
    random_state=42,
)


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_dataset = MentalHealthDataset(train_df, tokenizer, LABELS)
val_dataset = MentalHealthDataset(val_df, tokenizer, LABELS)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)


model = MentalHealthModel(MODEL_NAME)
model.to(device)


signal_loss_fn = FocalBCELoss()
severity_loss_fn = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

num_training_steps = len(train_loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * num_training_steps),
    num_training_steps=num_training_steps,
)


best_macro_f1 = 0
patience = 2
patience_counter = 0


for epoch in range(EPOCHS):
    model.train()

    total_loss = 0

    for batch in tqdm(train_loader):
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        severity = batch["severity"].to(device)

        outputs = model(input_ids, attention_mask)

        signal_loss = signal_loss_fn(
            outputs["signal_logits"],
            labels,
        )

        severity_loss = severity_loss_fn(
            outputs["severity_logits"],
            severity,
        )

        loss = signal_loss + 0.3 * severity_loss

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} Train Loss: {total_loss / len(train_loader)}")

    model.eval()

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            outputs = model(input_ids, attention_mask)

            probs = torch.sigmoid(outputs["signal_logits"])
            preds = (probs > 0.5).int().cpu()

            all_labels.extend(labels.numpy())
            all_preds.extend(preds.numpy())

    macro_f1 = f1_score(
        all_labels,
        all_preds,
        average="macro"
    )

    print(f"Validation Macro F1: {macro_f1}")

    if macro_f1 > best_macro_f1:
        best_macro_f1 = macro_f1

        torch.save(
            model.state_dict(),
            "checkpoints/best_teacher.pt"
        )

        patience_counter = 0

    else:
        patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping")
            break
```

---

# src/distill_student.py

```python
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from model import MentalHealthModel


TEMPERATURE = 3.0
ALPHA = 0.7


teacher = MentalHealthModel("microsoft/deberta-v3-base")
teacher.load_state_dict(torch.load("checkpoints/best_teacher.pt"))
teacher.eval()

student = MentalHealthModel("distilroberta-base")

optimizer = torch.optim.AdamW(student.parameters(), lr=2e-5)


for batch in train_loader:
    teacher_logits = teacher(
        batch["input_ids"],
        batch["attention_mask"]
    )["signal_logits"]

    student_logits = student(
        batch["input_ids"],
        batch["attention_mask"]
    )["signal_logits"]

    soft_teacher = F.softmax(teacher_logits / TEMPERATURE, dim=-1)

    soft_student = F.log_softmax(student_logits / TEMPERATURE, dim=-1)

    distill_loss = F.kl_div(
        soft_student,
        soft_teacher,
        reduction="batchmean"
    )

    hard_loss = F.binary_cross_entropy_with_logits(
        student_logits,
        batch["labels"]
    )

    loss = ALPHA * distill_loss + (1 - ALPHA) * hard_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

# src/calibrate.py

```python
import numpy as np
from scipy.optimize import minimize


class TemperatureScaler:
    def __init__(self):
        self.temperature = 1.0

    def nll(self, temperature, logits, labels):
        scaled = logits / temperature

        probs = 1 / (1 + np.exp(-scaled))

        eps = 1e-8

        loss = -np.mean(
            labels * np.log(probs + eps)
            + (1 - labels) * np.log(1 - probs + eps)
        )

        return loss

    def fit(self, logits, labels):
        result = minimize(
            lambda t: self.nll(t, logits, labels),
            x0=[1.0],
            bounds=[(0.05, 10.0)]
        )

        self.temperature = result.x[0]

    def transform(self, logits):
        return logits / self.temperature
```

---

# src/export_onnx.py

```python
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer


model = ORTModelForSequenceClassification.from_pretrained(
    "checkpoints/student"
)

model.save_pretrained("onnx/student")


tokenizer = AutoTokenizer.from_pretrained("checkpoints/student")

tokenizer.save_pretrained("onnx/student")
```

---

# src/quantize.py

```python
from onnxruntime.quantization import quantize_dynamic
from onnxruntime.quantization import QuantType


quantize_dynamic(
    model_input="onnx/student/model.onnx",
    model_output="onnx/student/model-int8.onnx",
    weight_type=QuantType.QInt8,
)
```

---

# src/inference.py

```python
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
```

---

# Example Training Commands

## DAPT

```bash
python src/train_dapt.py
```

## Teacher Training

```bash
python src/train_teacher.py
```

## Distillation

```bash
python src/distill_student.py
```

## ONNX Export

```bash
python src/export_onnx.py
```

## Quantization

```bash
python src/quantize.py
```

## Inference

```bash
python src/inference.py
```