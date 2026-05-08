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