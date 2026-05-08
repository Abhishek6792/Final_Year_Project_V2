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