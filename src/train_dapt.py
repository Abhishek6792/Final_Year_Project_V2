from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

from datasets import load_dataset


MODEL_NAME = "FacebookAI/roberta-base"


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)


dataset = load_dataset(
    "csv",
    data_files={"train": "annotate_data/Dataset/Unlabled_Data/Ayush_2000.csv"}
)


def tokenize(batch):
    return tokenizer(
        batch["selftext"],
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