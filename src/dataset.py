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