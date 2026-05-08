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