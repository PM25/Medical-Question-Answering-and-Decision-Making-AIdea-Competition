import torch
from torch import nn
import torch.nn.functional as F

from transformers import (
    BertModel,
    get_linear_schedule_with_warmup,
)


class BertClassifier(nn.Module):
    def __init__(self, freeze_bert=True):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-chinese")
        D_in, H, D_out = 768, 50, 1
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(H, D_out),
        )
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state_cls = outputs[0][:, 0, :]
        logits = self.classifier(last_hidden_state_cls)
        return torch.sigmoid(logits)

    def loss_func(self, input_ids, attention_mask, answer):
        pred = self(input_ids, attention_mask).reshape(-1)
        answer = answer.reshape(-1)
        return F.binary_cross_entropy(pred, answer)


class QA_Model(nn.Module):
    def __init__(self, freeze_bert=True):
        super().__init__()
        self.bert_classifier = BertClassifier(freeze_bert)

    def forward(self, input_ids, attention_mask):
        # input_ids Shape: [Batch_Size, num_choices, input_length]
        # attention Shape: [Batch_Size, num_choices, input_length]

        outputs = []
        for choice in range(input_ids.shape[1]):
            _input_ids = input_ids[:, choice, :]
            _attention_mask = attention_mask[:, choice, :]
            outputs.append(self.bert_classifier(_input_ids, _attention_mask))
        outputs = torch.cat(outputs, dim=-1)
        return outputs

    def loss_func(self, input_ids, attention_mask, answer):
        # pred = self(input_ids, attention_mask).reshape(-1)
        # answer = answer.reshape(-1)
        pred = self(input_ids, attention_mask)
        answer = torch.argmax(answer, dim=1)
        return F.cross_entropy(pred, answer)
        # return F.binary_cross_entropy(pred, answer)