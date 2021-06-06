import torch
from torch import nn
import torch.nn.functional as F

from transformers import BertModel, AutoModel


class BertClassifier(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.bert = {
            "Bert": lambda: BertModel.from_pretrained("ckiplab/bert-base-chinese"),
            "Roberta": lambda: AutoModel.from_pretrained("hfl/chinese-roberta-wwm-ext"),
        }.get(configs["model"], None)()
        D_in, hidden_dim, D_out = 768, configs["hidden_dim"], 1

        hidden_layers = []
        hidden_layers.append(nn.Linear(D_in, hidden_dim))
        for _ in range(configs["n_cls_layers"]):
            hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
            hidden_layers.append(nn.ReLU())
            hidden_layers.append(nn.Dropout(configs["dropout"]))
        hidden_layers.append(nn.Linear(hidden_dim, D_out))
        self.classifier = nn.Sequential(*hidden_layers)

        if configs["freeze_bert"]:
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
    def __init__(self, configs):
        super().__init__()
        self.bert_classifier = BertClassifier(configs)

    def forward(self, input_ids, attention_mask, answer=None):
        # input_ids Shape: [Batch_Size, num_choices, input_length]
        # attention Shape: [Batch_Size, num_choices, input_length]

        outputs = []
        for choice in range(input_ids.shape[1]):
            _input_ids = input_ids[:, choice, :]
            _attention_mask = attention_mask[:, choice, :]
            outputs.append(self.bert_classifier(_input_ids, _attention_mask))
        outputs = torch.cat(outputs, dim=-1)
        preds = torch.argmax(outputs, dim=1)

        if answer is not None:
            answer = answer.reshape(-1)
            loss = F.binary_cross_entropy(outputs.reshape(-1), answer)
            return preds, loss

        return preds

    def pred_label(self, input_ids, attention_mask, labels):
        preds = self(input_ids, attention_mask)
        _pred_label = [label[pred] for pred, label in zip(preds, zip(*labels))]

        return _pred_label