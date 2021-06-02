import yaml
import torch
from torch import nn
import torch.nn.functional as F

from transformers import BertModel, BertForSequenceClassification

with open("configs.yaml", "r") as stream:
    configs = yaml.safe_load(stream)


class BertClassifier(nn.Module):
    def __init__(self, freeze_bert=True):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-chinese")
        D_in, hidden_dim, D_out = 768, configs["hidden_dim"], 1

        hidden_layers = []
        hidden_layers.append(nn.Linear(D_in, hidden_dim))
        for _ in range(configs["n_cls_layers"]):
            hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
            hidden_layers.append(nn.ReLU())
            hidden_layers.append(nn.Dropout(configs["dropout"]))
        hidden_layers.append(nn.Linear(hidden_dim, D_out))
        self.classifier = nn.Sequential(*hidden_layers)

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

    def pred_and_loss(self, input_ids, attention_mask, answer):
        outputs = self(input_ids, attention_mask)
        preds = torch.argmax(outputs, dim=1)
        answer = answer.reshape(-1)
        return preds, F.binary_cross_entropy(outputs.reshape(-1), answer)
        # outputs = self(input_ids, attention_mask)
        # pred = torch.argmax(outputs, dim=1)
        # answer = torch.argmax(answer, dim=1)
        # return pred, F.cross_entropy(outputs, answer)

    def pred_label(self, input_ids, attention_mask, labels):
        outputs = self(input_ids, attention_mask)
        preds = torch.argmax(outputs, dim=1)
        _pred_label = [label[pred] for pred, label in zip(preds, zip(*labels))]
        return _pred_label