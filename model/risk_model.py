import yaml
import torch
from torch import nn
import torch.nn.functional as F

from transformers import BertModel

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
        return torch.sigmoid(logits).flatten()

    def pred_and_loss(self, input_ids, attention_mask, answer):
        outputs = self(input_ids, attention_mask)
        return outputs, F.binary_cross_entropy(outputs, answer)
        # outputs = self(input_ids, attention_mask)
        # pred = torch.argmax(outputs, dim=1)
        # answer = torch.argmax(answer, dim=1)
        # return pred, F.cross_entropy(outputs, answer)