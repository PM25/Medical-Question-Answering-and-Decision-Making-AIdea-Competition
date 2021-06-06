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

    def forward(self, input_ids, attention_mask, answer=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state_cls = outputs[0][:, 0, :]
        logits = self.classifier(last_hidden_state_cls)
        outputs = torch.sigmoid(logits).flatten()

        if answer is not None:
            loss = F.binary_cross_entropy(outputs, answer)
            # outputs = self(input_ids, attention_mask)
            # pred = torch.argmax(outputs, dim=1)
            # answer = torch.argmax(answer, dim=1)
            # return pred, F.cross_entropy(outputs, answer)
            return outputs, loss

        return outputs
