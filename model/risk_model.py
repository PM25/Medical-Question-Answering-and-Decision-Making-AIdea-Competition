import torch
from torch import nn
import torch.nn.functional as F

# from transformers import BertModel, AutoModel
from preprocessor import PretrainModel

class BertClassifier(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.latent_mode = configs['latent_mode']
        self.pretrained_model = PretrainModel(configs)
        
        D_in, hidden_dim, D_out = 768, configs["hidden_dim"], 1
        
        hidden_layers = []
        hidden_layers.append(nn.Linear(D_in, hidden_dim))
        act = configs['activation']
        hidden_layers.append(eval(f'nn.{act}()'))
        if act != 'GELU':
            hidden_layers.append(nn.Dropout(configs["dropout"]))
        hidden_layers.append(nn.Linear(hidden_dim, D_out))
        self.classifier = nn.Sequential(*hidden_layers)

        if configs["warmup_epoch"] > 0:
            for param in self.pretrained_model.bert.parameters():
                param.requires_grad = False

    def forward(self, inputs, device, answer=None):
        cls_emb, mean_emb, all_emb = self.pretrained_model(inputs, device)
        logits = self.classifier(eval(f'{self.latent_mode}_emb'))
        outputs = torch.sigmoid(logits).flatten()

        if answer is not None:
            loss = F.binary_cross_entropy(outputs, answer)
            return outputs, loss
        return outputs