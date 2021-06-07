from typing import Dict
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from .pretrained import PretrainModel
from .pooling import get_pooler


class Classifier(nn.Module):
    def __init__(
        self, pretrained_cfg: Dict, project_size: int, diag_pooler_cfg: Dict, **kwargs
    ):
        super().__init__()
        self.pretrained_model = PretrainModel(**pretrained_cfg)
        self.projector = nn.Linear(self.pretrained_model.output_size, project_size)
        self.diag_pooler = get_pooler(project_size, diag_pooler_cfg)
        self.predictor = nn.Linear(self.diag_pooler.output_size, 1)

    def forward(self, diags, diags_len, roles, **kwargs):
        embeddings = self.pretrained_model(diags, diags_len.device)
        embeddings = embeddings.split(diags_len.tolist())
        embeddings = pad_sequence(embeddings, batch_first=True)
        embeddings = self.projector(embeddings)
        pooled_embbedings = self.diag_pooler(embeddings, diags_len)
        logits = self.predictor(pooled_embbedings).view(-1)
        return F.sigmoid(logits)
