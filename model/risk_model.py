from typing import Dict
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from .pretrained import PretrainModel
from .pooling import get_pooler

NUM_ROLES = 5

class Classifier(nn.Module):
    def __init__(
        self, pretrained_cfg: Dict, project_size: int, pooler_cfg: Dict, **kwargs
    ):
        super().__init__()
        self.pretrained_model = PretrainModel(**pretrained_cfg)
        self.projector = nn.Linear(self.pretrained_model.output_size, project_size)
        self.role_embedding = nn.Embedding(NUM_ROLES, project_size)
        self.pooler = get_pooler(project_size, pooler_cfg)
        self.predictor = nn.Linear(self.pooler.output_size, 1)

    def forward(self, diags, diags_len, roles, **kwargs):
        content_embeddings = self.pretrained_model(diags, diags_len.device)
        content_embeddings = self.projector(content_embeddings)
        role_embeddings = self.role_embedding(roles)
        embeddings = content_embeddings + role_embeddings

        embeddings = embeddings.split(diags_len.tolist())
        diags_embeddings = pad_sequence(embeddings, batch_first=True)

        pooled_embbedings = self.pooler(diags_embeddings, diags_len)
        logits = self.predictor(pooled_embbedings).view(-1)
        return F.sigmoid(logits)
