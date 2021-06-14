from logging import log
from os import sep
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from .pretrained import PretrainModel


def get_qa_model(configs):
    model_cls_name = configs["model_class"]
    model_cfg = configs[model_cls_name]
    return eval(model_cls_name)(**model_cfg, configs=configs)


class RetrivalBinary(nn.Module):
    def __init__(self, pretrained_cfg, configs, **kwargs):
        super().__init__()
        self.pretrained = PretrainModel(**pretrained_cfg, configs=configs)
        self.predict = nn.Linear(self.pretrained.model.config.hidden_size, 1)

    def infer(self, **kwargs):
        logits = self._forward(**kwargs)
        return logits.cpu().tolist()

    def forward(self, is_answer, **kwargs):
        logits = self._forward(**kwargs, is_answer=is_answer)
        loss = F.binary_cross_entropy(logits, is_answer.float())
        acc = ((logits > 0.5).long() == is_answer).long().cpu().tolist()
        return logits.cpu().tolist(), acc, loss

    def _forward(self, article, question, choice, is_answer, **kwargs):
        device = is_answer.device
        tokenizer = self.pretrained.tokenizer
        cls_token, sep_token = tokenizer.cls_token, tokenizer.sep_token
        sentences = []
        for a, q, c, i in zip(article, question, choice, is_answer):
            sent = f"{cls_token}{a}{sep_token}{q}{sep_token}{c}{sep_token}"
            sentences.append(sent)

        article_features = self.pretrained(sentences, device)
        logits = F.sigmoid(self.predict(article_features).squeeze(-1))
        return logits


class RetrivalMultiple(nn.Module):
    def __init__(self, pretrained_cfg, configs, **kwargs):
        super().__init__()
        self.pretrained = PretrainModel(**pretrained_cfg, configs=configs)
        self.predict = nn.Linear(self.pretrained.model.config.hidden_size, 1)

    def infer(self, **kwargs):
        logits = self._forward(**kwargs)
        logits = logits.view(-1, 3)
        return logits.cpu().tolist()

    def forward(self, is_answer, **kwargs):
        logits = self._forward(**kwargs, is_answer=is_answer)
        logits = logits.view(-1, 3)
        answer = is_answer.view(-1, 3).argmax(dim=-1).view(-1)
        loss = F.cross_entropy(logits, answer)
        acc = (logits.argmax(dim=-1).view(-1) == answer).cpu().tolist()
        return logits.cpu().tolist(), acc, loss

    def _forward(self, question_article, choice_article, question, choice, is_answer, **kwargs):
        device = is_answer.device
        tokenizer = self.pretrained.tokenizer
        cls_token, sep_token = tokenizer.cls_token, tokenizer.sep_token
        sentences = []
        for qa, ca, q, c, i in zip(question_article, choice_article, question, choice, is_answer):
            sent = f"{cls_token}{ca}{sep_token}{c}{sep_token}{q}{sep_token}"
            sentences.append(sent)

        article_features = self.pretrained(sentences, device)
        logits = self.predict(article_features).squeeze(-1)
        return logits


class ClsAttention(nn.Module):
    def __init__(self, pretrained_cfg, hidden_size=128, configs=None, **kwargs):
        super().__init__()
        self.pretrained = PretrainModel(**pretrained_cfg, configs=configs)

        pretrained_size = self.pretrained.model.config.hidden_size
        self.key = nn.Linear(pretrained_size, hidden_size)
        self.value = nn.Linear(pretrained_size, hidden_size)
        self.query = nn.Linear(pretrained_size * 2, hidden_size)
        self.predict = nn.Linear(2 * pretrained_size, 1)

    def infer(self, **kwargs):
        logits = self._forward(**kwargs)
        logits = logits.view(-1, 3)
        return logits.cpu().tolist()

    def forward(self, is_answer, **kwargs):
        logits = self._forward(**kwargs, is_answer=is_answer)
        logits = logits.view(-1, 3)
        answer = is_answer.view(-1, 3).argmax(dim=-1).view(-1)
        loss = F.cross_entropy(logits, answer)
        acc = (logits.argmax(dim=-1).view(-1) == answer).cpu().tolist()
        return logits.cpu().tolist(), acc, loss

    def _forward(self, question_article, choice_article, question, choice, is_answer, **kwargs):
        device = is_answer.device
        tokenizer = self.pretrained.tokenizer
        cls_token, sep_token = tokenizer.cls_token, tokenizer.sep_token
        article_features, question_features, choice_features = [], [], []
        for qa, ca, q, c, i in zip(question_article, choice_article, question, choice, is_answer):
            article_features.append(self.pretrained([f"{cls_token}{qa}{ca}{sep_token}"], embedding_mode="last_seq"))
            question_features.append(self.pretrained([f"{cls_token}{q}{sep_token}"]))
            choice_features.append(self.pretrained([f"{cls_token}{c}{sep_token}"]))

        article_features = pad_sequence([a[0] for a in article_features], batch_first=True)
        question_features = torch.cat(question_features)
        choice_features = torch.cat(choice_features)

        matching_score = torch.bmm(article_features, choice_features.unsqueeze(-1)).squeeze(-1)
        # (batch_size, seqlen)
        attention = matching_score.softmax(dim=-1)
        attended_content = (
            attention.unsqueeze(-1) * article_features
        ).sum(dim=1)
        # (batch_size, seqlen)

        logits = self.predict(torch.cat([attended_content, question_features], dim=-1)).squeeze(-1)
        # (batch_size)
        return logits
