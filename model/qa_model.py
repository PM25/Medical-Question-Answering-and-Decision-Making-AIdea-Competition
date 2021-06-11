import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers.utils.dummy_pt_objects import LogitsProcessor
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

    def forward(self, role_with_article, question, choice, is_answer, **kwargs):
        device = is_answer.device
        tokenizer = self.pretrained.tokenizer
        sentences = []
        for r, q, c in zip(role_with_article, question, choice):
            sent = f"{tokenizer.cls_token}{''.join(['[' + sent[0] + ']' + sent[1] for sent in r])}{tokenizer.sep_token}{q}{tokenizer.sep_token}{c}{tokenizer.sep_token}"
            sentences.append(sent)

        article_features = self.pretrained(sentences, device)
        logits = self.predict(article_features).squeeze(-1)
        loss = F.binary_cross_entropy(F.sigmoid(logits), is_answer.float())
        return logits, loss


class ClsAttention(nn.Module):
    def __init__(self, pretrained_cfg, hidden_size=128, **kwargs):
        super().__init__()
        self.pretrained = PretrainModel(**pretrained_cfg)

        pretrained_size = self.pretrained.model.config.hidden_size
        self.key = nn.Linear(pretrained_size, hidden_size)
        self.query = nn.Linear(pretrained_size * 2, hidden_size)
        self.predict = nn.Linear(pretrained_size, 1)

    def infer(self, qa_id, article, role, question, choices, **kwargs):
        flatted_article = []
        flatted_role = []
        diag_num = []
        for a, r in zip(article, role):
            flatted_article += a
            flatted_role += r
            diag_num.append(len(a))

        article_features = torch.split(self.pretrained(flatted_article), diag_num)
        article_features = pad_sequence(article_features, batch_first=True)
        # (batch_size, dialogue_num, pretrained_size)
        key_features = self.key(article_features)
        # (batch_size, dialogue_num, hidden_size)

        question_features = self.pretrained(question)
        # (batch_size, pretrained_size)

        flatted_choices = []
        for choice in choices:
            flatted_choices += choice
        choices_features = torch.split(
            self.pretrained(flatted_choices), [3] * len(qa_id)
        )
        choices_features = pad_sequence(choices_features, batch_first=True)
        # (batch_size, 3, pretrained_size)

        qa_features = torch.cat(
            [question_features.unsqueeze(1).expand(-1, 3, -1), choices_features], dim=-1
        )
        # (batch_size, choices_num, pretrained_size)
        query_features = self.query(qa_features)
        # (batch_size, choices_num, hidden_size)

        matching_score = torch.bmm(query_features, key_features.transpose(1, 2))
        # (batch_size, choices_num, dialogue_num)
        attention_mask = (
            ~torch.lt(
                torch.arange(max(diag_num)).unsqueeze(0),
                torch.LongTensor(diag_num).unsqueeze(-1),
            )
        ).long() * -10000000000
        matching_score += attention_mask.unsqueeze(1).to(qa_id.device)
        attention = matching_score.softmax(dim=-1)
        attended_content = (attention.unsqueeze(-1) * article_features.unsqueeze(1)).sum(
            dim=2
        )
        # (batch_size, choices_num, hidden_size)

        logtis = self.predict(attended_content + question_features).squeeze(-1)
        # (batch_size, choices_num)
        return logtis

    def forward(self, answer, **kwargs):
        prediction = self.infer(**kwargs)
        return prediction, F.cross_entropy(prediction, answer)
