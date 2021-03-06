from typing import List

import torch
from torch import nn
import torch.nn.functional as F

from transformers import BertModel, AutoModel, AutoTokenizer, BertTokenizer, BertForPreTraining


class PretrainModel(nn.Module):
    def __init__(
        self,
        pretrained,
        trainable_from: int = -1,
        embedding_mode: str = "last_cls",
        max_tokens: int = 10000,
        configs=None,
        **kwargs,
    ):
        super().__init__()
        self.embedding_mode = embedding_mode  # last_cls, pooler_output, all_cls
        self.max_tokens = max_tokens
        special_tokens = ["[" + spkr + "]" for spkr in configs["spkr"]]

        TOKENIZERS = {
            "Bert": lambda: BertTokenizer.from_pretrained(
                "ckiplab/bert-base-chinese",
                additional_special_tokens=special_tokens,
            ),
            "Roberta": lambda: AutoTokenizer.from_pretrained(
                "hfl/chinese-roberta-wwm-ext",
                additional_special_tokens=special_tokens,
            ),
            "Medical": lambda: AutoTokenizer.from_pretrained(
                "hfl/chinese-roberta-wwm-ext",
                additional_special_tokens=special_tokens,
            ),
        }

        PRETRAINED_MODELS = {
            "Bert": lambda: BertModel.from_pretrained(
                "bert-base-chinese", output_hidden_states=True
            ),
            "Roberta": lambda: AutoModel.from_pretrained(
                "hfl/chinese-roberta-wwm-ext", output_hidden_states=True
            ),
            "Medical": lambda: BertForPreTraining.from_pretrained(
                configs["medical_bert_dir"],
                from_tf=True,
            ).bert,
        }

        self.tokenizer = TOKENIZERS[pretrained]()
        self.model = PRETRAINED_MODELS[pretrained]()
        self.model.resize_token_embeddings(len(self.tokenizer))

        if self.embedding_mode == "all_cls":
            self.weights = nn.Parameter(
                torch.zeros(self.model.config.num_hidden_layers + 1)
            )

        trainable = False
        for name, para in self.model.named_parameters():
            if "embeddings" in name:
                para.requires_grad = True
            if f"layer.{trainable_from}" in name:
                trainable = True
            para.requires_grad = trainable

    @property
    def output_size(self):
        return self.model.config.hidden_size

    def forward(self, sentences: List[str], device="cuda", embedding_mode = "last_cls"):
        embedding_mode = embedding_mode or self.embedding_mode
        tokenizer_result = self.tokenizer(
            sentences,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
            add_special_tokens=False,
        )
        for key in list(tokenizer_result.keys()):
            if isinstance(tokenizer_result[key], torch.Tensor):
                tokenizer_result[key] = tokenizer_result[key].to(device)

        input_ids = tokenizer_result["input_ids"]
        attention_mask = tokenizer_result["attention_mask"]
        minibatch_size = self.max_tokens // input_ids.size(1)

        outputs = []
        for start in range(0, input_ids.size(0), minibatch_size):
            minibatch = {
                k: v[start : start + minibatch_size]
                for k, v in tokenizer_result.items()
            }
            model_result = self.model(**minibatch, output_hidden_states=True)

            if embedding_mode == "last_cls":
                output = model_result.last_hidden_state[:, 0, :]
            elif embedding_mode == "all_cls":
                all_hidden_states = torch.stack(model_result.hidden_states, dim=0)
                all_cls = all_hidden_states[:, :, 0, :]
                output = (F.softmax(self.weights, dim=-1).view(-1, 1, 1) * all_cls).sum(
                    dim=0
                )
            elif embedding_mode == "pooler_output":
                output = model_result.pooler_output
            elif embedding_mode == "mean":
                output = model_result.last_hidden_state.mean(dim=1)
            elif embedding_mode == "last_seq":
                output = model_result.last_hidden_state
            else:
                print(embedding_mode)
                raise ValueError

            # output: (minibatch, hidden_size)
            outputs.append(output)

        return torch.cat(outputs, dim=0)
