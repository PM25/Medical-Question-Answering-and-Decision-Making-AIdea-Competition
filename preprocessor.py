import torch
from torch import nn
import torch.nn.functional as F

from transformers import BertModel, AutoModel, AutoTokenizer, BertTokenizer

class PretrainModel(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.bert = {
            "Bert": lambda: BertModel.from_pretrained("bert-base-chinese", output_hidden_states=True),
            "Roberta": lambda: AutoModel.from_pretrained("hfl/chinese-roberta-wwm-ext", output_hidden_states=True),
        }.get(configs["model"], None)()

        self.tokenizer = {
            "Bert": lambda: BertTokenizer.from_pretrained("bert-base-chinese", additional_special_tokens = ['[' + spkr + ']' for spkr in configs['spkr']]),
            "Roberta": lambda: AutoTokenizer.from_pretrained(
                "hfl/chinese-roberta-wwm-ext", additional_special_tokens = ['[' + spkr + ']' for spkr in configs['spkr']]
            ),
        }.get(configs["model"], None)()
        self.bert.resize_token_embeddings(len(self.tokenizer))

        self.max_doc_len = configs["max_document_len"]
        self.use_spkr_token = configs['use_spkr_token']
    
    @property
    def output_size(self):
        return self.bert.config.hidden_size

    def forward(self, raw_articles, device, multi_sent=False):
        if multi_sent:
            return self.pros_multi_sent(raw_articles, device)
        return self.pros_single_sent(raw_articles, device)

    def pros_single_sent(self, raw_articles, device):
        with torch.no_grad():
            input_ids_lst = []
            attention_mask_lst = []
            for raw_article in raw_articles:
                sent = ''.join(['[' + sent[0] + ']' + sent[1] if self.use_spkr_token else sent[1] for sent in raw_article])
                tokenize_data = self.tokenizer(
                    sent,
                    padding="max_length",
                    truncation=True,
                    add_special_tokens=True,
                    max_length=self.max_doc_len,
                    return_tensors="pt",
                )
                input_ids_lst.append(tokenize_data["input_ids"])
                attention_mask_lst.append(tokenize_data["attention_mask"])
        outputs = self.bert(input_ids=torch.cat(input_ids_lst).to(device), attention_mask=torch.cat(attention_mask_lst).to(device))
        cls_emb = outputs.last_hidden_state[:, 0, :]
        mean_emb = outputs.last_hidden_state.mean(dim=1)
        all_emb = outputs.hidden_states
        return cls_emb, mean_emb, all_emb


    def pros_multi_sent(self, raw_articles):
        with torch.no_grad():
            input_ids_lst = []
            attention_mask_lst = []
            sent_emb_list = []
            for raw_article in raw_articles:
                tokenize_data = [self.tokenizer(
                     '[' + sent[0] + ']' + sent[1] if self.use_spkr_token else sent[1],
                    padding="max_length",
                    truncation=True,
                    add_special_tokens=True,
                    max_length=self.max_doc_len,
                    return_tensors="pt",
                ) for sent in raw_article]
                input_ids_lst.append([tokenize_data[i]["input_ids"] for i in range(len(tokenize_data))])
                attention_mask_lst.append([t["attention_mask"] for t in tokenize_data])

            for input_ids, attention_mask in zip(input_ids_lst, attention_mask_lst):
                cls_emb = []
                mean_emb = []
                all_emb = []       
                outputs = self.bert(input_ids=torch.cat(input_ids), attention_mask=torch.cat(attention_mask))
                cls_emb = outputs.last_hidden_state[:, 0, :].cpu()
                mean_emb = outputs.last_hidden_state.mean(dim=1).cpu()
                all_emb = outputs.hidden_states
                sent_emb_list.append({'cls': cls_emb, 'mean': mean_emb, 'all': all_emb})

        return sent_emb_list