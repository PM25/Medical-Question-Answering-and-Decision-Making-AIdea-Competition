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
            "Bert": lambda: BertTokenizer.from_pretrained("bert-base-chinese"),
            "Roberta": lambda: AutoTokenizer.from_pretrained(
                "hfl/chinese-roberta-wwm-ext"
            ),
        }.get(configs["model"], None)()
        
        if configs["freeze_bert"]:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        self.cls = configs['cls']
        self.max_doc_len = configs["max_document_len"]

    def forward(self, raw_articles):
        with torch.no_grad():
            input_ids_lst = []
            attention_mask_lst = []
            sent_emb_list = []
            for raw_article in raw_articles:
                tokenize_data = [self.tokenizer(
                    sent[1],
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

# 下面只是 example

import yaml
with open("configs.yaml", "r") as stream:
    configs = yaml.safe_load(stream)
from dataset import risk_dataset
from torch.utils.data import DataLoader

train_dataset = risk_dataset(configs, configs["risk_data"])
train_loader = DataLoader(
        train_dataset, batch_size=configs["batch_size"], shuffle=True, num_workers=1, collate_fn=train_dataset.collate_fn
    )
bert = PretrainModel(configs)
for data in train_loader:
    sent_emb_list = bert([d['article'] for d in data][:1])
    print(sent_emb_list[0]['cls'].shape)
    break