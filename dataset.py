import os
import re
import csv
import json
import numpy as np
from unicodedata import normalize
from transformers import AutoTokenizer, BertTokenizer, RobertaTokenizer

import torch
from torch.utils.data import Dataset, DataLoader

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def split_sent(sentence: str):
    first_role_idx = re.search(":", sentence).end(0)
    out = [sentence[:first_role_idx]]

    tmp = sentence[first_role_idx:]
    while tmp:
        res = re.search(r"(護理師[\w*]\s*:|醫師\s*:|民眾\s*:|家屬[\w*]\s*:|個管師\s*:)", tmp)
        if res is None:
            break

        idx = res.start(0)
        idx_end = res.end(0)
        out[-1] = list(out[-1] + tmp[:idx])
        out.append(tmp[idx:idx_end])
        tmp = tmp[idx_end:]

    out[-1] = list(out[-1] + tmp)

    return out


def qa_preprocess(qa_file: str):
    with open(qa_file, "r", encoding="utf-8") as f_QA:
        # One sample of QA
        # [Question, [[Choice_1, Answer_1], [Choice_2, Answer_2], [Choice_3, Answer_3]]]
        data = []
        for datum in json.load(f_QA):
            _id = datum["id"]
            article = datum["text"]
            question = datum["question"]
            question_text = normalize("NFKC", question["stem"])

            choices = []
            answer = ""
            for choice in question["choices"]:
                text = normalize("NFKC", choice["text"])
                label = normalize("NFKC", choice["label"])

                if "answer" not in datum:
                    choices.append([text, label, -1])
                elif label in normalize("NFKC", datum["answer"]):
                    choices.append([text, label, 1])
                    answer = datum["answer"]
                else:
                    choices.append([text, label, 0])

            data.append(
                {
                    "id": _id,
                    "article": article,
                    "question": question_text,
                    "choices": choices,
                }
            )

    return data


def risk_preprocess(risk_file: str):
    with open(risk_file, "r", encoding="utf-8") as f_Risk:
        data = []
        # One smaple of Article
        # [[Sent_1], [Sent_2], ..., [Sent_n]]
        for i, line in enumerate(csv.reader(f_Risk)):
            if i == 0:
                continue
            article_id = eval(line[1])
            article = normalize("NFKC", line[2])
            article = article.replace(" ", "")
            label = normalize("NFKC", line[3])
            label = eval(label) if label.isdigit() else -1

            data.append(
                {
                    "article_id": article_id,
                    "article": article,
                    "label": label,
                }
            )

    return data


class qa_dataset(Dataset):
    def __init__(self, configs, qa_file):
        super().__init__()
        self.max_doc_len = configs["max_document_len"]
        self.max_q_len = configs["max_question_len"]
        self.max_c_len = configs["max_choice_len"]
        self.tokenizer = {
            "Bert": lambda: BertTokenizer.from_pretrained("bert-base-chinese"),
            "Roberta": lambda: AutoTokenizer.from_pretrained(
                "hfl/chinese-roberta-wwm-ext"
            ),
        }.get(configs["model"], None)()

        qa_data = qa_preprocess(qa_file)

        self.data = []
        for idx, qa_datum in enumerate(qa_data):
            processed_datum = self.process_qa(
                qa_datum["article"], qa_datum["question"], qa_datum["choices"]
            )
            processed_datum["id"] = qa_datum["id"]
            processed_datum["article"] = qa_datum["article"]
            processed_datum["question"] = qa_datum["question"]
            self.data.append(processed_datum)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]

    def process_qa(self, raw_doc, raw_question, raw_choices):
        out_datum = {
            "choices": [],
            "label": [],
            "input_ids": [],
            "attention_mask": [],
            "answer": [],
        }

        for idx, choice_ans in enumerate(raw_choices):
            raw_choice = "".join(choice_ans[0])
            label = choice_ans[1]
            answer = choice_ans[2]

            max_input_len = self.max_doc_len + self.max_q_len + self.max_c_len
            len_d, len_q, len_c = len(raw_doc), len(raw_question), len(raw_choice)
            if len_d + len_q + len_c > max_input_len:
                new_doc_len = max(max_input_len - len_q - len_c, self.max_doc_len)
                trunc_document = raw_doc[:new_doc_len]
                new_q_len = max(max_input_len - new_doc_len - len_c, self.max_q_len)
                trunc_question = raw_question[:new_q_len]
                choice_len = max(
                    max_input_len - new_doc_len - new_q_len, self.max_c_len
                )
                trunc_choice = raw_choice[:choice_len]
            else:
                trunc_document = raw_doc
                trunc_question = raw_question
                trunc_choice = raw_choice

            tokenize_data = self.tokenizer(
                trunc_document,
                trunc_question + trunc_choice,
                padding="max_length",
                truncation=True,
                add_special_tokens=True,
                max_length=max_input_len,
                return_tensors="pt",
            )

            out_datum["choices"].append(trunc_choice)
            out_datum["label"].append(label)
            out_datum["input_ids"].append(tokenize_data["input_ids"])
            out_datum["attention_mask"].append(tokenize_data["attention_mask"])
            out_datum["answer"].append(answer)

        out_datum["input_ids"] = torch.cat(out_datum["input_ids"])
        out_datum["attention_mask"] = torch.cat(out_datum["attention_mask"])
        out_datum["answer"] = torch.tensor(out_datum["answer"])

        return out_datum


class risk_dataset(Dataset):
    def __init__(self, configs, risk_file):
        super().__init__()
        self.max_doc_len = configs["max_document_len"]
        self.tokenizer = {
            "Bert": lambda: BertTokenizer.from_pretrained("bert-base-chinese"),
            "Roberta": lambda: AutoTokenizer.from_pretrained(
                "hfl/chinese-roberta-wwm-ext"
            ),
        }.get(configs["model"], None)()

        risk_data = risk_preprocess(risk_file)

        self.data = []
        for risk_datum in risk_data:
            article_id = risk_datum["article_id"]
            article = risk_datum["article"]
            label = risk_datum["label"]

            processed_datum = self.process_risk(article, label)
            processed_datum["article_id"] = article_id
            self.data.append(processed_datum)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]

    def process_risk(self, raw_article, label):
        tokenize_data = self.tokenizer(
            raw_article,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_doc_len,
            return_tensors="pt",
        )

        out_datum = {
            "article": raw_article,
            "input_ids": tokenize_data["input_ids"].flatten(),
            "attention_mask": tokenize_data["attention_mask"].flatten(),
            "label": torch.tensor(label),
        }

        return out_datum