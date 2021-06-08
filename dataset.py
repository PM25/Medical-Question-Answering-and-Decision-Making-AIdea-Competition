import os
import re
import csv
import json
import jieba
import numpy as np
from unicodedata import normalize

import torch
from torch.utils.data import Dataset, DataLoader

from utils.regular_exp import *

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# spkr token
# 標點符號標準化
# 語助詞過濾(欸, 啦, 了, 啊, 阿)
# 英文轉中文 No->不 Prep->藥
# 感嘆詞拉掉
# 比較有無標點符號
# 數字轉中文(或拉掉)

# LEO:
# HEAD訓練的過程寫好
# 繁簡體差異確定


def spkr_normalize(spkr: list, spkr_lst: list):
    for spkr_std in spkr_lst:
        if spkr[:-1] in spkr_std or spkr_std in spkr:
            return spkr_std
        elif spkr == "種:":
            return "民眾"
        elif spkr == "耍:":
            return "家屬"
        elif spkr == "生:":
            return "醫師"


def split_sent(article: str, spkr_lst: list):
    diag = []
    res = re.compile(
        r"(護理師[\w*]\s*:|醫師\s*:|民眾\s*:|家屬[\w*]\s*:|個管師\s*:|家屬:|護理師:|醫師A:|藥師:|民眾A:|醫師B:|管師:|不確定人物:|種:|眾:|耍:|生:|家屬、B:|女醫師:)"
    )
    spkr_place_lst = [
        (spkr_place.start(), spkr_place.end()) for spkr_place in res.finditer(article)
    ]
    sent_place_lst = [
        (spkr_place_lst[i][1], spkr_place_lst[i + 1][0])
        for i in range(len(spkr_place_lst) - 1)
    ]
    for (spkr_place, sent_place) in zip(spkr_place_lst[:-1], sent_place_lst):
        spkr_start, spkr_end = spkr_place
        sent_start, sent_end = sent_place
        spkr = article[spkr_start:spkr_end]
        if spkr != "不確定人物:":
            sent = article[sent_start:sent_end]
            if len(sent) != 0:
                diag.append([spkr_normalize(spkr, spkr_lst), sent])

    spkr_start, spkr_end = spkr_place_lst[-1]
    diag.append(
        [spkr_normalize(article[spkr_start:spkr_end], spkr_lst), article[spkr_end:]]
    )


def jieba_cut(article):
    out = []
    for sent in split_sent(article):
        sent = remove_unimportant(sent)
        sent = remove_repeated(sent.upper())
        sent = replace_mapping(sent)
        out.append(list(jieba.cut_for_search(sent)))
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
        # self.tokenizer = {
        #     "Bert": lambda: BertTokenizer.from_pretrained("bert-base-chinese"),
        #     "Roberta": lambda: AutoTokenizer.from_pretrained(
        #         "hfl/chinese-roberta-wwm-ext"
        #     ),
        # }.get(configs["model"], None)()

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
        risk_data = risk_preprocess(risk_file)
        self.data = []
        for risk_datum in risk_data:
            article_id = risk_datum["article_id"]
            article = risk_datum["article"]
            label = risk_datum["label"]
            diag = split_sent(article, configs["spkr"])
            processed_datum = self.process_risk(diag, label)
            processed_datum["article"] = article
            processed_datum["article_id"] = article_id
            self.data.append(processed_datum)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]

    def process_risk(self, raw_article, label):
        out_datum = {
            "article": raw_article,
            "label": torch.tensor(label),
        }
        return out_datum

    def collate_fn(self, data):
        return data


if __name__ == "__main__":
    import yaml

    with open("configs.yaml", "r") as stream:
        configs = yaml.safe_load(stream)

    dataset = risk_dataset(configs, configs["risk_data"])
    non_chinese = []
    for datum in dataset:
        article = datum["article"]
        sent = remove_unimportant(article)
        sent = remove_repeated(sent.upper())
        print(get_non_chinese(sent, digit=False))
        non_chinese.extend(get_non_chinese(sent, digit=False))
        # print(list(jieba.cut_for_search(sent)))
    print(set(non_chinese))
