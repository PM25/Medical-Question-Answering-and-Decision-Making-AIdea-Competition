import re
import csv
import json
import yaml
import unicodedata
import numpy as np
from transformers import BertTokenizer

import torch
from torch.utils.data import Dataset, DataLoader

with open("configs.yaml", "r") as stream:
    configs = yaml.safe_load(stream)


"""
Here we will do preprocessing on the dataset.
Something needs to be done here :
1. Read the file in.
2. Separate the article, question, answer.
3. Used PAD to round each sentence into the same length
"""


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


def preprocess(qa_file: str, risk_file: str):
    with open(qa_file, "r", encoding="utf-8") as f_QA, open(
        risk_file, "r", encoding="utf-8"
    ) as f_Risk:
        article = []
        risk = []

        # One smaple of Article
        # [[Sent_1], [Sent_2], ..., [Sent_n]]
        for i, line in enumerate(csv.reader(f_Risk)):
            if i == 0:
                continue
            text = unicodedata.normalize("NFKC", line[2])
            text = text.replace(" ", "")
            article.append(split_sent(text))
            risk.append(int(line[3]))

        # One sample of QA
        # [Question, [[Choice_1, Answer_1], [Choice_2, Answer_2], [Choice_3, Answer_3]]]
        article_id = 1
        qa = []
        qa.append([])
        for data in json.load(f_QA):
            question = data["question"]
            temp = []
            answer = ""
            for choice in question["choices"]:
                text = list(unicodedata.normalize("NFKC", choice["text"]))
                if "answer" not in data:
                    temp.append([text, None])
                elif unicodedata.normalize(
                    "NFKC", choice["label"]
                ) in unicodedata.normalize("NFKC", data["answer"]):
                    temp.append([text, 1])
                    answer = data["answer"]
                else:
                    temp.append([text, 0])
            question_text = list(unicodedata.normalize("NFKC", question["stem"]))
            if not answer:
                print("".join(question_text))
                print(question["choices"])
                print(unicodedata.normalize("NFKC", data["answer"]))
                print(question["choices"][2]["label"])
                continue
            if data["article_id"] != article_id:
                qa.append([])
                article_id = data["article_id"]

            qa[-1].append([question_text, temp])

    return article, risk, qa


class all_dataset(Dataset):
    def __init__(
        self,
        qa_file: str,
        risk_file: str,
        max_doc_len: int = configs["max_document_len"],
        max_q_len: int = configs["max_question_len"],
        max_c_len: int = configs["max_choice_len"],
    ):
        super().__init__()
        tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        articles, risk, qa = preprocess(qa_file, risk_file)

        # `risk` shape: [N]
        self.risk = np.array(risk)

        self.articles = []
        for document in articles:
            document = "".join(["".join(sen) for sen in document])
            self.articles.append(document)

        self.QA = []
        for idx, article in enumerate(qa):
            document = self.articles[idx]
            for question_data in article:
                question = "".join(question_data[0])
                QA_datum = {
                    "document": document,
                    "question": question,
                    "choices": [],
                    "input_ids": [],
                    "attention_mask": [],
                    "answer": [],
                }
                for idx, choice_data in enumerate(question_data[1]):
                    choice = "".join(choice_data[0])
                    answer = choice_data[1]

                    max_input_len = max_doc_len + max_q_len + max_c_len
                    len_d, len_q, len_c = len(document), len(question), len(choice)
                    if len_d + len_q + len_c > max_input_len:
                        doc_len = max(max_input_len - len_q - len_c, max_doc_len)
                        trunc_document = document[:doc_len]
                        question_len = max(max_input_len - doc_len - len_c, max_q_len)
                        trunc_question = question[:question_len]
                        choice_len = max(
                            max_input_len - doc_len - question_len, max_c_len
                        )
                        trunc_choice = choice[:choice_len]
                    else:
                        trunc_document = document
                        trunc_question = question
                        trunc_choice = choice

                    tokenize_data = tokenizer(
                        trunc_document,
                        trunc_question + trunc_choice,
                        padding="max_length",
                        truncation=True,
                        add_special_tokens=True,
                        max_length=max_input_len,
                        return_tensors="pt",
                    )

                    QA_datum["choices"].append(trunc_choice)
                    QA_datum["input_ids"].append(tokenize_data["input_ids"])
                    QA_datum["attention_mask"].append(tokenize_data["attention_mask"])
                    QA_datum["answer"].append(answer)

                QA_datum["input_ids"] = torch.cat(QA_datum["input_ids"])
                QA_datum["attention_mask"] = torch.cat(QA_datum["attention_mask"])
                QA_datum["answer"] = torch.tensor(QA_datum["answer"])

                self.QA.append(QA_datum)

    def __len__(self):
        return len(self.QA)

    def __getitem__(self, idx: int):
        return self.QA[idx]


class risk_dataset(Dataset):
    def __init__(
        self,
        qa_file: str,
        risk_file: str,
        max_doc_len: int = 512,
    ):
        super().__init__()
        tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        articles, risk, qa = preprocess(qa_file, risk_file)

        # `risk` shape: [N]
        self.risk = np.array(risk)

        self.articles = []
        for document in articles:
            document = "".join(["".join(sen) for sen in document])
            self.articles.append(document)

        self.QA = []
        for idx, article in enumerate(qa):
            document = self.articles[idx]
            QA_datum = {
                "document": document,
                "input_ids": None,
                "attention_mask": None,
                "answer": torch.tensor(self.risk[idx]),
            }

            tokenize_data = tokenizer(
                document,
                padding="max_length",
                truncation=True,
                add_special_tokens=True,
                max_length=max_doc_len,
                return_tensors="pt",
            )
            QA_datum["input_ids"] = tokenize_data["input_ids"].flatten()
            QA_datum["attention_mask"] = tokenize_data["attention_mask"].flatten()
            self.QA.append(QA_datum)

    def __len__(self):
        return len(self.QA)

    def __getitem__(self, idx: int):
        return self.QA[idx]