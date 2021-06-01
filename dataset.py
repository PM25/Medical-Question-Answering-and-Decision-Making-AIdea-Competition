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
            answer = unicodedata.normalize("NFKC", line[3])
            if answer.isdigit():
                risk.append(eval(answer))
            else:
                risk.append(-1)

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
                    temp.append([text, -1])
                elif unicodedata.normalize(
                    "NFKC", choice["label"]
                ) in unicodedata.normalize("NFKC", data["answer"]):
                    temp.append([text, 1])
                    answer = data["answer"]
                else:
                    temp.append([text, 0])
            question_text = list(unicodedata.normalize("NFKC", question["stem"]))
            if data["article_id"] != article_id:
                qa.append([])
                article_id = data["article_id"]
            qa[-1].append([data["article_id"], question_text, temp])

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
        self.max_doc_len = max_doc_len
        self.max_q_len = max_q_len
        self.max_c_len = max_c_len
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

        articles, risk, qa = preprocess(qa_file, risk_file)

        # `risk` shape: [N]
        self.risk = np.array(risk)

        self.articles = []
        for document in articles:
            document = "".join(["".join(sen) for sen in document])
            self.articles.append(document)

        self.data = []
        for idx, article in enumerate(qa):
            document = self.articles[idx]
            risk_datum = self.process_risk(document, self.risk[idx])

            for question_data in article:
                article_id = question_data[0]
                question = "".join(question_data[1])
                choices = question_data[2]

                qa_datum = self.process_qa(document, question, choices)

                # FIXME: risk datum duplicate (should be 1 per document)
                self.data.append(
                    {
                        "document_id": article_id,
                        "document": document,
                        "question": question,
                        "qa_choices": qa_datum["choices"],
                        "qa_input_ids": qa_datum["input_ids"],
                        "qa_attention_mask": qa_datum["attention_mask"],
                        "qa_answer": qa_datum["answer"],
                        "risk_input_ids": risk_datum["input_ids"],
                        "risk_attention_mask": risk_datum["attention_mask"],
                        "risk_answer": risk_datum["answer"],
                    }
                )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]

    def process_qa(self, raw_doc, raw_question, raw_choices):
        out_datum = {"choices": [], "input_ids": [], "attention_mask": [], "answer": []}

        for idx, choice_ans in enumerate(raw_choices):
            raw_choice = "".join(choice_ans[0])
            answer = choice_ans[1]

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
            out_datum["input_ids"].append(tokenize_data["input_ids"])
            out_datum["attention_mask"].append(tokenize_data["attention_mask"])
            out_datum["answer"].append(answer)

        out_datum["input_ids"] = torch.cat(out_datum["input_ids"])
        out_datum["attention_mask"] = torch.cat(out_datum["attention_mask"])
        out_datum["answer"] = torch.tensor(out_datum["answer"])

        return out_datum

    def process_risk(self, raw_doc, risk):
        tokenize_data = self.tokenizer(
            raw_doc,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_doc_len,
            return_tensors="pt",
        )

        out_datum = {
            "document": raw_doc,
            "input_ids": tokenize_data["input_ids"].flatten(),
            "attention_mask": tokenize_data["attention_mask"].flatten(),
            "answer": torch.tensor(risk),
        }

        return out_datum
