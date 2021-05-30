import re
import csv
import json
import unicodedata
import numpy as np
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader


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
                if unicodedata.normalize(
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
        max_sent_len: int = 52,
        max_doc_len: int = 170,
        max_q_len: int = 20,
        max_c_len: int = 18,
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
            for question_data in article:
                for idx, choice_data in enumerate(question_data[1]):
                    document = self.articles[idx]
                    question = "".join(question_data[0])
                    choice = "".join(choice_data[0])
                    answer = choice_data[1]

                    max_input_len = max_doc_len + max_q_len + max_c_len
                    len_d, len_q, len_c = len(document), len(question), len(choice)
                    if len_d + len_q + len_c > max_input_len:
                        doc_len = max(max_input_len - len_q - len_c, max_doc_len)
                        document = document[:doc_len]
                        question_len = max(max_input_len - doc_len - len_c, max_q_len)
                        question = question[:question_len]
                        choice_len = max(
                            max_input_len - doc_len - question_len, max_c_len
                        )
                        choice = choice[:choice_len]

                    tokenize_input = tokenizer(
                        document,
                        question + choice,
                        padding=True,
                        truncation=True,
                        max_length=max_input_len,
                    )

                    self.QA.append(
                        {
                            "document": document,
                            "question": question,
                            "choice": choice,
                            "input_ids": tokenize_input["input_ids"],
                            "token_type_ids": tokenize_input["token_type_ids"],
                            "attention_mask": tokenize_input["attention_mask"],
                            "answer": answer,
                        }
                    )

    def __len__(self):
        return len(self.QA)

    def __getitem__(self, idx: int):
        return self.QA[idx]