import os
import re
import csv
import json
import random
import numpy as np
from tqdm import tqdm
from unicodedata import normalize
from collections import defaultdict
from parsing import text_preprocessing
import torch
from torch.utils.data import Dataset, DataLoader

# import jieba

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# def jieba_cut(article):
#     out = []
#     for sent in split_sent(article):
#         sent = remove_unimportant(sent)
#         sent = remove_repeated(sent.upper())
#         sent = replace_mapping(sent)
#         out.append(list(jieba.cut_for_search(sent)))
#     return out


def spkr_normalize(spkr: str, spkr_lst: list):
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

    diag_pro = []
    for d in diag:
        d[1] = text_preprocessing(d[1])
        if len(d[1]) > 0:
            diag_pro.append([d[0], d[1]])
    return diag_pro


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


def sliding_window(role, article, max_characters=500, overlap_characters=100, **kwargs):
    new_role = []
    new_article = []
    for r, a in zip(role, article):
        remained = a
        while len(remained) > max_characters:
            new_role.append(r)
            new_article.append(remained[:max_characters])
            remained = remained[max_characters - overlap_characters:]
        if len(remained) > 0:
            new_role.append(r)
            new_article.append(remained)
    return new_role, new_article

class qa_dataset(Dataset):
    def __init__(self, configs, qa_file):
        super().__init__()
        self.configs = configs
        self.data = self.preprocess(qa_file)

    def __len__(self):
        return len(self.data)

    def preprocess(self, qa_file: str):
        with open(qa_file, "r", encoding="utf-8") as f_QA:
            data = []
            for datum in json.load(f_QA):
                _id = datum["id"]
                article = normalize("NFKC", datum["text"])
                role, article = zip(*split_sent(article, self.configs["spkr"]))
                role, article = sliding_window(role, article, **self.configs)
                role, article = zip(*[(r, a) for r, a in zip(role, article) if len(a) > self.configs["min_sentence_len"]])

                question = datum["question"]
                question_text = normalize("NFKC", question["stem"])

                answer = datum.get("answer", None)
                if answer is not None:
                    answer = normalize("NFKC", answer)

                choices = []
                answer_ids = []
                for choice_id, choice in enumerate(question["choices"]):
                    text = normalize("NFKC", choice["text"])
                    choices.append(text)

                    label = normalize("NFKC", choice["label"])
                    if label in answer or text in answer:
                        answer_ids.append(choice_id)

                assert len(answer_ids) <= 1

                data.append(
                    {
                        "qa_id": _id,
                        "article": article,
                        "role": role,
                        "question": question_text,
                        "choices": choices,
                        "answer": answer_ids[0] if len(answer_ids) > 0 else -1,
                    }
                )
        return data

    def __getitem__(self, idx: int):
        return self.data[idx]

    @staticmethod
    def collate_fn(samples):
        batch = defaultdict(list)
        keys = samples[0].keys()
        for sample in samples:
            for key in keys:
                batch[key].append(sample[key])

        batch["qa_id"] = torch.LongTensor(batch["qa_id"])
        batch["answer"] = torch.LongTensor(batch["answer"])
        return batch


class qa_binary_dataset(Dataset):
    def __init__(self, configs, qa_file, training=None):
        super().__init__()
        self.training = training
        self.configs = configs
        self.data = self.preprocess(qa_file)

    def __len__(self):
        return len(self.data)
    
    def retrival(self, role_and_dialogue, question_text, choice_text):
        span_size = self.configs["span_size"]
        max_context_size = self.configs["max_context_size"]

        char_level_role = []
        for r, d in role_and_dialogue:
            char_level_role += [r] * len(d)
        full_dialogue = "".join([d for _, d in role_and_dialogue])

        context_range = torch.zeros(len(full_dialogue))
        substrings = []
        for length in reversed(range(1, len(choice_text))):
            for start in range(len(choice_text) - length):
                substrings.append(choice_text[start : start + length])

        for substring in substrings:
            for result in re.finditer(substring, full_dialogue):
                if context_range.sum() + 2 * span_size < max_context_size:
                    context_range[result.start() - span_size : result.end() + span_size] = 1

        if context_range.sum() == 0:
            return "不"

        filtered = [content for inside, content in zip(context_range, zip(char_level_role, full_dialogue)) if inside]
        current_role = filtered[0][0]
        final_article = f"[{filtered[0][0]}]{filtered[0][1]}"
        for content in filtered[1:]:
            role, char = content
            if role != current_role:
                current_role = role
                final_article += f"[{role}]"
            final_article += char
        return final_article

    def preprocess(self, qa_file: str):
        with open(qa_file, "r", encoding="utf-8") as f_QA:
            data = []
            datums = list(json.load(f_QA))

            if self.training is not None:
                sampled = random.choices(datums, k=int(len(datums) * 0.2))
                if self.training:
                    datums = [d for d in datums if d not in sampled]
                else:
                    datums = sampled

            for datum in tqdm(datums):
                _id = datum["id"]
                article = normalize("NFKC", datum["text"])

                question = datum["question"]
                question_text = normalize("NFKC", question["stem"])
                question_text = text_preprocessing(question_text)

                answer = datum.get("answer", None)
                if answer is not None:
                    answer = normalize("NFKC", answer)

                has_answer = False
                for choice_id, choice in enumerate(question["choices"]):
                    choice_text = normalize("NFKC", choice["text"])
                    label = normalize("NFKC", choice["label"])

                    if answer is not None:
                        is_answer = False
                        if label in answer or choice_text in answer:
                            is_answer = True
                            has_answer = True
                    else:
                        is_answer = None

                    role_and_dialogue = split_sent(article, self.configs["spkr"])
                    sub_article = self.retrival(role_and_dialogue, question_text, choice_text)

                    data.append(
                        {
                            "qa_id": _id,
                            "article": sub_article,
                            "question": question_text,
                            "choice": choice_text,
                            "is_answer": is_answer if is_answer is not None else 0,
                            "label": label,
                        }
                    )
                assert answer is None or has_answer

        return data

    def __getitem__(self, idx: int):
        return self.data[idx]

    @staticmethod
    def collate_fn(samples):
        batch = defaultdict(list)
        keys = samples[0].keys()
        for sample in samples:
            for key in keys:
                batch[key].append(sample[key])
        
        batch["is_answer"] = torch.LongTensor(batch["is_answer"])
        return batch


class risk_dataset(Dataset):
    def __init__(self, configs, risk_file, test=True):
        super().__init__()
        self.aug_mode = None
        self.test = test
        self.data = []
        risk_data = risk_preprocess(risk_file)
        for risk_datum in risk_data:
            article_id = risk_datum["article_id"]
            article = risk_datum["article"]

            label = risk_datum["label"]
            diag = split_sent(article, configs["spkr"])
            processed_datum = self.process_risk(diag, label)
            processed_datum["article_id"] = article_id
            self.data.append(processed_datum)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        data = self.data[idx]
        if self.aug_mode is None:
            article_sample = data["article"]
        else:
            article_sample = eval(f'self.{self.aug_mode}_sent_aug(data["article"])')
        return {
            "label": data["label"],
            "article_id": data["article_id"],
            "article": article_sample,
        }

    def process_risk(self, raw_article, label):
        out_datum = {
            "article": raw_article,
            "label": torch.tensor(label),
        }
        return out_datum

    def long_sent_aug(self, data):
        idx_med = random.choice([i for i in range(len(data)) if len(data[i][1]) > 10])
        if self.test:
            return (
                self.backward_sample(data, idx_med, 252)
                + [data[idx_med]]
                + self.forward_sample(data, idx_med, 252)
            )
        else:
            return (
                self.backward_sample(data, idx_med, random.randint(200, 252))
                + [data[idx_med]]
                + self.forward_sample(data, idx_med, random.randint(200, 252))
            )

    def last_sent_aug(self, data):
        idx_last = len(data) - 1
        if self.test:
            return self.backward_sample(data, idx_last, 505) + [data[idx_last]]
        else:
            return self.backward_sample(data, idx_last, random.randint(400, 505)) + [
                data[idx_last]
            ]

    def first_sent_aug(self, data):
        idx_first = 0
        if self.test:
            return [data[idx_first]] + self.forward_sample(data, idx_first, 505)
        else:
            return [data[idx_first]] + self.forward_sample(
                data, idx_first, random.randint(400, 505)
            )

    def forward_sample(self, data, idx, thr):
        count = 0
        samples = []
        for i in range(idx + 1, len(data)):
            count += len(data[i][1])
            if count <= thr:
                samples.append(data[i])
            else:
                break
        return samples

    def backward_sample(self, data, idx, thr):
        count = 0
        samples = []
        for i in range(idx - 1, -1, -1):
            count += len(data[i][1])
            if count <= thr:
                samples.append(data[i])
            else:
                break
        samples.reverse()
        return samples

    def collate_fn(self, data):
        return data