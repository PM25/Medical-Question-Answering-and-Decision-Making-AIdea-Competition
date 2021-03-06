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
import opencc
# import jieba

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def find_relevant_sent(key, diag):
    idx_cand = []
    for i in range(len(diag)):
        spkr, sent = diag[i]
        if key in sent:
            idx_cand.append(i)
    return idx_cand

def expand_relevant_sent(idx_cand, diag):
    idx_cand_exp = []
    for i in idx_cand:
        idx_cand_exp.append(max(i - 1, 0))
        idx_cand_exp.append(i)
        idx_cand_exp.append(min(i + 1, len(diag) - 1))
    idx_cand_exp = sorted(list(set(idx_cand_exp)))
    return idx_cand_exp

def get_diag_subset(idx_cand_exp, diag):
    spkr_pre = ''
    ctx_pre = ''
    diag_subset = []
    for i in idx_cand_exp:
        if diag[i][0] != spkr_pre or diag[i][1] != ctx_pre:
            diag_subset.append([diag[i][0], diag[i][1]])
        spkr_pre = diag[i][0]
        ctx_pre = diag[i][1]
    return diag_subset

def count_word(diag):
    count = 0
    for i in range(len(diag)):
        spkr, sent = diag[i]
        count += (len(spkr) + len(sent) + 1)
    return count

def diag_prune(diag_subset, thr):
    if len(diag_subset) > 1 and count_word(diag_subset) > thr:
        diag_subset_prune = []
        count = 0
        for d in diag_subset:
            spkr, sent = d
            count += (len(spkr) + len(sent) + 1)
            if count < 470:
                diag_subset_prune.append(d)
            else:
                diag_subset = diag_subset_prune
                break
    elif len(diag_subset) == 1 and count_word(diag_subset) > thr:
        spkr, sent = diag_subset[0]
        sent = sent[:thr-2]
        diag_subset[0] = spkr, sent
    return diag_subset

def spkr_normalize(spkr: str, spkr_lst: list):
    for spkr_std in spkr_lst:
        if spkr[:-1] in spkr_std or spkr_std in spkr:
            return spkr_std
        elif spkr == "???:":
            return "??????"
        elif spkr == "???:":
            return "??????"
        elif spkr == "???:":
            return "??????"


def split_sent(article: str, spkr_lst: list):
    diag = []
    res = re.compile(
        r"(?????????[\w*]\s*:|??????\s*:|??????\s*:|??????[\w*]\s*:|?????????\s*:|??????:|?????????:|??????A:|??????:|??????A:|??????B:|??????:|???????????????:|???:|???:|???:|???:|?????????B:|?????????:)"
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
        if spkr != "???????????????:":
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


class qa_retrival_dataset(Dataset):
    def __init__(self, configs, training=None):
        super().__init__()
        self.training = training
        self.configs = configs
        self.term_count = {}

        for line in open("data/df.txt", "r"):
            term, count = line.strip().split(' ')
            self.term_count[term] = int(count)

        self.converter = None
        if self.configs.get("t2s") is not None:
            self.converter = opencc.OpenCC(self.configs["t2s"])

    
    def retrival1(self, role_and_dialogue, query_text, spkr_mode=None):
        span_size = self.configs["span_size"]
        max_context_size = self.configs["max_context_size"]

        char_level_role = []
        for r, d in role_and_dialogue:
            char_level_role += [r] * len(d)
        full_dialogue = "".join([d for _, d in role_and_dialogue])

        context_range = torch.zeros(len(full_dialogue))
        substrings = []
        for length in reversed(range(1, len(query_text))):
            for start in range(len(query_text) - length):
                substrings.append(query_text[start : start + length])

        for substring in substrings:
            for result in re.finditer(substring, full_dialogue):
                if context_range.sum() + 2 * span_size < max_context_size:
                    context_range[result.start() - span_size : result.end() + span_size] = 1

        if context_range.sum() == 0:
            return ""

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

    def retrival2(self, role_and_dialogue, question_text, choice_text, spkr_mode=None):
        idx_range = []
        substrings = []
        for length in reversed(range(1, len(choice_text))):
            for start in range(len(choice_text) - length):
                term = choice_text[start : start + length]
                if (not (term in self.term_count and self.term_count[term] > 1300)) or term == choice_text.replace('???', ''):
                    substrings.append(term)

            for term in substrings:
                idx_range += find_relevant_sent(term, role_and_dialogue)
            
            if len(idx_range) > 0 and length <= 3:
                idx_range = expand_relevant_sent(idx_range, role_and_dialogue)
                break

        diag_subset = get_diag_subset(idx_range, role_and_dialogue)
        thr = 502 - len(question_text) - len(choice_text)
        diag_subset = diag_prune(diag_subset, thr)

        if self.converter is not None:
            diag_subset = [(item[0], self.converter.convert(item[1])) for item in diag_subset]

        if spkr_mode is None:
            return ''.join([d[1] for d in diag_subset])
        
        elif spkr_mode == 'token':
            return ''.join([f'[{d[0]}]{d[1]}' for d in diag_subset])
        
        elif spkr_mode == 'content':
            return ''.join([f'{d[0]}???{d[1]}' for d in diag_subset])

class qa_binary_dataset(qa_retrival_dataset):
    def __init__(self, qa_file, **kwargs):
        super().__init__(**kwargs)
        self.data = self.preprocess(qa_file)

    def __len__(self):
        return len(self.data)
    
    def preprocess(self, qa_file: str):
        with open(qa_file, "r", encoding="utf-8") as f_QA:
            data = []
            datums = list(json.load(f_QA))

            if self.training is not None:
                random.seed(self.configs["seed"])
                sampled = random.choices(datums, k=int(len(datums) * self.configs["val_size"]))
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
                    sub_article = self.retrival2(role_and_dialogue, question_text, choice_text, self.configs["spkr_mode"])

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


class qa_multiple_dataset(qa_retrival_dataset):
    def __init__(self, qa_file, **kwargs):
        super().__init__(**kwargs)
        self.data = self.preprocess(qa_file)

    def __len__(self):
        return len(self.data)

    def preprocess(self, qa_file: str):
        with open(qa_file, "r", encoding="utf-8") as f_QA:
            data = []
            datums = list(json.load(f_QA))

            if self.training is not None:
                random.seed(self.configs["seed"])
                sampled = random.choices(datums, k=int(len(datums) * self.configs["val_size"]))
                if self.training:
                    datums = [d for d in datums if d not in sampled]
                else:
                    datums = sampled

            for i, datum in enumerate(tqdm(datums)):
                # if i > 10:
                #     break
                _id = datum["id"]
                article = normalize("NFKC", datum["text"])

                question = datum["question"]
                question_text = normalize("NFKC", question["stem"])
                question_text = text_preprocessing(question_text)

                answer = datum.get("answer", None)
                if answer is not None:
                    answer = normalize("NFKC", answer)

                data_point = []
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
                    
                    choice_text = text_preprocessing(choice_text)
                    role_and_dialogue = split_sent(article, self.configs["spkr"])
                    retrival_fn = eval(self.configs["retrival_fn"])
                    question_article = retrival_fn(role_and_dialogue, question_text, self.configs["spkr"])
                    choice_article = retrival_fn(role_and_dialogue, choice_text, self.configs["spkr"])

                    data_point.append(
                        {
                            "qa_id": _id,
                            "question_article": question_article,
                            "choice_article": choice_article,
                            "question": question_text,
                            "choice": choice_text,
                            "is_answer": is_answer if is_answer is not None else 0,
                            "label": label,
                        }
                    )
                assert answer is None or has_answer
                data.append(data_point)

        return data

    def __getitem__(self, idx: int):
        return self.data[idx]

    @staticmethod
    def collate_fn(samples):
        flatted_samples = []
        for sample in samples:
            flatted_samples += sample
        samples = flatted_samples

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