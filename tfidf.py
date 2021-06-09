import csv
import yaml
import jieba
import numpy as np
import jieba.analyse
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.tensorboard import SummaryWriter

from dataset import risk_dataset

jieba.analyse.set_stop_words("data/stopwords.txt")

with open("data/stopwords.txt", "r") as f:
    stop_words = [line.strip() for line in f.readlines()]

add_words = [
    "護理師",
    "個管師",
    "醫師",
    "民眾",
    "家屬",
    "HIV",
    "關係",
    "沒關係",
    "性行為",
    "戴套",
    "覺得",
    "看起來",
    "然後",
    "接下來",
    "這個",
    "藥",
    "想起來",
    "復發",
    "回復",
    "復元",
    "現在",
    "禮拜",
    "下降",
    "正常",
    "好不好",
    "高一點",
    "因為",
    "藥物",
]

break_words = ["的藥", "半年", "你現", "下禮", "阿因", "阿有", "藥還", "那個藥"]


for word in add_words:
    jieba.suggest_freq(word, True)

for word in break_words:
    jieba.del_word(word)


special_words_mapping = {
    "HIV": "病毒",
    "HPV": "病毒",
    "菜花": "梅毒",
    "防H": "防毒",
    "U=U": "測不到毒量",
}


def replace_words(text):
    for k, word in special_words_mapping.items():
        text = text.replace(k, word)
    return text


def process_data(dataset):
    corpus = []
    answer = []
    article_id = []
    for datum in dataset:
        article = replace_words(datum["article"])
        sent_words = jieba.cut(article.lower())
        sent = " ".join(sent_words)
        corpus.append(sent)
        answer.append(datum["label"].tolist())
        article_id.append(datum["article_id"])
    return corpus, answer, article_id


def train_and_evaluate(corpus, answer, min_df=0.1, max_df=0.7, seed=1009, val_size=0.2):
    if val_size > 0:
        train_x, test_x, train_y, test_y = train_test_split(
            corpus.copy(), answer.copy(), test_size=val_size, random_state=seed
        )
    else:
        train_x = corpus.copy()
        train_y = answer.copy()

    tfidf_vec, clf = train(train_x, train_y, min_df, max_df)

    train_score = evaluate(tfidf_vec, clf, train_x, train_y)

    if val_size > 0:
        test_score = evaluate(tfidf_vec, clf, test_x, test_y)
        return train_score, test_score
    else:
        return train_score


def train(corpus, answer, min_df=0.1, max_df=0.7, show=False):
    tfidf_vec = TfidfVectorizer(
        token_pattern=r"(?u)\b\w+\b",
        # ngram_range=(1, 2),
        stop_words=stop_words,
        min_df=min_df,
        max_df=max_df,
    )

    tfidf_vec.fit(corpus)
    train_x = tfidf_vec.transform(corpus)
    train_y = answer.copy()

    clf = LogisticRegression().fit(train_x, train_y)

    weight = list(tfidf_vec.vocabulary_.items())
    for i in range(len(weight)):
        weight[i] = (weight[i][0], round(clf.coef_[0][weight[i][1]], 2))

    if show:
        print(sorted(weight, key=lambda i: i[1], reverse=True))

    return tfidf_vec, clf


def evaluate(tfidf_vec, classifier, corpus, answer):
    val_x = tfidf_vec.transform(corpus)
    val_y = answer.copy()
    return roc_auc_score(val_y, classifier.predict_proba(val_x)[:, 1])


def predict(tfidf_vec, clf, test_data, article_id):
    test_x = tfidf_vec.transform(test_data)
    probs = clf.predict_proba(test_x)[:, 1]

    Path("output").mkdir(parents=True, exist_ok=True)
    with open("output/decision.csv", "w") as f:
        csvwriter = csv.writer(f, delimiter=",")
        csvwriter.writerow(["article_id", "probability"])
        for _id, prob in zip(article_id, probs):
            csvwriter.writerow([_id, prob])
    with open("output/decision_configs.yml", "w") as yaml_file:
        yaml.dump(configs, yaml_file, default_flow_style=False)
    print("*Successfully saved prediction to output/decision.csv")


MIN_DF = 0.1  # 0.155
MAX_DF = 0.8  # 0.72
val_size = 0.2

with open("configs.yaml", "r") as stream:
    configs = yaml.safe_load(stream)

dataset = risk_dataset(configs, configs["risk_data"])
corpus, answer, _ = process_data(dataset)


seeds = list(range(1, 50))
train_scores, test_scores = [], []
for seed in seeds:
    train_score, test_score = train_and_evaluate(
        corpus, answer, MIN_DF, MAX_DF, seed=seed, val_size=val_size
    )
    train_scores.append(train_score)
    test_scores.append(test_score)
    print(
        f"[seed={seed}] train score: {train_score:.3f} | test score: {test_score:.3f}"
    )

print(f"average train: {np.mean(train_scores):.5f}")
print(f"average test: {np.mean(test_scores):.5f}")


tfidf_vec, clf = train(corpus, answer, MIN_DF, MAX_DF, show=True)

test_dataset = risk_dataset(configs, configs["dev_risk_data"])
test_corpus, _, article_id = process_data(test_dataset)
predict(tfidf_vec, clf, test_corpus, article_id)
