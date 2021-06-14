import re
import csv
import yaml
import jieba
import random
import numpy as np
import jieba.analyse
from tqdm import tqdm
from pathlib import Path

from sklearn.metrics import roc_auc_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.experimental import enable_hist_gradient_boosting

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    BaggingClassifier,
)

from dataset import risk_dataset
from utils.setting import set_random_seed

with open("configs.yaml", "r") as stream:
    configs = yaml.safe_load(stream)
    configs["aug_mode"] = None

jieba.analyse.set_stop_words("data/stopwords.txt")
seed = configs["seed"]
set_random_seed(seed)

MIN_DF = 0.15  # 0.155
MAX_DF = 0.72  # 0.72


models = {
    "LogisticRegression": lambda: LogisticRegression(random_state=seed),
    "RandomForestClassifier": lambda: RandomForestClassifier(random_state=seed),
    # "HistGradientBoostingClassifier": lambda: HistGradientBoostingClassifier(
    #     random_state=seed
    # ),
    "RBF_SVM": lambda: SVC(probability=True, random_state=seed),
    # "LinearSVM": lambda: SVC(kernel="linear", probability=True, random_state=seed),
    "MLPClassifier": lambda: MLPClassifier(
        max_iter=500, warm_start=True, random_state=seed
    ),
    # "LogisticRegressionCV": lambda: LogisticRegressionCV(random_state=seed),
    # "KNeighborsClassifier": lambda: KNeighborsClassifier(n_neighbors=2),
}


with open("data/stopwords.txt", "r") as f:
    stop_words = [line.strip() for line in f.readlines()]

break_words = [
    "的藥",
    "你現",
    "下禮",
    "阿因",
    "阿有",
    "藥還",
    "那個藥",
    "藥有",
    "我剛",
    "樣我",
    "候會",
    "務型",
    "們有",
    "我調",
    "是膽",
    "膽腸",
    "胃科",
    "藥試",
    "些藥",
    "明胖",
]

with open("data/vocab.txt", "r") as f:
    add_words = [line.strip() for line in f.readlines()]

for word in add_words:
    jieba.suggest_freq(word, True)

for word in break_words:
    jieba.del_word(word)


special_words_mapping = {
    # "HIV": "病毒",
    # "HPV": "病毒",
    "菜花": "梅毒",
    "防H": "防毒",
    "U=U": "測不到毒量",
    "炮": "砲",
    # "a肝": "肝炎",
    # "b肝": "肝炎",
    # "c肝": "肝炎",
}


def get_catgorical_feat(corpus):
    with open("data/pos_key.txt", "r") as f:
        key_term = [line.strip() for line in f.readlines()]

    with open("data/neg_key.txt", "r") as f:
        key_term.extend([line.strip() for line in f.readlines()])

    cat_feat = np.zeros((len(corpus), len(key_term)))
    for i in range(len(corpus)):
        article = corpus[i].replace(" ", "")
        for j in range(len(key_term)):
            if key_term[j] in article:
                cat_feat[i, j] = 1.0

    return cat_feat


def replace_words(text):
    for k, word in special_words_mapping.items():
        text = text.replace(k, word)
    return text


def get_sent_num(corpus):
    n_sent = []
    for article in corpus:
        sents = re.split(r"[。!?？！。]", article)
        n_sent.append((len(sents) + 1) // 50)
    return np.expand_dims(n_sent, axis=1)


def process_data(dataset):
    corpus = []
    answer = []
    article_id = []

    for datum in dataset:
        article = " ".join([word for sent in datum["article"] for word in sent])
        article = article.lower()
        sent_words = jieba.cut(article)
        sent = " ".join(sent_words)
        answer.append(datum["label"].tolist())
        article_id.append(datum["article_id"])
        corpus.append(sent)

    return corpus, answer, article_id


def train_and_evaluate(corpus, answer, min_df=0.1, max_df=0.8, seed=1009, val_size=0.2):
    if val_size > 0:
        train_x, test_x, train_y, test_y = train_test_split(
            corpus.copy(), answer.copy(), test_size=val_size, random_state=seed
        )
    else:
        train_x = corpus.copy()
        train_y = answer.copy()

    clfs, tfidf_vec = train(train_x, train_y, min_df, max_df)

    probs = predict_prob(clfs, tfidf_vec, train_x)
    train_score = roc_auc_score(train_y, probs)

    if val_size > 0:
        probs = predict_prob(clfs, tfidf_vec, test_x)
        val_roc = roc_auc_score(test_y, probs)

        preds = predict_class(clfs, tfidf_vec, test_x)
        val_f1 = f1_score(test_y, preds)

        with open("output.txt", "a") as f:
            for x, a, b, c in zip(test_x, probs, preds, test_y):
                if b != c:
                    x = re.sub(" +", " ", x)
                    f.write(f"{x}\nprob:{a}, label: {c}\n\n")
            f.write("=" * 10 + "\n")
        # for a, b, c in zip(test_x, probs, test_y):
        #     print(a, b, c)
        print(classification_report(test_y, preds))

        return clfs, (train_score, val_roc, val_f1)
    else:
        return clfs, (train_score)


def train(corpus, answer, min_df=0.1, max_df=0.8):
    tfidf_vec = TfidfVectorizer(
        token_pattern=r"(?u)\b\w+\b",
        stop_words=stop_words,
        min_df=min_df,
        max_df=max_df,
    )

    tfidf_vec.fit(corpus)

    tfidf_x = tfidf_vec.transform(corpus)
    cat_feat = get_catgorical_feat(corpus)
    n_sent = get_sent_num(corpus)
    train_x = np.concatenate((tfidf_x.todense(), cat_feat, n_sent), 1)

    train_y = answer.copy()

    clfs = []
    for name, model in models.items():
        clf = model().fit(train_x, train_y)
        clfs.append(clf)

    # weight = list(tfidf_vec.vocabulary_.items())
    # for i in range(len(weight)):
    #     weight[i] = (weight[i][0], round(clfs[0].coef_[0][weight[i][1]], 2))
    # print(sorted(weight, key=lambda i: i[1], reverse=True))

    return clfs, tfidf_vec


def predict_prob(classifiers, tfidf_vec, corpus):
    tfidf_x = tfidf_vec.transform(corpus)
    cat_feat = get_catgorical_feat(corpus)
    n_sent = get_sent_num(corpus)
    data_x = np.concatenate((tfidf_x.todense(), cat_feat, n_sent), 1)

    probs = []
    for clf in classifiers:
        prob = clf.predict_proba(data_x)[:, 1]
        probs.append(prob)

    probs = np.array(probs)
    probs = np.mean(probs, axis=0)

    return probs


def predict_class(classifiers, tfidf_vec, corpus):
    preds = predict_prob(classifiers, tfidf_vec, corpus)

    preds[preds > 0.5] = 1
    preds[preds <= 0.5] = 0

    return preds


def save_predict(probs, article_id):
    Path("output").mkdir(parents=True, exist_ok=True)
    with open("output/decision.csv", "w") as f:
        csvwriter = csv.writer(f, delimiter=",")
        csvwriter.writerow(["article_id", "probability"])
        for _id, prob in zip(article_id, probs):
            csvwriter.writerow([_id, prob])
    with open("output/decision_configs.yml", "w") as yaml_file:
        yaml.dump(configs, yaml_file, default_flow_style=False)
    print("*Successfully saved prediction to output/decision.csv")


if __name__ == "__main__":
    dataset = risk_dataset(configs, configs["risk_data"])
    corpus, answer, _ = process_data(dataset)

    print("[Testing with 50 different seeds]")
    train_scores, val_rocs, val_f1s = [], [], []
    for _ in range(50):
        seed = random.randint(0, 9999)
        clfs, (train_score, val_roc, val_f1) = train_and_evaluate(
            corpus, answer, MIN_DF, MAX_DF, seed=seed, val_size=configs["val_size"]
        )
        train_scores.append(train_score)
        val_rocs.append(val_roc)
        val_f1s.append(val_f1)
        print(
            f"[seed={seed:<4}] train roc: {train_score:.3f} | test roc: {val_roc:.3f} | test f1: {val_f1:.3f}"
        )

    print("=" * 25)
    print(f"average train roc auc score: {np.mean(train_scores):.5f}")
    print(f"average val roc auc score: {np.mean(val_rocs):.5f}")
    print(f"average val f1 score: {np.mean(val_f1s):.5f}")

    clfs, tfidf_vec = train(corpus, answer, MIN_DF, MAX_DF)

    test_dataset = risk_dataset(configs, configs["dev_risk_data"])
    test_corpus, _, article_id = process_data(test_dataset)

    probs = predict_prob(clfs, tfidf_vec, test_corpus)
    save_predict(probs, article_id)
