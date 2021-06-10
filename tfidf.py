import csv
import yaml
import jieba
import random
import numpy as np
import jieba.analyse
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from dataset import risk_dataset

# from utils.regexp import remove_repeated, remove_unimportant

jieba.analyse.set_stop_words("data/stopwords.txt")
random.seed(1009)

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
    "睡前",
    "有一點",
    "個月",
    "比較少",
    "吃藥",
    "食慾",
    "類固醇",
    "肝炎",
]

break_words = ["的藥", "半年", "你現", "下禮", "阿因", "阿有", "藥還", "那個藥", "藥有", "我剛"]


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
    # "a肝": "肝炎",
    # "b肝": "肝炎",
    # "c肝": "肝炎",
}


def add_catgorical_feat(corpus):
    key_term = []
    for line in open("data/pos_key.txt", "r"):
        key_term.append(line.strip())
    for line in open("data/neg_key.txt", "r"):
        key_term.append(line.strip())

    catgorical_feat = np.zeros((len(corpus), len(key_term)))
    for i in range(len(corpus)):
        for j in range(len(key_term)):
            if key_term[j] in corpus[i]:
                catgorical_feat[i, j] = 1.0
    return catgorical_feat


def replace_words(text):
    for k, word in special_words_mapping.items():
        text = text.replace(k, word)
    return text


def process_data(dataset):
    corpus = []
    answer = []
    article_id = []
    for datum in dataset:
        article = " ".join([word for sent in datum["article"] for word in sent])
        article = article.lower()
        sent_words = jieba.cut(article)
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

    tfidf_vec, (clf1, clf2, clf3) = train(train_x, train_y, min_df, max_df)

    train_score = evaluate(tfidf_vec, (clf1, clf2, clf3), train_x, train_y)

    if val_size > 0:
        test_score = evaluate(tfidf_vec, (clf1, clf2, clf3), test_x, test_y)
        return train_score, test_score, (clf1, clf2, clf3), tfidf_vec
    else:
        return train_score


def train(corpus, answer, min_df=0.1, max_df=0.7, show=False):
    tfidf_vec = TfidfVectorizer(
        token_pattern=r"(?u)\b\w+\b",
        stop_words=stop_words,
        min_df=min_df,
        max_df=max_df,
    )
    tfidf_vec.fit(corpus)
    train_x = tfidf_vec.transform(corpus)
    cat_feat = add_catgorical_feat(corpus)
    train_x = np.concatenate((train_x.todense(), cat_feat), 1)

    train_y = answer.copy()

    clf1 = LogisticRegression().fit(train_x, train_y)
    clf2 = RandomForestClassifier().fit(train_x, train_y)
    clf3 = HistGradientBoostingClassifier().fit(train_x, train_y)

    # weight = list(tfidf_vec.vocabulary_.items())
    # for i in range(len(weight)):
    #     weight[i] = (weight[i][0], round(clf.coef_[0][weight[i][1]], 2))

    if show:
        # print(sorted(weight, key=lambda i: i[1], reverse=True))
        pass
        # for w in weight:
        #     print(w[0])
    return tfidf_vec, (clf1, clf2, clf3)


def evaluate(tfidf_vec, classifier, corpus, answer):
    val_x = tfidf_vec.transform(corpus)
    cat_feat = add_catgorical_feat(corpus)
    val_x = np.concatenate((val_x.todense(), cat_feat), 1)

    val_y = answer.copy()
    prob = (
        classifier[0].predict_proba(val_x)[:, 1]
        + classifier[1].predict_proba(val_x)[:, 1]
        + classifier[2].predict_proba(val_x)[:, 1]
    ) / 3

    return roc_auc_score(val_y, prob)


def predict(tfidf_vec, clf, test_data, article_id):
    test_x = tfidf_vec.transform(test_data)
    cat_feat = add_catgorical_feat(test_data)
    test_x = np.concatenate((test_x.todense(), cat_feat), 1)

    probs = (
        clf[0].predict_proba(test_x)[:, 1]
        + clf[1].predict_proba(test_x)[:, 1]
        + clf[2].predict_proba(test_x)[:, 1]
    ) / 3

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
configs["aug_mode"] = None
dataset = risk_dataset(configs, configs["risk_data"])
corpus, answer, _ = process_data(dataset)
corpus_all, answer_all, _ = process_data(dataset)

for _ in range(100):
    seed = random.randint(1, 1000)
    corpus, final_test_corpus, answer, final_test_answer = train_test_split(
        corpus_all.copy(), answer_all.copy(), test_size=0.1, random_state=seed
    )
    print("[Testing with 50 different seeds]")
    train_scores, test_scores = [], []

all_model = []
for _ in range(50):
    seed = random.randint(1, 1000)
    train_score, test_score, clf, tfidf = train_and_evaluate(
        corpus, answer, MIN_DF, MAX_DF, seed=seed, val_size=val_size
    )
    all_model.append([tfidf, clf])
    train_scores.append(train_score)
    test_scores.append(test_score)
    print(
        f"[seed={seed:<3}] train score: {train_score:.3f} | test score: {test_score:.3f}"
    )

print("=" * 25)
print(f"average train: {np.mean(train_scores):.5f}")
print(f"average test: {np.mean(test_scores):.5f}")

tfidf_vec, (clf1, clf2, clf3) = train(corpus, answer, MIN_DF, MAX_DF)

test_dataset = risk_dataset(configs, configs["dev_risk_data"])
test_corpus, _, article_id = process_data(test_dataset)
predict(tfidf_vec, (clf1, clf2, clf3), test_corpus, article_id)
