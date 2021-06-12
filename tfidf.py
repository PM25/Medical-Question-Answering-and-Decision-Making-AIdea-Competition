import csv
import yaml
import jieba
import random
import numpy as np
import jieba.analyse
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from torch.utils.tensorboard import SummaryWriter

from dataset import risk_dataset
from utils.setting import set_random_seed

with open("configs.yaml", "r") as stream:
    configs = yaml.safe_load(stream)
    configs["aug_mode"] = None

jieba.analyse.set_stop_words("data/stopwords.txt")
seed = configs["seed"]
set_random_seed(seed)

MIN_DF = 0.1  # 0.155
MAX_DF = 0.8  # 0.72


models = {
    # "LogisticRegression": lambda: LogisticRegression(random_state=seed),
    "RandomForestClassifier": lambda: RandomForestClassifier(random_state=seed),
    "HistGradientBoostingClassifier": lambda: HistGradientBoostingClassifier(
        random_state=seed
    ),
    "SVC": lambda: SVC(probability=True, random_state=seed),
    # "MLPClassifier": lambda: MLPClassifier(early_stopping=True, random_state=seed),
}


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
    "這樣",
    "任務型",
    "接下來",
]

break_words = [
    "的藥",
    "半年",
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
]


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


def get_catgorical_feat(corpus):
    with open("data/pos_key.txt", "r") as f:
        key_term = [line.strip() for line in f.readlines()]

    with open("data/neg_key.txt", "r") as f:
        key_term.extend([line.strip() for line in f.readlines()])

    cat_feat = np.zeros((len(corpus), len(key_term)))
    for i in range(len(corpus)):
        for j in range(len(key_term)):
            if key_term[j] in corpus[i]:
                cat_feat[i, j] = 1.0

    return cat_feat


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
        val_score = roc_auc_score(test_y, probs)
        return clfs, (train_score, val_score)
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
    train_x = np.concatenate((tfidf_x.todense(), cat_feat), 1)

    train_y = answer.copy()

    clfs = []
    for name, model in models.items():
        clf = model().fit(train_x, train_y)
        clfs.append(clf)

    return clfs, tfidf_vec


def predict_prob(classifiers, tfidf_vec, corpus):
    tfidf_x = tfidf_vec.transform(corpus)
    cat_feat = get_catgorical_feat(corpus)
    data_x = np.concatenate((tfidf_x.todense(), cat_feat), 1)

    probs = []
    for clf in classifiers:
        prob = clf.predict_proba(data_x)[:, 1]
        probs.append(prob)

    probs = np.array(probs)
    probs = np.mean(probs, axis=0)

    return probs


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
    train_scores, val_scores = [], []
    for _ in range(50):
        seed = random.randint(1, 1000)
        clfs, (train_score, val_score) = train_and_evaluate(
            corpus, answer, MIN_DF, MAX_DF, seed=seed, val_size=configs["val_size"]
        )
        train_scores.append(train_score)
        val_scores.append(val_score)
        print(
            f"[seed={seed:<3}] train score: {train_score:.3f} | test score: {val_score:.3f}"
        )

    print("=" * 25)
    print(f"average train score: {np.mean(train_scores):.5f}")
    print(f"average val score: {np.mean(val_scores):.5f}")

    clfs, tfidf_vec = train(corpus, answer, MIN_DF, MAX_DF)

    test_dataset = risk_dataset(configs, configs["dev_risk_data"])
    test_corpus, _, article_id = process_data(test_dataset)

    probs = predict_prob(clfs, tfidf_vec, test_corpus)
    save_predict(probs, article_id)
