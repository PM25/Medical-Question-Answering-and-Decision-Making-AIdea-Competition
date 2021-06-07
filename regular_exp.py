import re
from dataset import risk_dataset

punction_mapping = {
    "······": "⋯",
    "⋯⋯": "⋯",
    "……": "⋯",
    "～": ",",
    ",。": "。",
    ",?": "?",
    ".。": "。",
    "。。": "。",
    "‧": ",",
    "?......": "?",
    "......?": "?",
    "。......": "。",
    "::": ":",
    "......。": "。",
    "~。": "。",
    ",......": ",",
    "。,": "。",
    "⋯⋯'": "⋯",
    "...?": "?",
    "⋯⋯。": "。",
    "⋯⋯?": "?",
}


def get_non_chinese(text, digit=True):
    non_chinese = set(re.findall("[a-zA-Z0-9]+", text))

    if digit is False:
        return [item for item in non_chinese if not item.isdigit()]

    return list(non_chinese)


def get_punctions(text, digit=True):
    punctions = set(re.findall("[^a-zA-Z0-9\s\w\u4e00-\u9fa5 %]+", text))
    return list(punctions)


def get_repeated(text):
    matcher = re.compile(r"(.)\1+")
    return set([match.group() for match in matcher.finditer(text)])


def remove_repeated(text):
    repeateds = get_repeated(text)
    for repeated in repeateds:
        text = text.replace(repeated, repeated[0])
    return text


import yaml
from torch.utils.data import DataLoader

with open("configs.yaml", "r") as stream:
    config = yaml.safe_load(stream)

if __name__ == "__main__":
    dataset = risk_dataset(config, config["risk_data"])
    dataloader = DataLoader(dataset)
    duplicate = []
    # for batch in dataloader:
    #     for article in batch["article"]:
    #         matcher = re.compile(r"(.)\1+")
    #         duplicate += [match.group() for match in matcher.finditer(article)]

    # print(set(duplicate))
    print(remove_repeated("嗨嗨你好嗨嗨我是誰是的是得"))