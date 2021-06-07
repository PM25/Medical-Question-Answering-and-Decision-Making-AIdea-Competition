import re
import yaml
from torch.utils.data import DataLoader

with open("configs.yaml", "r") as stream:
    config = yaml.safe_load(stream)


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
