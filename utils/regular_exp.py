import re

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

unimportant = ["···", "⋯", "…", "..."]


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


def remove_unimportant(text):
    for t in unimportant:
        text = text.replace(t, "")
    return text


def remove_repeated(text):
    repeateds = get_repeated(text)
    for repeated in repeateds:
        text = text.replace(repeated, repeated[0])
    return text
