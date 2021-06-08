import re
import json

with open("utils/mapping.json", "r") as f:
    mapping = json.load(f)

punction_mapping = mapping["punction"]
english_mapping = mapping["english"]
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
    matcher = re.compile(r"(.+)\1+")
    return set([match.group() for match in matcher.finditer(text)])


def replace_mapping(text):
    for k, m in mapping.items():
        # long terms first
        for s1, s2 in sorted(list(m.items()), key=lambda x: len(x[0]), reverse=True):
            if s1 == "HIV":
                print(s1, s2, text)
            text = text.replace(s1, s2)
    return text


def remove_unimportant(text):
    for t in unimportant:
        text = text.replace(t, "")
    return text


def remove_repeated(text):
    repeateds = get_repeated(text)
    for repeated in repeateds:
        repeat_unit = re.findall(r"(.+)\1+", repeated)
        text = text.replace(repeated, repeat_unit[0])
    return text


# Test
if __name__ == "__main__":
    print(remove_unimportant(".。測試測試~。測試測試......?......"))
    print(remove_repeated("嗨嗨你好你好嗨嗨我是誰是的是得"))
    print(replace_mapping("OKOK，IG好用，HIV要小心，NO"))