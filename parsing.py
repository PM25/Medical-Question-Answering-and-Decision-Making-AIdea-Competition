from torch.utils.data import DataLoader
import yaml
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
    "......": "⋯",
    "...?": "?",
    "⋯⋯。": "。",
    "⋯⋯?": "?",
}
redundant_words = ['哇', '咦', '蛤', '喔', '囉', '恩', '嘛',
                   '诶', '阿', '恩', '啊', '嗯', '耶', '拉', '啦', '唉',
                   '哼', '哦', '吼', '嘿', '欸', '呵', '亨', '咧', '哪',
                   '呦', '餒', '齁', '吧', '誒', '呀', '痾', '痾', '呃']

special_words_mapping = {'HIV': '愛滋病', 'HPV': '乳突病毒', '菜花': '乳突病毒',
                         'PREP': '預防性投藥', 'NO': '不', 'YES': '是',
                         '防H，': '防愛滋病', '防H。': '防愛滋病', 'U=U': '測不到病毒量',
                         'SO': '普通'}


def punction_normalize(text):
    for punc in punction_mapping:
        if punc in text:
            text = text.replace(punc, punction_mapping[punc])

        elif '~' in text:
            if text[-1] == '~':
                text = text[:-1] + '。'
            text = text.replace('~', '，')

        elif '!' in text:
            if text[-1] == '!':
                text = text[:-1] + '。'
            text = text.replace('!', ',')

    text = text.replace(',', '，').replace('?', '？')
    if text[0] in ['，', '。']:
        text = text[1:]
    if not text[-1] in ['？', '。', '⋯']:
        text += '。'
    return text


def remove_redundant_words(text):
    if '幹嘛' in text:
        text = text.replace('幹嘛', '幹麻')

    for word in redundant_words:
        if word + '⋯' in text:
            text = text.replace(word + '⋯', '')

        text = text.replace(word, '')

    text = text.replace('，。', '。').replace(
        '。，', '。').replace('，，', '，').replace('。。', '。').replace('、。', '。').replace('？。', '。')

    if text in [',', '。'] or text == '':
        return ''

    if text[0] in ['，', '。']:
        text = text[1:]

    text = text.replace('幹麻', '幹嘛')
    return text


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


def remove_repeated(text):
    repeateds = get_repeated(text)
    for repeated in repeateds:
        repeat_unit = re.findall(r"(.+)\1+", repeated)
        text = text.replace(repeated, repeat_unit[0])

    repeateds = get_repeated(text.replace('、', ''))
    if len(repeateds) > 0:
        for repeated in repeateds:
            repeat_unit = re.findall(r"(.+)\1+", repeated)
            text = text.replace('、' + repeat_unit[0], '')

    repeateds = get_repeated(text.replace('⋯', ''))
    if len(repeateds) > 0:
        for repeated in repeateds:
            repeat_unit = re.findall(r"(.+)\1+", repeated)
            text = text.replace('⋯' + repeat_unit[0], '')

    repeateds = get_repeated(text)
    for repeated in repeateds:
        repeat_unit = re.findall(r"(.+)\1+", repeated)
        text = text.replace(repeated, repeat_unit[0])
    return text


def special_word_normalize(text):
    text = text.upper()
    for word in special_words_mapping:
        if word in text:
            text = text.replace(word, special_words_mapping[word])
    return text

def text_preprocessing(text):
    text = punction_normalize(text)
    text = remove_redundant_words(text)
    text = remove_repeated(text)
    text = special_word_normalize(text)
    return text