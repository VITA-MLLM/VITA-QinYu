import re
import regex as rex
# from config import config

abbv = []
with open('cosyvoice/utils/special_english_abbv','r') as f:
    abbv = [line.strip() for line in f.readlines()]

language2id ={
    'zh': 0,
    'en': 1,
    'other': 2,
    'paralanguage': 3
}
zh_pattern = rex.compile(r'[\u4e00-\u9fa5 ]+')
en_pattern = rex.compile(r"^[a-zA-Z' ]+$")
jp_pattern = rex.compile(r'[\u3040-\u30ff\u31f0-\u31ff]')
kr_pattern = rex.compile(r'[\uac00-\ud7af\u1100-\u11ff\u3130-\u318f\ua960-\ua97f]')
num_pattern=rex.compile(r'[0-9]')

paralanguage=[]
with open('tools/paralanguage.txt', 'r') as f:
    for line in f.readlines():
        paralanguage.append(line.strip())

def detect_langage(text):
    lang='other'

    if bool(zh_pattern.match(text)) or bool(num_pattern.match(text)):
        lang = 'zh'
        return lang

    if bool(en_pattern.match(text.strip())):
        lang = 'en'
        return lang

    if text in paralanguage:
        lang = 'paralanguage'
    return lang


module ='langid'
langid_languages = [
    "af",
    "am",
    "an",
    "ar",
    "as",
    "az",
    "be",
    "bg",
    "bn",
    "br",
    "bs",
    "ca",
    "cs",
    "cy",
    "da",
    "de",
    "dz",
    "el",
    "en",
    "eo",
    "es",
    "et",
    "eu",
    "fa",
    "fi",
    "fo",
    "fr",
    "ga",
    "gl",
    "gu",
    "he",
    "hi",
    "hr",
    "ht",
    "hu",
    "hy",
    "id",
    "is",
    "it",
    "ja",
    "jv",
    "ka",
    "kk",
    "km",
    "kn",
    "ko",
    "ku",
    "ky",
    "la",
    "lb",
    "lo",
    "lt",
    "lv",
    "mg",
    "mk",
    "ml",
    "mn",
    "mr",
    "ms",
    "mt",
    "nb",
    "ne",
    "nl",
    "nn",
    "no",
    "oc",
    "or",
    "pa",
    "pl",
    "ps",
    "pt",
    "qu",
    "ro",
    "ru",
    "rw",
    "se",
    "si",
    "sk",
    "sl",
    "sq",
    "sr",
    "sv",
    "sw",
    "ta",
    "te",
    "th",
    "tl",
    "tr",
    "ug",
    "uk",
    "ur",
    "vi",
    "vo",
    "wa",
    "xh",
    "zh",
    "zu",
]
def convert_to_lowercase(sentence):
    # 使用空格分割句子以获取单词列表
    words = sentence.split()

    # 使用列表推导式将每个单词的首字母转换为小写，如果它是大写的
    lowercase_words = [word[0].lower() + word[1:] if word[0].isupper() else word for word in words]
    #lowercase_words = []
    #for word in words:
    #    if word in abbv:
    #        word = word.upper() # 全部转成大写
    #        word = ' '.join(list(word))# 中间加空格
    #    else:
    #        word = word.lower()
    #    lowercase_words.append(word)

    # 使用 ' '.join 将处理过的单词列表重新组合成句
    lowercase_sentence = ' '.join(lowercase_words)
    return lowercase_sentence

def classify_language(text: str, target_languages: list = None) -> str:
    if module == "fastlid" or module == "fasttext":
        from fastlid import fastlid, supported_langs

        classifier = fastlid
        if target_languages != None:
            target_languages = [
                lang for lang in target_languages if lang in supported_langs
            ]
            fastlid.set_languages = target_languages
    elif module == "langid":
        import langid

        classifier = langid.classify
        if target_languages != None:
            target_languages = [
                lang for lang in target_languages if lang in langid_languages
            ]
            langid.set_languages(target_languages)
    else:
        raise ValueError(f"Wrong module {module}")

    lang = classifier(text)[0]

    return lang


def classify_zh_ja(text: str) -> str:
    for idx, char in enumerate(text):
        unicode_val = ord(char)

        # 检测日语字符
        if 0x3040 <= unicode_val <= 0x309F or 0x30A0 <= unicode_val <= 0x30FF:
            return "ja"

        # 检测汉字字符
        if 0x4E00 <= unicode_val <= 0x9FFF:
            # 检查周围的字符
            next_char = text[idx + 1] if idx + 1 < len(text) else None

            if next_char and (
                0x3040 <= ord(next_char) <= 0x309F or 0x30A0 <= ord(next_char) <= 0x30FF
            ):
                return "ja"

    return "zh"


def split_alpha_nonalpha(text):
    return re.split(
        r"(?:(?<=[\u4e00-\u9fff])|(?<=[\u3040-\u30FF]))(?=[a-zA-Z])|(?<=[a-zA-Z])(?:(?=[\u4e00-\u9fff])|(?=[\u3040-\u30FF]))",
        text,
    )


import regex as rex
zh_pattern = rex.compile(r'[\u4e00-\u9fa5]')
en_pattern = rex.compile(r'[a-zA-Z]')
jp_pattern = rex.compile(r'[\u3040-\u30ff\u31f0-\u31ff]')
kr_pattern = rex.compile(r'[\uac00-\ud7af\u1100-\u11ff\u3130-\u318f\ua960-\ua97f]')
num_pattern=rex.compile(r'[0-9]')
comma=r"(?<=[.。!！?？；;，,、:：'\"‘“”’()（）《》「」~——])"    #向前匹配但固定长度
tags={'ZH':'<|zh|>','EN':'<|en|>','JP':'[JA]','KR':'[KR]'}
#tags = {'ZH':'','EN':'','JP':'','KR':''}

def tag_cjke(text):
    '''为中英日韩加tag,中日正则分不开，故先分句分离中日再识别，以应对大部分情况'''
    sentences = rex.split(r"([.。!！?？；;，,、:：'\"‘“”’()（）【】《》「」~——]+ *(?![0-9]))", text) #分句，排除小数点
    sentences.append("")
    sentences = ["".join(i) for i in zip(sentences[0::2],sentences[1::2])]
    prev_lang=None
    tagged_text = ""
    tagged_text_list = []
    for s in sentences:
        #全为符号跳过
        nu = rex.sub(r'[\s\p{P}]+', '', s, flags=rex.U).strip()
        if len(nu)==0:
            continue
        s = rex.sub(r'[()（）《》「」【】‘“”’]+', '', s)
        # print(s)
        prev_lang,tagged_cke=tag_cke(s,prev_lang)
        tagged_cke_list = tagged_cke.split('<|en|>')
        if len(tagged_cke_list) > 1:
            # print(tagged_cke_list)
            tagged_cke_list[1] = convert_to_lowercase(tagged_cke_list[1])
            tagged_cke = ''.join(tagged_cke_list)

        tagged_text_list.append(tagged_cke)
        tagged_text +=tagged_cke
    return tagged_text, tagged_text_list


def tag_jke(text,prev_sentence=None):
    '''为英日韩加tag'''
    # 初始化标记变量
    tagged_text = ""
    prev_lang = None
    tagged=0
    # 遍历文本
    for char in text:
        # 判断当前字符属于哪种语言
        if jp_pattern.match(char):
            lang = "JP"
        elif zh_pattern.match(char):
            lang = "JP"
        elif kr_pattern.match(char):
            lang = "KR"
        elif en_pattern.match(char):
            lang = "EN"
        # elif num_pattern.match(char):
        #     lang = prev_sentence
        else:
            lang = None
            tagged_text += char
            continue
        # 如果当前语言与上一个语言不同，就添加标记
        if lang != prev_lang:
            tagged=1
            if prev_lang==None: # 开头
                tagged_text =tags[lang]+tagged_text
            else:
                tagged_text =tagged_text +tags[prev_lang]+tags[lang]

            # 重置标记变量
            prev_lang = lang

        # 添加当前字符到标记文本中
        tagged_text += char

    # 在最后一个语言的结尾添加对应的标记
    if prev_lang:
            tagged_text += tags[prev_lang]
    if not tagged:
        prev_lang=prev_sentence
        tagged_text =tags[prev_lang]+tagged_text+tags[prev_lang]

    return prev_lang,tagged_text

def tag_cke(text,prev_sentence=None):
    '''为中英韩加tag'''
    # 初始化标记变量
    tagged_text = ""
    prev_lang = None
    # 是否全略过未标签
    tagged=0

    # 遍历文本
    for char in text:
        # 判断当前字符属于哪种语言
        if zh_pattern.match(char):
            lang = "ZH"
        elif kr_pattern.match(char):
            lang = "KR"
        elif en_pattern.match(char):
            lang = "EN"
        # elif num_pattern.match(char):
        #     lang = prev_sentence
        else:
            # 略过
            lang = None
            tagged_text += char
            continue

        # 如果当前语言与上一个语言不同，添加标记
        if lang != prev_lang:
            tagged=1
            if prev_lang==None: # 开头
                tagged_text =tags[lang]+tagged_text
            else:
                tagged_text = convert_to_lowercase(tagged_text )
                tagged_text =tagged_text+tags[prev_lang]+tags[lang]

            # 重置标记变量
            prev_lang = lang

        # 添加当前字符到标记文本中
        tagged_text += char

    # 在最后一个语言的结尾添加对应的标记
    if prev_lang:
            tagged_text += tags[prev_lang]
    # 未标签则继承上一句标签
    if tagged==0 and prev_lang:
        prev_lang=prev_sentence
        tagged_text =tags[prev_lang]+tagged_text+tags[prev_lang]
    return prev_lang,tagged_text
if __name__ == "__main__":
    text = "《whataya want from me》，来自才华横溢的Adam Lambert，发行于2009年。Adam Lambert曾获得第17届全球华语榜中榜-最受欢迎国际歌手奖，而这首歌曲展现了ta一贯的动人风采，不容错过。"
    print(tag_cjke(text))
    #print(classify_language(text))
    #print(classify_zh_ja(text))  # "zh"

    #text = "これはテストテキストです"
    #print(classify_language(text))
    #print(classify_zh_ja(text))  # "ja"
