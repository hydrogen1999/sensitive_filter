import unidecode
import re
from re import search


def remove_accent(sentence):
    return unidecode.unidecode(sentence)

class CheckBadWords:
    def __init__(self, badwords, blockwords):
        self.pattern1 = '|'.join([r"\b" + badword + r"\b" for badword in badwords])
        self.pattern2 = '|'.join(blockwords)
    def __call__(self, str_):
        print(str_)
        pattern1 = re.compile(self.pattern1, re.IGNORECASE)
        out = pattern1.sub("***", str_)
        pattern2 = re.compile(self.pattern2, re.IGNORECASE)
        replace_text = pattern2.findall(str_)
        for word in replace_text:
            out = out.replace(word,'*'*len(word))
        return out