import unidecode
import re
from re import search

def check_bad_words(list_bad_words, sentence):
    remove = []
    for key in list_bad_words:
        if re.search(r"\b{}\b".format(key), sentence.lower().strip()):
            remove.append(key)
    return remove 
def check_url(scam, domain_shorteners, sentence):
    remove = []
    for domain in scam:
        if search(domain, sentence):
            remove.append(domain)
    for domain in domain_shorteners:
        if search(domain, sentence):
            remove.append(domain)
    return remove 
def remove_accent(sentence):
    return unidecode.unidecode(sentence)

def remove_dirty_word(scam, domain_shorteners, list_bad_words, sentence):
    if check_url(scam, domain_shorteners, sentence) != []:
        replace_text = check_url(scam, domain_shorteners, sentence)
        for word in replace_text:
            sentence = sentence.replace(word,'*'*len(word)) 
    if check_bad_words(list_bad_words, sentence) != []:
        replace_text = check_bad_words(list_bad_words, sentence)
        for word in replace_text:
            sentence = sentence.replace(word,'*'*len(word)) 
    if replace_text != []:
        y_pred = 1
    else:
        y_pred = 0
    return sentence, y_pred, 0.99