import torch
import torch.nn as nn
import re
import string
import pickle
from fastapi import FastAPI
from ultils import *
from model.model import SensitiveClassifier

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

app = FastAPI(
    title="Sensitive content filter API",
    description="Test lọc nội dung nhạy cảm",
    version="0.1",
)
# load lib sensitive
with open('./vietnam-sensitive-words/profanity_wordlist.txt',
          'r',
          encoding="utf8") as f:
    list_bad_words = [word[:-1] for word in f]
with open('./vietnam-sensitive-words/block_words.txt', 'r',
          encoding="utf8") as f:
    block_words = [word[:-1] for word in f]
with open('./domain_shorteners/domain_shorteners.txt', 'r') as f:
    domain_shorteners = [domain[:-1] for domain in f]
with open('./scam-links/links.txt', 'r') as f:
    scam_links = [link[:-1] for link in f]
scam = scam_links + domain_shorteners
list_bad_words = list_bad_words + scam

# Load model
model = SensitiveClassifier(n_classes=2)
model.to(device)
model.load_state_dict(
    torch.load('phobert_fold3.pth', map_location=torch.device(device)))

#Load tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
# load stopwords vn
with open("./vietnamese.txt", 'r', encoding="utf8") as f:
    stop_words = [word[:-1] for word in f]
len(stop_words)

# cleaning the data


def clean_text(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text


def preprocess_data(text):
    # Clean puntuation, urls, and so on
    text = clean_text(text)
    # Remove stopwords
    text = ' '.join(word for word in text.split(' ') if word not in stop_words)
    return text


@app.get("/predict-text")
def predict_sentiment(text: str):
    """
    A simple function that receive a text content and predict the sentiment of the content.
    :param text:
    :return: prediction, probabilities
    """
    badwords_censor = CheckBadWords(list_bad_words, block_words)
    if badwords_censor(text) == 1:
        result = {"text": text, "Sensitive": 1}
        return result
    else:
        # # clean the text
        text = clean_text(text)
        y_pred = infer(text, tokenizer, model)[0]
        print("pred: ", y_pred)
        # show results
        result = {"cleaned_text": text, "Sensitive": int(y_pred)}
        return result


#uvicorn main:app --host 0.0.0.0 --port 8000