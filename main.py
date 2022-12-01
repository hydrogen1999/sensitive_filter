import keras
from keras_preprocessing.sequence import pad_sequences
import nltk
import re
import string
import pickle
from fastapi import FastAPI 
from ultils import CheckBadWords, remove_accent

app = FastAPI(
    title="Sensitive content filter API",
    description="Test lọc nội dung nhạy cảm",
    version="0.1",
)
# load lib sensitive
with open('./vietnam-sensitive-words/profanity_wordlist.txt', 'r', encoding="utf8") as f:
    list_bad_words = [word[:-1] for word in f]
with open('./vietnam-sensitive-words/block_words.txt', 'r', encoding="utf8") as f:
    block_words = [word[:-1] for word in f]
with open('./domain_shorteners/domain_shorteners.txt', 'r') as f:
    domain_shorteners = [domain[:-1] for domain in f]
with open('./scam-links/links.txt', 'r') as f:
    scam_links = [link[:-1] for link in f]
scam = scam_links + domain_shorteners
list_bad_words = list_bad_words + scam

# load the sentiment model
model = keras.models.load_model('./checkpoints/model_name.h5')
with open('tokenizer.pickle', 'rb') as handle:
        word_tokenizer = pickle.load(handle)

# cleaning the data
stemmer = nltk.SnowballStemmer("english")
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
    # Stemm all the words in the sentence
    text = ' '.join(stemmer.stem(word) for word in text.split(' '))
    
    return text

def embed(corpus): 
    return word_tokenizer.texts_to_sequences(corpus)

@app.get("/predict-text")
def predict_sentiment(text: str, AI: int):
    """
    A simple function that receive a text content and predict the sentiment of the content.
    :param text:
    :return: prediction, probabilities
    """
    badwords_censor = CheckBadWords(list_bad_words, block_words)
    text = badwords_censor(text)
    if re.search("\*", text):
        result = {"cleaned_text": text, "Sensitive": 1, "Score": 0.99}
        return result
    else:
        if AI == 1:
            # # clean the text
            cleaned_text = preprocess_data(text)
            input_model =  pad_sequences(
                    embed([cleaned_text]), 
                    219
                    )
            
            # perform prediction
            y_pred = model.predict(input_model).astype("int32")[0][0].tolist()
            y_prob = 0.8
            
            # output dictionary
            sentiments = {0: "Negative", 1: "Positive"}
            
            # show results
            result = {"cleaned_text": cleaned_text, "Sensitive": y_pred, "Score": y_prob}
        else:
            result = {"cleaned_text": text, "Sensitive": 0, "Score": 0.99}
        return result