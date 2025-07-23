import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
#from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import joblib
import tensorflow as tf

# Download stopwords (only once)
nltk.download('stopwords')
stemmer = nltk.SnowballStemmer("english")
stopword = set(stopwords.words('english'))

# Clean function with raw strings to fix warnings
def clean(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text = " ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    return " ".join(text)

# Load pre-trained models and vectorizers (make sure these files exist!)
tfidf = joblib.load("model/tfidf.pkl")
multinomial_nb = joblib.load("model/mnb.pkl")
bernoulli_nb = joblib.load("model/bnb.pkl")
lgb_model = joblib.load("model/lgb.pkl")
tokenizer = joblib.load("model/tokenizer.pkl")
rnn_model = tf.keras.models.load_model("model/rnn_model.h5")

# Predict function for user input
def predict_stress(user_input):
    cleaned = clean(user_input)
    vector = tfidf.transform([cleaned])
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=500)

    pred1 = multinomial_nb.predict_proba(vector)[:, 1][0]
    pred2 = bernoulli_nb.predict_proba(vector)[:, 1][0]
    pred3 = lgb_model.predict_proba(vector)[:, 1][0]
    pred4 = rnn_model.predict(padded).flatten()[0]

    average = np.mean([pred1, pred2, pred3, pred4])
    label = "Stress" if average > 0.5 else "No Stress"
    level = get_stress_level(average)
    return label, level

# Function to determine stress level
def get_stress_level(prob):
    if prob < 0.5:
        return "No Stress"
    elif 0.5 <= prob < 0.6:
        return "Low Stress"
    elif 0.6 <= prob < 0.8:
        return "Medium Stress"
    else:
        return "High Stress"
