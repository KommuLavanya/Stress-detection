import pandas as pd
import re
import string
import joblib
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.feature_extraction.text import TfidfVectorizer
import lightgbm as lgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Download stopwords
nltk.download('stopwords')
stopword = set(stopwords.words('english'))
stemmer = nltk.SnowballStemmer("english")

def clean(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = [word for word in text.split() if word not in stopword]
    text = " ".join([stemmer.stem(word) for word in text])
    return text

# Load and clean data
df = pd.read_csv('stress.csv')
df['text'] = df['text'].apply(clean)
X = df['text']
y = df['label']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
joblib.dump(tfidf, 'model/tfidf.pkl')

# Multinomial Naive Bayes
mnb = MultinomialNB()
mnb.fit(X_train_tfidf, y_train)
joblib.dump(mnb, 'model/mnb.pkl')

# Bernoulli Naive Bayes
bnb = BernoulliNB()
bnb.fit(X_train_tfidf, y_train)
joblib.dump(bnb, 'model/bnb.pkl')

# LightGBM
lgb_model = lgb.LGBMClassifier()
lgb_model.fit(X_train_tfidf, y_train)
joblib.dump(lgb_model, 'model/lgb.pkl')

# Tokenizer and RNN
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

max_len = 500
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)

joblib.dump(tokenizer, 'model/tokenizer.pkl')

rnn_model = Sequential([
    Embedding(10000, 64, input_length=max_len),
    LSTM(64),
    Dense(1, activation='sigmoid')
])
rnn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
rnn_model.fit(X_train_pad, y_train, epochs=5, batch_size=32, validation_data=(X_test_pad, y_test))
rnn_model.save('model/rnn_model.h5')

print("✅ All models retrained and saved to 'model/'")
