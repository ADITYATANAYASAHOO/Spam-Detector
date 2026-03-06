import pickle
import nltk
import string
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report

df = pd.read_csv("SMSSpamCollection", sep='\t', names=['label','message'])

print(df.head())
print (df.shape)
print(df['label'].value_counts())
df['label'] = df['label'].map({'ham':0, 'spam':1})

print(df.head())

#nltk.download('stopwords')

def clean_text(text):
    
    # convert to lowercase
    text = text.lower()
    
    # remove punctuation
    text = "".join([char for char in text if char not in string.punctuation])
    
    # remove stopwords
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    
    return " ".join(words)

df['clean_message'] = df['message'].apply(clean_text)

print(df[['message','clean_message']].head())

tfidf = TfidfVectorizer()

X = tfidf.fit_transform(df['clean_message'])

print(X.shape)
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = MultinomialNB()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

def predict_spam(message):
    cleaned = clean_text(message)
    vector = tfidf.transform([cleaned])
    prediction = model.predict(vector)

    if prediction[0] == 1:
        print("Spam message")
    else:
        print("Not spam")

predict_spam("Congratulations! You won a free iPhone. Click now!")
predict_spam("Hey bro are we meeting tomorrow?")
predict_spam("Free entry in a weekly competition to win cash prize")

pickle.dump(model, open("spam_model.pkl", "wb"))
pickle.dump(tfidf, open("vectorizer.pkl", "wb"))