from flask import Flask, request, render_template
import pickle
import string
from nltk.corpus import stopwords

app = Flask(__name__)

model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

def clean_text(text):
    text = text.lower()
    text = "".join([c for c in text if c not in string.punctuation])
    words = text.split()
    words = [w for w in words if w not in stopwords.words('english')]
    return " ".join(words)

@app.route("/", methods=["GET","POST"])
def index():
    prediction = ""

    if request.method == "POST":
        message = request.form["message"]
        cleaned = clean_text(message)
        vector = vectorizer.transform([cleaned])
        result = model.predict(vector)

        if result[0] == 1:
            prediction = "Spam Message"
        else:
            prediction = "Not Spam"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)