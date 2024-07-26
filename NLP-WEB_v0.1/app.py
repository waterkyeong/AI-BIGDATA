import joblib
import os
import re

from flask import Flask, render_template, request
from konlpy.tag import Okt

app = Flask(__name__)
app.debug = True

okt = Okt()

tfidf_vector = None
model_lr = None


def tw_tokenzier(text):
    tokenzier_ko = okt.morphs(text)
    return tokenzier_ko


def load_lr():
    global tfidf_vector, model_lr
    tfidf_vector = joblib.load(os.path.join(app.root_path, "model/tfidf_vect.pkl"))
    model_lr = joblib.load(os.path.join(app.root_path, "model/lr.pkl"))


def lt_transform(review):
    review = re.sub(r"\d+", " ", review)
    tfidf_matrix = tfidf_vector.transform([review])    
    return tfidf_matrix


@app.route("/predict", methods=["GET","POST"])
def nlp_predict():
    if request.method == "GET":
        return render_template("predict.html")
    else:
        _review = request.form["review"]
        _review_matrix = lt_transform(_review)
        _review_result = model_lr.predict(_review_matrix)[0]
        _predict_result = "긍정" if _review_result else "부정"
        result = {
            "review":_review,
            "review_result" : _predict_result
        }
        return render_template("predict_result.html",result=result)


@app.route("/")
def index():
 
    return render_template("index.html")

if __name__ == "__main__":
    load_lr()
    app.run()