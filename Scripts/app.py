from flask import Flask, request, json, render_template
import tensorflow
import numpy as np
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC 

from sklearn import preprocessing

import nltk 
# nltk.download('all')
from nltk.corpus import stopwords, wordnet 
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
import pickle

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import TFAutoModelForSequenceClassification
import os
import re
import sys

os.chdir("..") 
from preprocess_and_tokenize import *
from dialects import dialects_dict

tokenizer = AutoTokenizer.from_pretrained("UBC-NLP/MARBERT")
marebert_model = TFAutoModelForSequenceClassification.from_pretrained("UBC-NLP/MARBERT", num_labels=18) 

marebert_model.load_weights("MARBERT/tf_model.h5")
svm_pipeline = joblib.load("ClassificationModel/classification_pipeline.pkl")

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/prediction", methods=["POST"])
def prediction():
    
    # Request parsing and preprocessing
    text_inp = list(request.form.values())[0]
    preprocessed_text = PreprocessTweets(text_inp).preprocessing_pipeline()
    
    # Tokenization is done twice, one for svm and other for marbert
    tokenized_for_ml = TweetsTokenizing(preprocessed_text).tokenize_pipeline()
    tokenized_for_bert = tokenizer.encode(preprocessed_text, truncation=True, padding=True, return_tensors="tf")
    
    # Do prediction using two models
    ml_predict = f"SVM Prediction: {svm_pipeline.predict([f'{tokenized_for_ml}'])[0]}"
    bert_predict = f"MARBERT Prediction: {np.argmax(marebert_model.predict(tokenized_for_bert)[0], axis=1)[0]}"
    
    # Rendering with predictions
    return render_template("index.html", prediction_ml=ml_predict, prediction_dl=prediction_dl)

app.run(debug=True)
