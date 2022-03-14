# Arabic-Dialects-Identification
- This repo is dedicated to build a classification pipeline to the problem of Arabic dialects identification, labels contain 18 clasess/dialects, using a classic ML and modern DL approaches. 

## Description
- The Dataset and the dialect identification problem were addressed by Qatar Computing Research Institute. More on: https://arxiv.org/pdf/2005.06557.pdf
- I Implemented a data-scraper, data-preprocessor (using Regex), data-modeling (SVM with TF-IDF and MarBert using HuggingFace) and deployment-script using FlaskAPIs locally.

---

## Files/Source
#### Fetching (Notebooks/Fetch-Dialects-Data.ipynb or Scripts/data_fetching.py)
- Data fetching is done by sending 1000 (max-json-length) ids per post-request using a start & end index, parsing its content and appending it in the data-frame. There's a small   trick in the request loop, if the end index exceeded the length of the ids, it will take the remainder.
- The output of this process is a data-frame that has the retrieved text and labels.
  
#### Preprocessing (Notebooks/Preprocessing-Dialect-Data.ipynb or Scripts/preprocess_and_tokenize.py)
- I have implemented two classes, one for preprocessing and the other is for tokenization. 
- Preprocessing has a pipeline that applies letters normalization, removing tashkeel, substitute characters, removing symbols, punctuation, spaces, quotation marks and English     letters. All of this is done by regex python module.
- Tokenization has a pipeline that applies a normal tokenizing for text, removing stop words and removing repeated words. There’s also a small EDA in the notebook.
  
#### Classic ML Modeling (Notebooks/Modeling-Dialect-Data-ClassicML.ipynb)
- Before training any models, I prepared data by dropping the non-value rows in both class (dialect) and text columns. 
- There were an imbalance in the dialects, so I used SMOTE (Synthetic Minority Oversampling Technique) to balance the classes. The amount of data is big but if I used             Undersampling there will be loss, more text data will help ML-Models to learn better.
- I used TF-IDF vectorizer as it's one of the best statistical vector spaces used in NLP Classic solutions. 
- The Model trained is LinearSVM as it's one of the best linear models in text classification problems.
- SVM understands relationships between features better than naïve-bayes. Also working with well prepared vector spaces makes SVM work well as in text classification problems.
- Trained using Kaggle (CPU).
  
#### DL Modeling (Notebooks/MARBERT-FineTuning.ipynb)
- I Used MARBERT (A Deep Bidirectional Transformer for Arabic).
- MARBERT is a large-scale pre-trained masked language model focused on both Dialectal Arabic (DA) and MSA.
- Fine-tunning a pre-trained dialectical Arabic transformer will excel in this task better than any BI-LSTM/LSTM model, as it has a tremendous word-embeddings trained on 1B       Arabic tweets from a large in-house dataset of about 6B tweets.
- MARBERT is the same network architecture as ARBERT (BERT-base), but without the next sentence prediction (NSP) objective since tweets are short.
- MARBERT tokenizer and fine-tuned transformer will be used later in deployment. 
- Trained using Kaggle GPU (NVIDIA V100).

#### Model Deployment (Scripts/app.py)
- This is a small script, based on Flask micro-framework, that takes an input from user and displayes an output using a static page.
  Every back-end processing is done in the previous stages.
  
---

## Evaluation Metrics and Results:
- Imbalance-learn module uses **Macro Average** internally.
- Macro averaging is perhaps the most straightforward amongst the numerous averaging methods.
- The macro-averaged F1 score (or macro F1 score) is computed by taking the arithmetic mean (aka unweighted mean) of all the per-class F1 scores.
- This method treats all classes equally regardless of their support values.
  working with an imbalanced dataset where all classes are equally important, using the macro average would be a good choice as it treats all classes equally.
  
  ![image](https://user-images.githubusercontent.com/36515196/158068910-f54e4753-9b46-4a0c-99ec-7974128deb73.png)
  ![image](https://user-images.githubusercontent.com/36515196/158068918-d08e985a-b4c8-4cf9-94fe-c9618770d780.png)

--- 

## Install
You have to download **Python 3** and the following Python libraries:
All are provided in **env.yml**

- [Tensorflow](https://www.tensorflow.org/)
- [Transformers](https://huggingface.co/docs/transformers/index)
- [MARBERT](https://huggingface.co/UBC-NLP/MARBERT)
- [NLTK](https://www.nltk.org/)
- [imblearn](https://imbalanced-learn.org/stable/)
- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- [scikit-learn](http://scikit-learn.org/stable/)
- [JobLib](https://joblib.readthedocs.io/en/latest/)
- [Flask](https://flask.palletsprojects.com/)

## API
![Screenshot_13](https://user-images.githubusercontent.com/36515196/158060575-32fda8d5-6272-4cd6-87ff-461b99335ca3.png)

