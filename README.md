# Arabic-Dialects-Identification
- This repo is dedicated to build a classification pipeline to the problem of Arabic dialects identification, labels contain 18 clasess/dialects, using a classic ML and modern DL approaches. 

## Description
- The Dataset and the dialect identification problem were addressed by Qatar Computing Research Institute. More on: https://arxiv.org/pdf/2005.06557.pdf
- I Implemented a data-scraper, data-preprocessor (using Regex), data-modeling (SVM with TF-IDF and MarBert using HuggingFace) and deployment-script using FlaskAPIs locally.

## Files/Source
#### Fetching (Notebooks/Fetch-Dialects-Data.ipynb or Scripts/data_fetching.py)
  Data fetching is done by sending 1000 (max-json-length) ids per post-request using a start & end index, parsing its content and appending it in the dataframe. 
  There's a small trick in the request loop, if the end index exceded the length of the ids, it will take the remainder.
  
#### Preprocessing (Notebooks/Fetch-Dialects-Data.ipynb or Scripts/preprocess_and_tokenize.py)
  This solution has two classes, one for preprocessing and the other for tokenization.
  Preprocessing has a pipeline that applies: letters normalizatiom, removing tashkeel, substitute characters, removing symbols and english letters.
  Tokenization has a pipeline that applies: tokenizing text, removing stop words and removing repeated words.
  There's a small EDA in the notebook.
  
#### Classic ML Modeling (Notebooks/Modeling-Dialect-Data-ClassicML.ipynb)
  At first, before training any model, i dropped the non-value rows in both class (dialect) and text columns.
  There were an imbalance in the dialects so i used SMOTE (Synthetic Minority Oversampling Technique) to balance the classes.
  I also used TF-IDS vectorizer as it's one of the best statistical vector spaces used in NLP Classic solution.
  The Model trained is LinearSVM as it's one of the best linear models in text classification problems.
  The pipeline will be used later in local deployment.
  Trained using kaggle (CPU)
  
#### DL Modeling (Notebooks/MARBERT-FineTuning.ipynb.ipynb)
  Using MARBERT (A Deep Bidirectional Transformer for Arabic) MARBERT is a large-scale pre-trained masked language model focused on both Dialectal Arabic (DA) and MSA.
  I prefered to fine-tune an arabic tranformer as it should excel in this task better than any BI-LSTM/LSTM model.
  MARBERT tokenizer and fine-tuned transformer will be used later in deployment.
  Trained using Kaggle GPU (NVIDIA V100).

#### Model Deployment (Scripts/api.py)
  This is a small script, based on Flask micro-framework, that takes an input from the used and displayes an output using a static page.
  Every back-end processing is done in the previous stages.


## Install
You have to download **Python 3** and the following Python libraries:

- [Tensorflow](https://www.tensorflow.org/)
- [Transformers](https://huggingface.co/docs/transformers/index)
- [MARBERT](https://huggingface.co/UBC-NLP/MARBERT)
- [NKT](https://www.nltk.org/)
- [imblearn](https://imbalanced-learn.org/stable/)
- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- [scikit-learn](http://scikit-learn.org/stable/)
- [JobLib](https://joblib.readthedocs.io/en/latest/)
- [Flask](https://flask.palletsprojects.com/)
