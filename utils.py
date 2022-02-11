import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.stats.contingency_tables as ct
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.weightstats import ztest
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from gensim import corpora
from numpy import linalg as LA
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.corpus import stopwords
import joblib
import re

def remove_outlayers(col,df):
    q_low = df[col].quantile(0.01)
    q_hi  = df[col].quantile(0.99)
    df_filtered = df[(df[col] < q_hi) & (df[col] > q_low)]
    return df_filtered

def fill_Nan(col,df,by):
    df[col] = df[col].fillna(by)
    return df


nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer() 

def preprocess(sentence):
    sentence=str(sentence)
    # Lowercase text
    sentence = sentence.lower()
    # Remove whitespace
    sentence=sentence.replace('{html}',"") 
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', sentence)
    # Remove weblinks
    rem_url=re.sub(r'http\S+', '',cleantext)
    # Remove numbers
    rem_num = re.sub('[0-9]+', '', rem_url)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(rem_num)  
    # Remove StopWords
    filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('french')]
    filtered_words = [w for w in filtered_words if len(w) > 2 if not w in stopwords.words('english')]
    # Use lemmatization
    lemma_words=[lemmatizer.lemmatize(w) for w in filtered_words]
    #return " ".join(filtered_words)
    #stemmer = PorterStemmer()
    #stemmed_words = [stemmer.stem(word) for word in lemma_words]
    return " ".join(lemma_words)

dictPropTYpe = {'Appartement':0,
               'Appartement en résidence':1,
               'Autre':2,
               'Bed & Breakfast':3,
               'Bungalow':4,
               'Cabane':5,
               'Dortoir':6,
               'Inconnue':7,
               'Loft':8,
               'Maison':9,
               'Maison de ville':10,
               'Maison écologique':11,
                'Villa':12}
dictLogemmType = {'Chambre partagée':0,'Chambre privée':1, 'Logement entier':2}


def categorize(df):
    for k in dictPropTYpe.keys():
        df.loc[(df['type_propriete']==k),['type_propriete']]=dictPropTYpe[k]
    for k in dictLogemmType.keys():
        df.loc[(df['Type_logement']==k),['Type_logement']]=dictLogemmType[k]
    return df

def tokenize_Descr(df_T):
    #Find words spreading (each word frequency)
    freq_d = pd.Series(" ".join(df_T['Desc_pre']).split()).value_counts()
    wordCount = pd.DataFrame(data = {
        'word':freq_d.index,
        'occurence':freq_d.values
    })
    freq_d[0]
    #15 -> 3000
    #Remove the least frequent words
    rare_d = list(wordCount['word'].where(wordCount['occurence']<15).dropna().values)
    df_T['Desc_pre'] = df_T['Desc_pre'].apply(lambda x: " ".join(x for x in x.split() if x not in rare_d))
    #Remove the most frequent words
    freq_w = list(wordCount['word'].where(wordCount['occurence']>3000).dropna().values)
    df_T['Desc_pre'] = df_T['Desc_pre'].apply(lambda x: " ".join(x for x in x.split() if x not in freq_w))
    df_T['Desc_pre'] = [text.split() for text in df_T['Desc_pre']]
    dict_d = corpora.Dictionary(df_T['Desc_pre'])
    corpus_d = [dict_d.doc2bow(line) for line in df_T['Desc_pre']]
    #Transform vectors of texts to scalar values (calculating norms of vectors)
    corpus_d_vec_norm = [LA.norm(vec) for vec in corpus_d]
    #Replace text descriptions in the database with norms of vectors
    df_T['Desc_pre'] = corpus_d_vec_norm
    return df_T

def tokenize_Reg(df_T):
    #Find words spreading (each word frequency)
    freq_d = pd.Series(" ".join(df_T['Reg_pre']).split()).value_counts()
    wordCount = pd.DataFrame(data = {
        'word':freq_d.index,
        'occurence':freq_d.values
    })
    freq_d[0]
    #15 -> 3000
    #Remove the least frequent words
    rare_d = list(wordCount['word'].where(wordCount['occurence']<15).dropna().values)
    df_T['Reg_pre'] = df_T['Reg_pre'].apply(lambda x: " ".join(x for x in x.split() if x not in rare_d))
    #Remove the most frequent words
    freq_w = list(wordCount['word'].where(wordCount['occurence']>2500).dropna().values)
    df_T['Reg_pre'] = df_T['Reg_pre'].apply(lambda x: " ".join(x for x in x.split() if x not in freq_w))
    df_T['Reg_pre'] = [text.split() for text in df_T['Reg_pre']]
    dict_d = corpora.Dictionary(df_T['Reg_pre'])
    corpus_d = [dict_d.doc2bow(line) for line in df_T['Reg_pre']]
    #Transform vectors of texts to scalar values (calculating norms of vectors)
    corpus_d_vec_norm = [LA.norm(vec) for vec in corpus_d]
    #Replace text descriptions in the database with norms of vectors
    df_T['Reg_pre'] = corpus_d_vec_norm
    return df_T

def tokenize_Titre(df_T):
    #Find words spreading (each word frequency)
    freq_d = pd.Series(" ".join(df_T['Titre_pre']).split()).value_counts()
    wordCount = pd.DataFrame(data = {
        'word':freq_d.index,
        'occurence':freq_d.values
    })
    freq_d[0]
    #15 -> 3000
    #Remove the least frequent words
    rare_d = list(wordCount['word'].where(wordCount['occurence']<15).dropna().values)
    df_T['Titre_pre'] = df_T['Titre_pre'].apply(lambda x: " ".join(x for x in x.split() if x not in rare_d))
    #Remove the most frequent words
    freq_w = list(wordCount['word'].where(wordCount['occurence']>2500).dropna().values)
    df_T['Titre_pre'] = df_T['Titre_pre'].apply(lambda x: " ".join(x for x in x.split() if x not in freq_w))
    df_T['Titre_pre'] = [text.split() for text in df_T['Titre_pre']]
    dict_d = corpora.Dictionary(df_T['Titre_pre'])
    corpus_d = [dict_d.doc2bow(line) for line in df_T['Titre_pre']]
    #Transform vectors of texts to scalar values (calculating norms of vectors)
    corpus_d_vec_norm = [LA.norm(vec) for vec in corpus_d]
    #Replace text descriptions in the database with norms of vectors
    df_T['Titre_pre'] = corpus_d_vec_norm
    return df_T



def predict_Adv(X):
    #[['Longitude','Latitude','type_propriete',
    #'NbChambres','Capacite_accueil','Desc_pre','Titre_pre','Reg_pre']]
    #------------------------------------- Classifier
    clf_model = joblib.load('Result/class_model.sav')#BEST CLASSIFIER
    classes = clf_model.predict(X.drop(['Latitude','Longitude'],axis=1))
    X['classes']=classes
    print('Classes Generated!!')
    #------------------------------------- Regressor I
    X_1 = X.loc[X['classes']== 0]
    reg_modelI = joblib.load('Result/regressOne_model.sav')#BEST REGRESOR|ONE
    prixNuitee_1 = reg_modelI.predict(X_1.drop(['Latitude','Longitude','classes'],axis=1))
    result_1 = pd.DataFrame(data = {
        'latitude': X_1['Latitude'].values,
        'longitude':X_1['Longitude'].values,
        'PrixNuitee':prixNuitee_1
    }
    )
    print('RegressorOne : Passed')
    #------------------------------------- Regressor II
    X_2 = X.loc[X['classes']== 1]
    reg_modelII = joblib.load('Result/regressTwo_model.sav')#BEST REGRESSOR|TWO
    prixNuitee_2 = reg_modelII.predict(X_2.drop(['Latitude','Longitude','classes'],axis=1))
    result_2 = pd.DataFrame( data = {
        'latitude':X_2['Latitude'].values,
        'longitude':X_2['Longitude'].values,
        'PrixNuitee':prixNuitee_2}
    )
    print('RegressorTwo : Passed')
    #------------------------------------- File Json
    result = pd.concat([result_1, result_2], ignore_index=True)
    result.to_json('predict.json',orient='records')
    print('Json File Saved!!')
    return result #prixNuitee_1,prixNuitee_2