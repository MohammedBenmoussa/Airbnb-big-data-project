import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.stats.contingency_tables as ct
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.weightstats import ztest
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from gensim import corpora
from numpy import linalg as LA
import xgboost as xgb
import nltk
from sklearn.metrics import mean_squared_error
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.corpus import stopwords
import re
from utils import *



# REad dataFrame
df = pd.read_csv("33000-BORDEAUX_nettoye.csv")
# Drop 0s
df = df[df['PrixNuitee']!=0]

df = df[['Longitude','Latitude','type_propriete','Type_logement','NbChambres','Capacite_accueil','Description','Titre','reglement_interieur']]
# Categorical Data Labeling
print('Labeling ...........................')
df = categorize(df)
print('Labeling Done!!')
# Textual Data Tokenizing
print('Preprocessing Text..........................')
df['Desc_pre']=df['Description'].map(lambda s:preprocess(s)) 
df['Reg_pre']=df['reglement_interieur'].map(lambda s:preprocess(s)) 
df['Titre_pre']=df['Titre'].map(lambda s:preprocess(s)) 
print('Preprocessing Done')
print('Tokenizing Words..........................')
df = tokenize_Descr(df)
df = tokenize_Reg(df)
df = tokenize_Titre(df)
print('Tokenizing Done!!')
df = df[['Longitude','Latitude','Type_logement','type_propriete','NbChambres','Capacite_accueil','Desc_pre','Titre_pre','Reg_pre']]
print('Preparing DataSet.........................')
df.to_csv('Result/df_predict.csv')
