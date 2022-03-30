import os
import re
import time
import pandas as pd
import numpy as np
import datetime
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity 
from sklearn.metrics import pairwise_distances

import seaborn as sns
import matplotlib.pyplot as plt

def read_file(FILE_PATH):
    df = pd.read_csv(FILE_PATH, delimiter=',', encoding='ISO-8859-1')
    df.drop(columns=['headline_category'],inplace=True)
    df.rename(columns = {'headline_text':'NEWS_Headline'}, inplace = True)
    df['publish_date'] = df['publish_date'].astype(str)
    df['publish_date'] = pd.to_datetime(df['publish_date'], format='%Y-%m-%d')
    df['year'] = pd.DatetimeIndex(df['publish_date']).year
    df['month'] = pd.DatetimeIndex(df['publish_date']).month
    df['day'] = pd.DatetimeIndex(df['publish_date']).day
    return(df)

def preprocess(text):
    """ Preprocess the input, i.e. lowercase, remove html tags, special character and digits."""

    # to lower case
    text = text.lower()

    # remove tags
    text = re.sub("</?.*?>"," <> ", text)

    # remove special characters and digits
    text = re.sub("(\\d|\\W)+"," ", text).strip()
    return text
    
def refresh_tf_idf_model(df,tfidf_filename,tfidf_features_filename):
    text_corpus = list(df['clean_text'])
    tfidf = TfidfVectorizer(lowercase = True, stop_words={'english'}, ngram_range=(1,2),max_features=50000)
    tfidf_features = tfidf.fit_transform(text_corpus)
    
    pickle.dump(tfidf, open(tfidf_filename,"wb"))
    pickle.dump(tfidf_features, open(tfidf_features_filename,"wb"))
    
    return(tfidf,tfidf_features)
    
def nlp_search(search_text, tfidf_filename, tfidf_features_filename):
    tfidf = pickle.load(open(tfidf_filename, "rb")) 
    tfidf_features = pickle.load(open(tfidf_features_filename, "rb"))
    
    formatted_search_text = [preprocess(search_text)]
    query = tfidf.transform(formatted_search_text)
    
    pairwise_dist = pairwise_distances(tfidf_features, query, metric='cosine').flatten()
    
    return(pairwise_dist)
    
def format_results(df,pairwise_distance):
    df['distance'] = pairwise_distance
    df['rank'] = df['distance'].rank(method='min')
    df['similarity'] = (1 - df['distance'])*100

    filtered_df = df[df['rank']<10000]
    filtered_df = filtered_df[filtered_df['similarity']>25]
    filtered_df = filtered_df.sort_values('rank')
    filtered_df['publish_date'] = filtered_df['publish_date'].astype(str)
    
    filtered_df = filtered_df.reset_index(drop=True)
    return(filtered_df)