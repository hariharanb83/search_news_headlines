import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
# import streamlit_wordcloud as wordcloud
from search_functions import *

FILE_PATH = os.path.join('C:\\Users\\Hariharan\\Desktop\\ML Projects','Search using tfidf','data','india-news-headlines.csv')
TFIDF_FILENAME = os.path.join('C:\\Users\\Hariharan\\Desktop\\ML Projects','Search using tfidf','model','tfidf.pk')
TFIDF_FEATURES_FILENAME = os.path.join('C:\\Users\\Hariharan\\Desktop\\ML Projects','Search using tfidf','model','tfidf_features.pk')

@st.cache
def load_data(FILE_PATH):
    df = read_file(FILE_PATH)
    data = [preprocess(raw_text) for raw_text in (df['NEWS_Headline'])]
    df['clean_text'] = data    
    
    return(df)

@st.cache
def refresh_models(df,TFIDF_FILENAME,TFIDF_FEATURES_FILENAME):
    # Create tfidf features
    tfidf,tfidf_features = refresh_tf_idf_model(df,TFIDF_FILENAME,TFIDF_FEATURES_FILENAME)
    return(tfidf,tfidf_features)

def word_cloud_generator(df, selected_month, selected_year):
    wc_df = df.loc[(df['year']==selected_year) & (df['month']==selected_month)]

    # Read the whole text.
    text = ' '.join(wc_df['NEWS_Headline'].tolist())

    # Generate a word cloud image
    wordcloud = WordCloud().generate(text)

    # take relative word frequencies into account, lower max_font_size
    wordcloud = WordCloud(background_color="white",max_words=100,max_font_size=40, relative_scaling=.5,width=1000, height=400).generate(text)

    return(wordcloud)


    

#====================================================

st.title('Search News Headlines')

raw_df = load_data(FILE_PATH)
df = raw_df.copy()

st.subheader("TRENDS")
col1, col2, col3 = st.columns(3)
with col1:
    selected_year = st.slider("Select Year", min_value=2001, max_value=2020, step=1,value=2020)
with col2:
    selected_month = st.slider("Select Month", min_value=1, max_value=12, step=1,value=6)
with col3:
    trend_button = st.button("What's Trending?")

if trend_button:
    st.write("Trends during the period ", str(selected_year) + "-" + str(selected_month))
    wordcloud=word_cloud_generator(df, selected_month, selected_year)   
    st.image(wordcloud.to_array())
st.write("")
st.write("")

#=============
st.subheader("KEYWORD SEARCH")
start_year, end_year = st.select_slider("Select Period", options=list(range(2001,2021)),value=(2001, 2020))
st.write("Search News Headlines between the years ", str(start_year) + "-" + str(end_year))

search_text = st.text_input("Search using keywords", value="")
submit_button = st.button("Submit") 

if submit_button:
    pairwise_distance = nlp_search(search_text, TFIDF_FILENAME, TFIDF_FEATURES_FILENAME)
    filtered_df = format_results(df,pairwise_distance)
    
    if filtered_df.shape[0]==0:
        st.warning("No results found. Please try different keywords.")
        st.stop()
    
    # Bar Chart 1
    if filtered_df['year'].nunique() > 1:
        chart1_df = filtered_df.groupby('year').size()  
    else:
        st.write("Results found only in the year - ",str(filtered_df['year'][0]))
        chart1_df = filtered_df.groupby('month').size()  
        
    st.bar_chart(data=chart1_df, width=0, height=0, use_container_width=True)

    # table 1
    table1_df = filtered_df.loc[(filtered_df['year']>=start_year) & (filtered_df['year']<=end_year)]
    table1_df.reset_index(inplace=True)
    if table1_df.shape[0]==0:
        st.warning("No results found during this period. Please expand the search period.")
        st.stop()
    with st.expander(str(table1_df.shape[0]) + " Results found. Click to view details."):
        st.dataframe(table1_df[['NEWS_Headline']], height=400)    
#========
        
        
        
#=====================================================================
    