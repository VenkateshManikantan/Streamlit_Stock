import pandas as pd
from pandas_datareader import data as pdr
import numpy as np

import requests 
import yfinance as yf
import selenium
import praw
from praw.models import MoreComments

import datetime as dt

import reddit_config as r_cnf
from reddit_extraction import extract_reddit_post_com_rep_ramp




from sklearn.feature_extraction.text import CountVectorizer

import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})
import numpy 

import string
from nltk.tokenize import word_tokenize, sent_tokenize
from random import choices
from time import time
import gensim 
from gensim.models import KeyedVectors


import matplotlib.cm as cm
from sklearn.manifold import TSNE


import streamlit as st

#Getting Reddit Data:-

def get_stock_data(stock_name, time_range): #Name of the stock - input string , Time range - yyyy-mm-dd - input string
    yf.pdr_override()
    Stock = pdr.get_data_yahoo(stock_name, start=time_range)                                         
    return Stock 
def tsne_plot_similar_words(title, labels, embedding_clusters, word_clusters, a, filename=None):
    plt.figure(figsize=(15,15))
    
    
    plt.rcParams.update({'font.size': 10})
    colors = cm.rainbow(np.linspace(0, 1, len(labels)))
    for label, embeddings, words, color in zip(labels, embedding_clusters, word_clusters, colors):
        x = embeddings[:, 0]
        y = embeddings[:, 1]
        plt.scatter(x, y, c=color, alpha=a, label=label,)
        for i, word in enumerate(words):
            plt.annotate(word, alpha=0.5, xy=(x[i], y[i]), xytext=(5, 5),
                         textcoords='offset points', ha='right', va='bottom', size= 15)
    plt.legend(loc=2, prop={'size': 10})
    plt.title(title)
    plt.grid(True)
    if filename:
        plt.savefig(filename, format='png', bbox_inches='tight')
    st.pyplot()
def fast_tokenize(text):
    punct = string.punctuation + '“' + '”' + '‘' + "’"
    lower_case = text.lower()
    lower_case = lower_case.replace('—', ' ').replace('\n', ' ')
    no_punct = "".join([char for char in lower_case if char not in punct])
    tokens = no_punct.split()
    return tokens

st.set_option('deprecation.showPyplotGlobalUse', False)


st.title('NLP analysis MEME Vs Normal Stocks')


add_selectbox = st.sidebar.selectbox(
    'List of Popular Meme Stocks',
    ('GME', 'HOOD','AMC','TSLA','PLTR','CLOV')
)
add_selectbox = st.sidebar.selectbox(
    'List of Popular Non-meme Stocks',
    ('IRBT', 'RDFN', 'ROKU','PM','UPWK','PINS')
)

st.text('Please enter the name of a stock and start date to get stock price' )
st.text('from that date to present')
with st.form(key='my_form'):
    text_input = st.text_input(label='Enter the name of a stock')
    text_input2 = st.text_input(label='Start date format (yyyy-mm-dd)')
    submit_button = st.form_submit_button(label='Submit')
    
    
if len(text_input) != 0:
    GME_data = get_stock_data(text_input,text_input2)
    GME_data1 = pd.DataFrame(data= GME_data)
    GME_data1 = GME_data1.drop(['Volume'], axis=1)

    st.dataframe(GME_data1)

    fig, ax = plt.subplots(figsize= (70,70))
    sns.set(font_scale=7)
    ax.axes.set_title(text_input,fontsize=50)
    ax.set_xlabel("Time",fontsize=30)
    ax.set_ylabel("Stock Prices",fontsize=30)    
    sns.lineplot(data= GME_data1)
    st.pyplot()
    
    fig, ax = plt.subplots(figsize= (70,70))
    GME_dataV = pd.DataFrame(data= GME_data)
    GME_dataV = GME_dataV.drop(['Open','High','Low','Close','Adj Close'], axis=1)
    sns.set(font_scale=7)
    ax.axes.set_title(text_input,fontsize=50)
    ax.set_xlabel("Time",fontsize=30)
    ax.set_ylabel("Volume",fontsize=30)    
    sns.lineplot(data= GME_dataV)
    st.pyplot()

st.text('You can perform a basic NLP S-TNE analysis of the comment tree scraped')
st.text('from the subreddit of your intrest.')
with st.form(key='my_for'):
    text_input3 = st.text_input(label='Enter the name of a subreddit page:')
    text_input4 = st.text_input(label='How many posts you want to analyze in the subreddit:')
    submit_button2 = st.form_submit_button(label='Submit')


if len(text_input3) != 0:
    post_data,stock_data = extract_reddit_post_com_rep_ramp(text_input3,int(text_input4))
    cols = ['com_body','comm_tier1', 'comm_tier2']
    stock_data['combined'] = stock_data[cols].apply(lambda row: ','.join(row.values.astype(str)), axis=1)
    df_S_M = stock_data.groupby([stock_data['timestamp'].dt.date])['combined'].apply(lambda x: ','.join(x)).reset_index()
    df_S_M = pd.DataFrame(data = df_S_M)
    
    def encode_1(a):
        a = a.encode('utf-8').strip()
        return a

    df_S_M['combined'] = df_S_M['combined'].apply(encode_1)
    df_S_M['combined'] = df_S_M['combined'].astype(str)

    texts = df_S_M['combined'].values
 
    sentences = [s for t in texts for s in sent_tokenize(t)]


    words_by_sentence = [fast_tokenize(s) for s in sentences]
    
    w2v_model = gensim.models.Word2Vec(min_count=20,
                     window=2,
                     vector_size=300,
                     sample=6e-5, 
                     alpha=0.03, 
                     min_alpha=0.0007, 
                     negative=20,
                     workers=8)
    w2v_model.build_vocab(words_by_sentence, 
                      progress_per=10000)
    t = time()
    w2v_model.train(words_by_sentence, 
                total_examples=w2v_model.corpus_count, 
                epochs=30, 
                report_delay=1)
    
    keys = ['buy','sell','price','hold','stock','crash','cash','money','trade','market']

    embedding_clusters = []
    word_clusters = []
    for word in keys:
        embeddings = []
        words = []
        for similar_word, _ in w2v_model.wv.most_similar(word, topn=30):
            words.append(similar_word)
            embeddings.append(w2v_model.wv[similar_word])
        
        embedding_clusters.append(embeddings)
        word_clusters.append(words)
    
    

    embedding_clusters = np.array(embedding_clusters)
    n, m, k = embedding_clusters.shape
    tsne_model_en_2d = TSNE(perplexity=15, n_components=2, init='pca', n_iter=3500, random_state=32)
    embeddings_en_2d = np.array(tsne_model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)
    
    tsne_plot_similar_words('Similar words trained from GME stock conversations', keys, embeddings_en_2d, word_clusters, 0.7,
                        'GME_stock.png')

