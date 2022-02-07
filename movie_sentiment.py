from tkinter import *
import tensorflow as tf
import keras 
import emoji
import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.preprocessing import OneHotEncoder
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
import pickle

root = Tk()
root.title("Movie Sentiment Analysis")

frame = LabelFrame(root)
frame.pack()

myLabel1 = Label(frame,text='In three sentences or more give your opinion of any movie you have seen.')
e1 = Entry(frame,width=50,borderwidth=2)
myLabel1.grid(row=0,column=0)
e1.grid(row=1,column=0)


stopwords= nltk.corpus.stopwords.words('english')
punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'URL',text)

def remove_HTML(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'',text)


def remove_not_ASCII(text):
    text = ''.join([word for word in text if word in string.printable])
    return text

def remove_number(text):
    num = re.compile(r'[-+]?[.\d]*[\d]+[:,.\d]*')
    return num.sub(r'NUMBER', text)


def remove_stopwords(text):
    text_tokens = word_tokenize(text)
    tokens_without_sw = [word for word in text_tokens if not word in stopwords]
    text = ' '.join([str(elem) for elem in tokens_without_sw])
    return text

def lemma(text):
    tokens = word_tokenize(text)
    wnl = WordNetLemmatizer()
    text = ' '.join([wnl.lemmatize(words) for words in tokens])
    return text

def remove_punc(text):
    for ele in text:
        if ele in punc:
            text = re.sub(ele, "",text)
    return text
    
def cleanText(txt):
    txt = re.sub(r'[^\w\s]', '', txt)
    txt = re.sub(r'\n','',txt)
    # to remove emojis
    txt = re.sub(emoji.get_emoji_regexp(), r"", txt)
    txt = re.sub(r'https?:\/\/[A-Za-z0-9\.\/]+','',txt)
    txt = re.sub(r"https?://\S+|www\.\S+","",txt)
    txt = re.sub(r"<.*?>","",txt)
    
    txt = remove_URL(txt)
    txt = remove_HTML(txt)
    txt = remove_not_ASCII(txt)
    
    txt = txt.lower()
    
    txt = remove_number(txt)
    
    txt = remove_stopwords(txt)
    txt = lemma(txt)
    return txt  

def label(prob):
    if prob > 0.50:
        lab = "This is a Postive Movie Review."
    else:
        lab = "This is a Negative Movie Review."
    return lab

def sentiment():
    vocab = 50000
    oov = '<OOV>'
    embedding = 500
    padding = 'post'
    truncate = 'post'
    maxlength = 5000
    
    sentence = e1.get()
    clean_sen = cleanText(sentence)
    
    sen_lst = [clean_sen]
    
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    
    
    test_sen_token = tokenizer.texts_to_sequences(sen_lst)
    test_pad = pad_sequences(test_sen_token, maxlen=maxlength, padding=padding, truncating=truncate)
    
    model = keras.models.load_model(r'C:\Users\robtu')
    
    prob = model.predict(test_pad)
    
    myLabel = Label(frame,text=label(prob))
    myLabel.grid(row=2,column=0)
    
    
    

myButton = Button(frame,width=20,text="Find Sentiment",activebackground='#8af',command=sentiment)
myButton.grid(row=3,column=0)

root.mainloop()





