#%%
import pandas as pd
import random as rd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer  
from sklearn.preprocessing import OneHotEncoder      
from statistics import mean
from sklearn.model_selection import KFold   
import joblib 
import os
import playsound
import speech_recognition as sr
import time
import sys
import ctypes
import wikipedia
import datetime
import json
import re
import webbrowser
import smtplib
import requests
import urllib
import urllib.request as urllib2
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
from time import strftime
from gtts import gTTS
from youtube_search import YoutubeSearch
#%%
wikipedia.set_lang('vi')
language = 'vi'
path = ChromeDriverManager().install()
# %%
def chao_hoi():
    hoc_chao_hoi(get_audio()) # Tiếp thu lời chào từ con người và
    # nhớ để lần sau chào lại người khác
    result = pd.read_csv('Loi_chao.csv')
    chao = rd.choice(result['LoiChao'].values)
    speak(chao)
#%%
def hoc_chao_hoi(greeting):
    result = pd.read_csv('Học cách chào hỏi\\Loi_chao.csv')
    C = {'LoiChao': [greeting]}
    df = pd.DataFrame(C, columns= ['LoiChao'])
    df_new = pd.concat([result, df], ignore_index=True)
    df_new.to_csv ('Loi_chao.csv', index = None, header=True) # here you have to write path, where result file will be stored

# %%
def get_audio():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Tôi: ", end='')
        audio = r.listen(source, phrase_time_limit=5)
        try:
            text = r.recognize_google(audio, language="vi-VN")
            print(text)
            return text
        except:
            print("...")
            return '0'

def speak(text):
    print("Bot: {}".format(text))
    tts = gTTS(text=text, lang=language, slow=False)
    tts.save("sound.mp3")
    playsound.playsound("sound.mp3", True)
    os.remove("sound.mp3")
#%%
chao_hoi()











# %%
