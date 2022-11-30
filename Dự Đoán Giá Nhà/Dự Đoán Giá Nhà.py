
'''
The following code is mainly from Chap 2, Géron 2019 
See https://github.com/ageron/handson-ml2/blob/master/02_end_to_end_machine_learning_project.ipynb
'''
''' THAM KHẢO: CODE, DATA của thầy Trần Nhật Quang
'''
# In[]: DỰ ĐOÁN GIÁ NHÀ

#%% THƯ VIỆN
import pandas as pd
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

#%% NÓI TIẾNG VIỆT
wikipedia.set_lang('vi')
language = 'vi'
path = ChromeDriverManager().install()

# In[2]: PART 1. NHẬN DỮ LIỆU GIÁ NHÀ
raw_data = pd.read_csv('datasets\GiaChungCu_HCM_June2021_laydulieu_com.csv')

#%% MÔ TẢ SƠ QUA VỀ DỰ LIỆU

print(raw_data.info())  # Thông tin            

print(raw_data.head(3)) # 3 dòng đầu data

print(raw_data['GIẤY TỜ PHÁP LÝ'].value_counts()) # Số lượng của 
#mỗi dữ liệu khác nhau trong feature

print(raw_data.describe()) # mô tả tổng quát

#%% Xóa những cột không có ích cho việc train
raw_data.drop(columns = ["GIỐNG - LOẠI", "GIỐNG - NHU CẦU", "GIỐNG - TỈNH THÀNH", "SỐ TẦNG"], inplace=True) 

#%% Chia tập data thành 2 tập: train và test
raw_data["KHOẢNG GIÁ"] = pd.cut(raw_data["GIÁ - TRIỆU ĐỒNG"],
                                    bins=[0, 2000, 4000, 6000, 8000, np.inf],
                                    #labels=["<2 tỷ", "2-4 tỷ", "4-6 tỷ", "6-8 tỷ", "8-10 tỷ", ">10 tỷ"])
                                    labels=[2,4,6,8,100]) # use numeric labels to plot histogram
    
# Tạo training và test set
from sklearn.model_selection import StratifiedShuffleSplit  
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42) # n_splits: no. of re-shuffling & splitting = no. of train-test sets 
                                                                                  # (if you want to run the algorithm n_splits times with different train-test set)
for train_index, test_index in splitter.split(raw_data, raw_data["KHOẢNG GIÁ"]): # Feature "KHOẢNG GIÁ" must NOT contain NaN
        train_set = raw_data.loc[train_index]
        test_set = raw_data.loc[test_index]      

# Xóa feature "KHOANG GIA"
print(train_set.info())
for _set_ in (train_set, test_set):
        #_set_.drop("income_cat", axis=1, inplace=True) # axis=1: drop cols, axis=0: drop rows
    _set_.drop(columns="KHOẢNG GIÁ", inplace=True) 
    
# Thông tin về tập train và test    
print(train_set.info())
print(test_set.info()) 
print(len(train_set), "training +", len(test_set), "test examples")
print(train_set.head(4))

#%% 4.3 Separate labels from data, since we do not process label values
train_set_labels = train_set["GIÁ - TRIỆU ĐỒNG"].copy()
train_set = train_set.drop(columns = "GIÁ - TRIỆU ĐỒNG") 
test_set_labels = test_set["GIÁ - TRIỆU ĐỒNG"].copy()
test_set = test_set.drop(columns = "GIÁ - TRIỆU ĐỒNG") 

#%% XỬ LÝ DỮ LIỆU 

# Những feature có giá trị thuộc kiểu số
num_feat_names = ['DIỆN TÍCH - M2', 'SỐ PHÒNG', 'SỐ TOILETS']

# Những feature có giá trị thuộc kiểu chữ
cat_feat_names = ['QUẬN HUYỆN', 'HƯỚNG', 'GIẤY TỜ PHÁP LÝ']

class ColumnSelector(BaseEstimator, TransformerMixin): # Hàm xử lý dữ liệu chữ
    def __init__(self, feature_names):
        self.feature_names = feature_names
    def fit(self, dataframe, labels=None):
        return self
    def transform(self, dataframe):
        return dataframe[self.feature_names].values         

# Bộ xử lý kiểu giá trị chữ
cat_pipeline = Pipeline([
    ('selector', ColumnSelector(cat_feat_names)),
    ('imputer', SimpleImputer(missing_values=np.nan, strategy="constant", fill_value = "NO INFO", copy=True)), # complete missing values. copy=False: imputation will be done in-place 
    ('cat_encoder', OneHotEncoder()) # Chuyển dữ liệu thành dạng One Hot Vector
    ])    

# Hàm xử lý kiểu dữ liệu số 
class MyFeatureAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_TONG_SO_PHONG = True): # MUST NO *args or **kargs
        self.add_TONG_SO_PHONG = add_TONG_SO_PHONG
    def fit(self, feature_values, labels = None):
        return self  # nothing to do here
    def transform(self, feature_values, labels = None):
        if self.add_TONG_SO_PHONG:        
            SO_PHONG_id, SO_TOILETS_id = 1, 2 # column indices in num_feat_names. can't use column names b/c the transformer SimpleImputer removed them
            # NOTE: a transformer in a pipeline ALWAYS return dataframe.values (ie., NO header and row index)
            TONG_SO_PHONG = feature_values[:, SO_PHONG_id] + feature_values[:, SO_TOILETS_id]
            feature_values = np.c_[feature_values, TONG_SO_PHONG] #concatenate np arrays
        return feature_values

# 4.4.4 Pipeline for numerical features
num_pipeline = Pipeline([
    ('selector', ColumnSelector(num_feat_names)),
    ('imputer', SimpleImputer(missing_values=np.nan, strategy="median", copy=True)), # copy=False: imputation will be done in-place 
    ('attribs_adder', MyFeatureAdder(add_TONG_SO_PHONG = True)),
    ('std_scaler', StandardScaler(with_mean=True, with_std=True, copy=True)) # Scale features để thu hẹp khoảng cách của các giá trị trong feature
    ])  
  
# Gom 2 cái trê thành 1 cái pipeline tổng
full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline) ])  


# Áp dụng pipeline vào bộ train_set để xử lý chúng         
processed_train_set_val = full_pipeline.fit_transform(train_set)
print('\n____________ Processed feature values ____________')
print(processed_train_set_val[[0, 1, 2],:].toarray())
print(processed_train_set_val.shape)
print('We have %d numeric feature + 1 added features + 35 cols of onehotvector for categorical features.' %(len(num_feat_names)))
joblib.dump(full_pipeline, r'models/full_pipeline.pkl')


#%% 5.3 Try RandomForestRegressor model
# Training (NOTE: may take time if train_set is large)
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators = 5) # n_estimators: no. of trees
model.fit(processed_train_set_val, train_set_labels)
 
# Cho nó làm bài kiểm tra nhỏ với tập validate
print("\nPredictions: ", model.predict(processed_train_set_val[0:9]).round(decimals=1))
print("Labels:      ", list(train_set_labels[0:9]))


#%% 5.5 Evaluate with K-fold cross validation 
from sklearn.model_selection import cross_val_score

model_name = "RandomForestRegressor" 
rmse_scores = joblib.load('saved_objects/' + model_name + '_rmse.pkl')
print("RandomForestRegressor rmse: ", rmse_scores.round(decimals=1))
print("Avg. rmse: ", mean(rmse_scores.round(decimals=1)),'\n')



#%% FINE-TUNE MODELS 
# Dùng grid_search để tìm ra những hyperparameter tốt nhất của model Randomforest
# từ đó cải thiện model tốt hơn nữa

print('\n____________ Fine-tune models ____________')
def print_search_result(grid_search, model_name = ""): 
    print("\n====== Fine-tune " + model_name +" ======")
    print('Best hyperparameter combination: ',grid_search.best_params_)
    print('Best rmse: ', np.sqrt(-grid_search.best_score_))  
    #print('Best estimator: ', grid_search.best_estimator_) # NOTE: require refit=True in  SearchCV
    print('Performance of hyperparameter combinations:')
    cv_results = grid_search.cv_results_
    for (mean_score, params) in zip(cv_results["mean_test_score"], cv_results["params"]):
        print('rmse =', np.sqrt(-mean_score).round(decimals=1), params) 

from sklearn.model_selection import GridSearchCV
cv = KFold(n_splits=5,shuffle=True,random_state=37) # cv data generator



run_new_search = 1      
if run_new_search:        
        # 6.1.1 Fine-tune RandomForestRegressor
        model = RandomForestRegressor()
        param_grid = [
            # try 15 (3x4) combinations of hyperparameters (bootstrap=True: drawing samples with replacement)
            {'bootstrap': [True], 'n_estimators': [3, 15, 30], 'max_features': [2, 12, 20, 39]},
            # then try 12 (4x3) combinations with bootstrap set as False
            {'bootstrap': [False], 'n_estimators': [3, 5, 10, 20], 'max_features': [2, 6, 10]} ]
            # Train across 5 folds, hence a total of (15+12)*5=135 rounds of training 
        grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='neg_mean_squared_error', return_train_score=True, 
        refit=True) # refit=True: after finding best hyperparam, it fit() the model with whole data (hope to get better result)
        grid_search.fit(processed_train_set_val, train_set_labels)
        joblib.dump(grid_search,'saved_objects/RandomForestRegressor_gridsearch.pkl')
        print_search_result(grid_search, model_name = "RandomForestRegressor")      

        # 6.1.2 Fine-tune Polinomial regression          
        '''model = Pipeline([ ('poly_feat_adder', PolynomialFeatures()), # add high-degree features
                           ('lin_reg', LinearRegression()) ]) 
        param_grid = [
            # try 3 values of degree
            {'poly_feat_adder__degree': [1, 2, 3]} ] # access param of a transformer: <transformer>__<parameter> https://scikit-learn.org/stable/modules/compose.html
            # Train across 5 folds, hence a total of 3*5=15 rounds of training 
        grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='neg_mean_squared_error', return_train_score=True)
        grid_search.fit(processed_train_set_val, train_set_labels)
        #joblib.dump(grid_search,'saved_objects/PolinomialRegression_gridsearch.pkl') 
        #print_search_result(grid_search, model_name = "PolinomialRegression") '''
else:
        # Load grid_search
        grid_search = joblib.load('saved_objects/RandomForestRegressor_gridsearch.pkl')
        print_search_result(grid_search, model_name = "RandomForestRegressor")         
        #grid_search = joblib.load('saved_objects/PolinomialRegression_gridsearch.pkl')
        #print_search_result(grid_search, model_name = "PolinomialRegression") 

#%% Lưu lại model tốt nhất của tốt nhất: best_model
search = joblib.load('saved_objects/RandomForestRegressor_gridsearch.pkl')
best_model = search.best_estimator_

#%%
if 0:
    feature_importances = best_model.feature_importances_
    onehot_cols = []
    for val_list in full_pipeline.transformer_list[1][1].named_steps['cat_encoder'].categories_: 
        onehot_cols = onehot_cols + val_list.tolist()
    feature_names = train_set.columns.tolist() + ["TỔNG SỐ PHÒNG"] + onehot_cols
    for name in cat_feat_names:
        feature_names.remove(name)
    print('\nFeatures and importance score: ')
    print(*sorted(zip( feature_names, feature_importances.round(decimals=4)), key = lambda row: row[1], reverse=True),sep='\n')

#%% Chạy thử với bộ Test
# 7.3 Run on test data
if 0:
    full_pipeline = joblib.load(r'models/full_pipeline.pkl')
    processed_test_set = full_pipeline.transform(test_set)  
    # 7.3.2 Predict labels for some test instances
    print("\nTest data: \n", test_set.iloc[0:9])
    print("\nPredictions: ", best_model.predict(processed_test_set[0:9]).round(decimals=1))
    print("Labels:      ", list(test_set_labels[0:9]),'\n')
#%% Hàm chuyển đổi giọng nói sang data
def chuyen_doi(text):
    key = []
    if 1:
        if "thủ đức" in text:
            key.append("Quận Thủ Đức")
        elif "quận 9" in text:
            key.append("Quận 9")
        elif "tân bình" in text:
            key.append("Quận Tân Bình")
        elif "tân phú" in text:
            key.append("Quận Tân Phú")
        elif "quận 11" in text:
            key.append("Quận 11")
        elif "quận 7" in text:
            key.append("Quận 7")
        elif "quận 2" in text:
            key.append("Quận 2")
        elif "bình chánh" in text:
            key.append("Huyện Bình Chánh")            
        elif "bình tân" in text:
            key.append("Quận Bình Tân")
        elif "quận 12" in text:
            key.append("Quận 12")
        elif "quận 5" in text:
            key.append("Quận 5")
        elif "quận 8" in text:
            key.append("Quận 8")
        elif "phú nhuận" in text:
            key.append("Quận Phú Nhuận")
        elif "bình thạnh" in text:
            key.append("Quận Bình Thạnh")
        elif "nhà bè" in text:
            key.append("Huyện Nhà Bè")            
        elif "quận 6" in text:
            key.append("Quận 6")
        elif "quận 10" in text:
            key.append("Quận 10")
        elif "gò vấp" in text:
            key.append("Quận Gò Vấp")
        elif "quận 4" in text:
            key.append("Quận 4")
        elif "quận 1" in text:
            key.append("Quận 1")
        elif "quận 3" in text:
            key.append("Quận 3")
        elif "hóc môn" in text:
            key.append("Huyện Hóc Môn")
        else:
            key.append(np.nan)
    if 2:
        if 'mét' in text:
            key.append(lay_so_lieu(text, 'mét'))
        else:
            key.append(np.nan)          
    if 3:
        if "đông nam" in text:
            key.append("Đông Nam")
        elif "tây nam" in text:
            key.append("Tây Nam")
        elif "tây" in text:
            key.append("Tây")
        elif "đông bắc" in text:
            key.append("Đông Bắc")
        elif "nam" in text:
            key.append("Nam")
        elif "bắc" in text:
            key.append("Bắc")
        elif "đông" in text:
            key.append("Đông")  
        elif "tây bắc" in text:
            key.append("Tây Bắc")       
        else:
            key.append(np.nan)
    if 4:
        if 'phòng' in text:
            key.append(lay_so_lieu(text, 'phòng'))
        else:
            key.append(np.nan)  
    if 5:
        if 'toilet' in text:
            key.append(lay_so_lieu(text, 'toilet'))
        else:
            key.append(np.nan)  
    if 6:
        if "chưa có sổ" in text:
            key.append("Đang chờ sổ")
        elif "có sổ" in text:
            key.append("Đã có sổ")
        else:
            key.append(np.nan)
    return key    

#%% Thu âm
def get_audio():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Tôi: ", end='')
        audio = r.listen(source, phrase_time_limit=6)
        try:
            text = r.recognize_google(audio, language="vi-VN")
            print(text)
            if "Cảm ơn" in text:
                speak("Hẹn gặp lại bạn")
                os._exit(2)
            return text
        except:
            print("...")
            return '0'
#%% Nói
def speak(text):
    print("Anya: {}".format(text))
    tts = gTTS(text=text, lang=language, slow=False)
    tts.save("sound.mp3")
    playsound.playsound("sound.mp3", True)
    os.remove("sound.mp3")

#%% Các hàm cắt gọt dữ liệu giọng nói

def doi_chu_so(num):
    arr = ['không', 'một', 'hai', 'ba',
           'bốn', 'năm', 'sáu',
           'bảy', 'tám', 'chín']
    if num in arr:
        return str(index_text(arr, num))
    return num  

def vi_tri_cua_tu(list_text, tu): # vị trí của từ cần kiếm ở đâu trong list
    i = 0
    while i < len(list_text):
        if list_text[i] == tu:
            return i
        i += 1
    return np.nan

def lay_so_lieu(text, tu):
    list_text = text.split()
    vi_tri_tu = vi_tri_cua_tu(list_text, tu)
    so = list_text[vi_tri_tu-1] # Số liệu cần tìm
    so = doi_chu_so(so) # đổi chữ số (dạng chữ) sang chữ số dạng số
    try:
        S = float(so)
        return S
    except:
        speak("Có lẽ anh đã nói nhầm từ " + so + " " + list_text[vi_tri_tu] + " rồi")
        return np.nan
    
#%%
def doan_gia_nha(test_set):
    try:
        feature = chuyen_doi(get_audio().lower())
        
        new_test  = pd.Series(data = [feature[0], feature[1], feature[2],
                                feature[3], feature[4], feature[5]],
                        index = test_set.columns, name = 2000)
        
        test_set = test_set.append(new_test)
        full_pipeline = joblib.load(r'models/full_pipeline.pkl')
        processed_test_set = full_pipeline.transform(test_set[-1:]) 
        gia_du_doan = best_model.predict(processed_test_set[0]).round(decimals=1)
        gia_du_doan = "Em nghĩ giá của nó đâu khoảng " + (str(gia_du_doan).strip('[]')).replace(".", "") + "000000 đồng"
        speak(gia_du_doan)
        test_set = test_set.drop(labels=[2000])
    except:
        speak("Em nghe không rõ, anh nói lại đi")
        doan_gia_nha(test_set)

doan_gia_nha(test_set)
#%%
'''
import pandas as pd
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
import urllib.request as urllib2`
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
from time import strftime
from gtts import gTTS
from youtube_search import YoutubeSearch
import pandas as pd
wikipedia.set_lang('vi')
language = 'vi'
path = ChromeDriverManager().install()

#PART 1. NHẬN DỮ LIỆU GIÁ NHÀ
raw_data = pd.read_csv('datasets\GiaChungCu_HCM_June2021_laydulieu_com.csv')
raw_data.drop(columns = ["GIỐNG - LOẠI", "GIỐNG - NHU CẦU",
                         "GIỐNG - TỈNH THÀNH", "SỐ TẦNG"], inplace=True) 

raw_data["KHOẢNG GIÁ"] = pd.cut(raw_data["GIÁ - TRIỆU ĐỒNG"],
                                    bins=[0, 2000, 4000, 6000, 8000, np.inf],
                                    #labels=["<2 tỷ", "2-4 tỷ", "4-6 tỷ", "6-8 tỷ", "8-10 tỷ", ">10 tỷ"])
                                    labels=[2,4,6,8,100]) # use numeric labels to plot histogram

# Tạo training và test set
from sklearn.model_selection import StratifiedShuffleSplit  
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42) # n_splits: no. of re-shuffling & splitting = no. of train-test sets 
                                                                                  # (if you want to run the algorithm n_splits times with different train-test set)
for train_index, test_index in splitter.split(raw_data, raw_data["KHOẢNG GIÁ"]): # Feature "KHOẢNG GIÁ" must NOT contain NaN
        train_set = raw_data.loc[train_index]
        test_set = raw_data.loc[test_index]  
test_set = test_set.drop(columns = "GIÁ - TRIỆU ĐỒNG") 
test_set.drop(columns = "KHOẢNG GIÁ",  inplace=True)

def get_audio():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Tôi: ", end='')
        audio = r.listen(source, phrase_time_limit=6)
        try:
            text = r.recognize_google(audio, language="vi-VN")
            print(text)
            if "Cảm ơn" in text:
                speak("Hẹn gặp lại bạn")
                os._exit(2)
            return text
        except:
            print("...")
            return '0'
#Nói
def speak(text):
    print("Anya: {}".format(text))
    tts = gTTS(text=text, lang=language, slow=False)
    tts.save("sound.mp3")
    playsound.playsound("sound.mp3", True)
    os.remove("sound.mp3")

#Các hàm cắt gọt dữ liệu giọng nói
def chuyen_doi(text):
    key = []
    if 1:
        if "thủ đức" in text:
            key.append("Quận Thủ Đức")
        elif "quận 9" in text:
            key.append("Quận 9")
        elif "tân bình" in text:
            key.append("Quận Tân Bình")
        elif "tân phú" in text:
            key.append("Quận Tân Phú")
        elif "quận 11" in text:
            key.append("Quận 11")
        elif "quận 7" in text:
            key.append("Quận 7")
        elif "quận 2" in text:
            key.append("Quận 2")
        elif "bình chánh" in text:
            key.append("Huyện Bình Chánh")            
        elif "bình tân" in text:
            key.append("Quận Bình Tân")
        elif "quận 12" in text:
            key.append("Quận 12")
        elif "quận 5" in text:
            key.append("Quận 5")
        elif "quận 8" in text:
            key.append("Quận 8")
        elif "phú nhuận" in text:
            key.append("Quận Phú Nhuận")
        elif "bình thạnh" in text:
            key.append("Quận Bình Thạnh")
        elif "nhà bè" in text:
            key.append("Huyện Nhà Bè")            
        elif "quận 6" in text:
            key.append("Quận 6")
        elif "quận 10" in text:
            key.append("Quận 10")
        elif "gò vấp" in text:
            key.append("Quận Gò Vấp")
        elif "quận 4" in text:
            key.append("Quận 4")
        elif "quận 1" in text:
            key.append("Quận 1")
        elif "quận 3" in text:
            key.append("Quận 3")
        elif "hóc môn" in text:
            key.append("Huyện Hóc Môn")
        else:
            key.append(np.nan)
    if 2:
        if 'mét' in text:
            key.append(lay_so_lieu(text, 'mét'))
        else:
            key.append(np.nan)          
    if 3:
        if "đông nam" in text:
            key.append("Đông Nam")
        elif "tây nam" in text:
            key.append("Tây Nam")
        elif "tây" in text:
            key.append("Tây")
        elif "đông bắc" in text:
            key.append("Đông Bắc")
        elif "nam" in text:
            key.append("Nam")
        elif "bắc" in text:
            key.append("Bắc")
        elif "đông" in text:
            key.append("Đông")  
        elif "tây bắc" in text:
            key.append("Tây Bắc")       
        else:
            key.append(np.nan)
    if 4:
        if 'phòng' in text:
            key.append(lay_so_lieu(text, 'phòng'))
        else:
            key.append(np.nan)  
    if 5:
        if 'toilet' in text:
            key.append(lay_so_lieu(text, 'toilet'))
        else:
            key.append(np.nan)  
    if 6:
        if "chưa có sổ" in text:
            key.append("Đang chờ sổ")
        elif "có sổ" in text:
            key.append("Đã có sổ")
        else:
            key.append(np.nan)
    return key    

def doi_chu_so(num):
    arr = ['không', 'một', 'hai', 'ba',
           'bốn', 'năm', 'sáu',
           'bảy', 'tám', 'chín']
    if num in arr:
        return str(index_text(arr, num))
    return num  

def vi_tri_cua_tu(list_text, tu): # vị trí của từ cần kiếm ở đâu trong list
    i = 0
    while i < len(list_text):
        if list_text[i] == tu:
            return i
        i += 1
    return np.nan

def lay_so_lieu(text, tu):
    list_text = text.split()
    vi_tri_tu = vi_tri_cua_tu(list_text, tu)
    so = list_text[vi_tri_tu-1] # Số liệu cần tìm
    so = doi_chu_so(so) # đổi chữ số (dạng chữ) sang chữ số dạng số
    try:
        S = float(so)
        return S
    except:
        speak("Có lẽ anh đã nói nhầm từ " + so + " " + list_text[vi_tri_tu] + " rồi")
        return np.nan
  
def doan_gia_nha(test_set):
    try:
        feature = chuyen_doi(get_audio().lower())
        
        new_test  = pd.Series(data = [feature[0], feature[1], feature[2],
                                feature[3], feature[4], feature[5]],
                        index = test_set.columns, name = 2000)
        
        test_set = test_set.append(new_test)
        full_pipeline = joblib.load(r'models/full_pipeline.pkl')
        processed_test_set = full_pipeline.transform(test_set[-1:]) 
        gia_du_doan = best_model.predict(processed_test_set[0]).round(decimals=1)
        gia_du_doan = "Em nghĩ giá của nó đâu khoảng " + (str(gia_du_doan).strip('[]')).replace(".", "") + "000000 đồng"
        speak(gia_du_doan)
        test_set = test_set.drop(labels=[2000])
    except:
        speak("Em nghe không rõ, anh nói lại đi")
        doan_gia_nha(test_set)
# %%
'''