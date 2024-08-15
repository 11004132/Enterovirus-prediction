# -*- coding: utf-8 -*-
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.font_manager import fontManager
from datetime import datetime, timedelta
import xgboost as xgb 
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np
acc=[]

#讀取csv檔
data=pd.read_csv("/home/nick/Downloads/20081-202430_全國及六區門診腸病毒每週就診人次.csv")
data_rate=pd.read_csv('/home/nick/Downloads/20081-202430全國及六區門診腸病毒每週就診率.csv')

#plt字體檔設定
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']  

#轉成datetime
def year_week_to_date(year_week_int):
    year_week_str = str(year_week_int)
    year = int(year_week_str[:4])
    week = int(year_week_str[4:])
    first_day_of_year = datetime(year, 1, 1)
    first_week_start = first_day_of_year - timedelta(days=first_day_of_year.weekday()-1)
    target_week_start = first_week_start + timedelta(weeks=week-1)
    return target_week_start

#建立資料特徵
def feature(data):
    data['year']=[int(str(i)[:4]) for i in data['就診年週']]
    data['week']=[int(str(i)[4:]) for i in data['就診年週']]
    data['就診年週'].iloc[:]=[year_week_to_date(i) for i in data['就診年週']]
    data['quarter']=pd.Series([i for i in data.iloc[:,0]]).dt.quarter
    data['month']=pd.Series([i for i in data.iloc[:,0]]).dt.month
    data['vir']=[0 for i in range(628)]+[1 for i in range(803-628)]+[0 for i in range(866-803)]
    data['hot']=[0 for i in range(len(data))]
    hot_rows=data[((data['month']>4)&(data['month']<6)) | ((data['month']>9)&(data['month']<10))].index
    data.iloc[hot_rows,13]+=1
    return data

#直接進行測試與預測
def prediction_G(data,i):
    train=data.iloc[:800,]
    test=data.iloc[800:,]
    x_train=train.drop(i, axis =1)
    y_train=train[i]
    x_test=test.drop(i, axis =1)
    y_test=test[i]
    reg = xgb.XGBRegressor(n_estimators=2000)
    reg.fit(x_train.iloc[:,1:], y_train, verbose = False)
    test['prediction'] = reg.predict(x_test.iloc[:,1:])
    plt.figure(figsize=(14, 7))
    plt.plot(test['就診年週'],test['prediction'].rolling(window=5).mean(),label='Prediction', color='red')
    plt.plot(test['就診年週'],test[i].rolling(window=5).mean(),label='Test Data', color='blue')
    plt.plot(train['就診年週'],train[i].rolling(window=5).mean(), label='Train Data', color='green')
    plt.title(i)
    plt.legend()
    plt.show()
    
#使用可知的未來特徵紀行預測
def prediction_G_f(data,i):
    train=data.iloc[:800,]
    test=data.iloc[800:,]
    x_train=train.drop(i, axis =1)
    y_train=train[i]
    x_test=test.drop(i, axis =1)
    y_test=test[i]
    reg = xgb.XGBRegressor(n_estimators=3000)
    reg.fit(x_train.iloc[:,8:], y_train, verbose = False)
    test['prediction'] = reg.predict(x_test.iloc[:,8:])
    plt.figure(figsize=(14, 7))
    plt.plot(test['就診年週'],test['prediction'].rolling(window=5).mean(),label='Prediction', color='red')
    plt.plot(test['就診年週'],test[i].rolling(window=5).mean(),label='Test Data', color='blue')
    plt.plot(train['就診年週'],train[i].rolling(window=5).mean(), label='Train Data', color='green')
    mean=[((abs(test.reset_index(drop=True)[i][q]-test['prediction'].reset_index(drop=True)[q])/test.reset_index(drop=True)[i][q])*100) for q in range(len(test))]
    acc.append(np.mean(mean))
    plt.title(i+"（只將預測資訊作為feature)")
    plt.legend()
    plt.show()
    

#帶入函數    
data_rate=feature(data_rate)    
data=feature(data)
for i in data.columns[1:8]:
#    prediction_G(data,i)
#    prediction_G_f(data,i)
    prediction_G(data_rate,i)
    prediction_G_f(data_rate,i)
    
