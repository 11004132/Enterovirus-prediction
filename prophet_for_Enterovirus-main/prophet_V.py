import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from matplotlib import font_manager

#讀取csv檔
df=pd.read_csv('/home/nick/Downloads/20081-202430_全國及六區門診腸病毒每週就診人次.csv')
df_rate=pd.read_csv('/home/nick/Downloads/20081-202430全國及六區門診腸病毒每週就診率.csv')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']  


#更改成datetime格式
def year_week_to_date(year_week_int):   
    year_week_str = str(year_week_int)
    year = int(year_week_str[:4])
    week = int(year_week_str[4:])
    first_day_of_year = datetime(year, 1, 1)
    first_week_start = first_day_of_year - timedelta(days=first_day_of_year.weekday()-1)
    target_week_start = first_week_start + timedelta(weeks=week-1)
    return target_week_start


#進行預測
def prediction_G(df_all,title):
    df_all.columns = ['ds','y']  
    df_tend=df_all
    df_all.y=df_all.y.rolling(window=10).mean()
    df_all_test=df_all.iloc[800:,:]
    df_all=df_all.iloc[:800,:]
    model=Prophet()
    model.add_country_holidays('TW')
    model.fit(df_all)
    future_dates=model.make_future_dataframe(periods=66,freq="W")
    prediction=model.predict(future_dates)
    model.plot(prediction)
    plt.scatter(x=df_all.ds, y=df_all.y)
    plt.title(title)
    plt.legend(['train data dot', 'Predict','range'])
    plt.show();
    plt.figure(figsize=(14, 7))
    plt.plot(prediction['ds'],prediction['yhat'],'red')
    plt.plot(df_all['ds'],df_all['y'],'blue')
    plt.plot(df_all_test['ds'],df_all_test['y'],'green')
    plt.title(title)
    plt.legend(['model_Predict','train','test'])
    plt.show();
    
#趨勢線    
#    plt.plot(prediction['ds'],prediction['yhat'].rolling(window=100).mean(),'green')
#    plt.plot(df_tend['ds'],df_tend['y'].rolling(window=100).mean(),'blue')
    
    
#帶入函數
df['就診年週'].iloc[:]=[year_week_to_date(i) for i in df['就診年週']]
#df["就診年週"]=[df["就診年週"][i].strftime('%Y-%m-%d') for i in range(len(df["就診年週"]))]
for i in range(len(df.columns)-1):
    prediction_G(df.iloc[:,[0,i+1]],df.columns[i+1])
    prediction_G(df_rate.iloc[:,[0,i+1]],df.columns[i+1]+"(rate)")