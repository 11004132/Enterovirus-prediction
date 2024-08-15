import pmdarima as pm
from pmdarima.model_selection import train_test_split
from pmdarima import model_selection
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
from matplotlib import font_manager
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np
acc=[]
def year_week_to_date(year_week_int):

    year_week_str = str(year_week_int)
    year = int(year_week_str[:4])
    week = int(year_week_str[4:])
    

    first_day_of_year = datetime(year, 1, 1)
    

    first_week_start = first_day_of_year - timedelta(days=first_day_of_year.weekday())
    

    target_week_start = first_week_start + timedelta(weeks=week-1)
    
    return target_week_start
Data=pd.read_csv("/home/nick/Downloads/20081-202430_全國及六區門診腸病毒每週就診人次.csv")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP'] 
train = Data[:len(Data)-50]
train.index=train['就診年週']
train=train.iloc[:,[4]]
test = Data[len(Data)-50:len(Data)]
test.index=test['就診年週']
test=test.iloc[:,[4]]
    #arima_model=pm.auto_arima(train,start_p=0,max_q=5,seasonal=True, m=12,trace=True,stepwise=True)
arima_model =pm.auto_arima(train,
                          start_P=2, D=2, start_Q=5,max_P=2, m=12,max_Q=5,seasonal=True,
                           stationary=False,
                           error_action='warn', trace=True,
                           suppress_warnings=True, stepwise=True,n_fits=10)


prediction = pd.DataFrame(arima_model.predict(n_periods=50))
prediction.index=[year_week_to_date(test.index[i]) for i in range(len(test.index))]
train.index=[year_week_to_date(train.index[i]) for i in range(len(train.index))]
test.index=[year_week_to_date(test.index[i]) for i in range(len(test.index))]
plt.figure(figsize=(14, 7)) 
prediction.columns = ['predicted_value']
plt.plot(train, label="Training")
plt.plot(test, label="Test")
plt.plot(prediction, label="Predicted")
plt.title("中區")  
plt.legend(loc='upper right')
plt.show()
