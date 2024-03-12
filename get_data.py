import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import json
import requests
import pandas_ta as ta



def get_rsidata(ticker, start):
    response = requests.get('https://api.senticrypt.com/v2/all.json')
    if response.status_code == 200:
        data = response.json()
        print(data)
    else:
        print(f"Error: {response.status_code}")


    df = pd.DataFrame(data)
    df = df.iloc[::-1]
    df['Date'] = pd.to_datetime(df['date'])
    df.drop(columns=['date'], inplace = True)
    
    

    
    data1 = yf.download(tickers = ticker, start = start,end = end)
    data = pd.merge(data1, df, on = 'Date')
    data.drop(columns=['price','volume','score1','score2','score3','count',	'sum'], inplace = True)
    data['RSI']=ta.rsi(data.Close, length=15)
    data['EMAF']=ta.ema(data.Close, length=20)
    data['EMAM']=ta.ema(data.Close, length=100)
    data['EMAS']=ta.ema(data.Close, length=150)

    data['Target'] = data['Adj Close']-data.Open
    data['Target'] = data['Target'].shift(-1)

    data['TargetClass'] = [1 if data.Target[i]>0 else 0 for i in range(len(data))]

    data['TargetNextClose'] = data['Adj Close'].shift(-1)

    data.dropna(inplace=True)
    data.reset_index(inplace = True)
    data.drop(['Volume', 'Close','Date','index'], axis=1, inplace=True)
    data_set = data.iloc[:, 0:12]
    pd.set_option('display.max_columns', None)
    
    return data_set


def get_adjclose(ticker, s):
    df = yf.download(tickers = ticker, start = s,end = str(pd.Timestamp.today()).split(' ')[0])
    df = df.drop(columns=['Open','High','Low','Close','Volume'])
    df = df.dropna()
    return df