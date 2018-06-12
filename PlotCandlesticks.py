# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 00:51:38 2018

@author: Tim George Kabue
@email: timkabue@yahoo.com
@phone_number: +2540706762054
"""
import matplotlib.pyplot as plt
from mpl_finance import candlestick_ohlc
import matplotlib.dates as mdates
import pandas as pd

#******Retrieve financial data.******
df = pd.read_csv("C:/Users/User/Projects/MT4/Data/EURUSD1_latest.csv",
                 names=['Date','Time','Open','High','Low','Close','Volume'],                
                 parse_dates=[['Date','Time']],
                 index_col=['Date_Time'])

#******Preprocess date format for candlestick_ohlc.****** 
df['Dates'] = df.index.map(mdates.date2num) 

ohlc = df[['Dates','Open','High','Low','Close']].tail(200)

#******Draw the candlesticks.******
f1 , ax = plt.subplots(figsize=(10,5))

#******Plot the candlesticks.******
candlestick_ohlc(ax,ohlc.values,width=.6/(24*60),colorup='green',colordown='red')

ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))

plt.show()