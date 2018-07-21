# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 19:52:59 2018

@author: Tim George Kabue
@email: timkabue@yahoo.com
@phone_number: +2540706762054
"""
import matplotlib.pyplot as plt
from mpl_finance import candlestick_ohlc
import matplotlib.dates as mdates
import pandas as pd

#******Retrieve financial data.******
df = pd.read_csv("C:/Users/User/Projects/MT4/Data/EURUSD15.csv",
                 names=['Date','Time','Open','High','Low','Close','Volume'],                
                 parse_dates=[['Date','Time']],
                 index_col=['Date_Time'])

#******Preprocess date format for candlestick_ohlc.****** 
df['Dates'] = df.index.map(mdates.date2num) 

ohlc = df[['Dates','Open','High','Low','Close']].tail(200)

#******Draw the candlesticks.******
f1 , ax = plt.subplots(2,1)

#******Plot the candlesticks.******
(lines,patches) = candlestick_ohlc(ax[0],ohlc.values,width=.6/(24*4),colorup='green',colordown='red')
ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))

#******Plot volume.******
volume = df['Volume'].tail(200)
ax[1].fill_between(volume.index.map(mdates.date2num),volume.values,0)
ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))


plt.show()

