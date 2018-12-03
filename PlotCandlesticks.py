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
df = pd.read_csv("C:/Users/User/Projects/MT4/Data/EURUSD15.csv",
                 names=['Date','Time','Open','High','Low','Close','Volume'],                
                 parse_dates=[['Date','Time']],
                 index_col=['Date_Time'])

#******Preprocess date format for candlestick_ohlc.****** 
df['Dates'] = df.index.map(mdates.date2num)

no_of_bars = 5 

ohlc = df[['Dates','Open','High','Low','Close']].tail(no_of_bars)

high = df['High'].tail(no_of_bars)

#******Draw the candlesticks.******
#f1 , ax = plt.subplots(figsize=(10,5)) ORIGINAL
f1 , ax = plt.subplots(figsize=(10,5))

f1.suptitle("EURUSD chart M15 timeframe")
ax.set_ylabel("Price")
ax.set_xlabel("Date-Time")

#******Plot the candlesticks.******
(lines,polys) = candlestick_ohlc(ax,ohlc.values,width=.6/(24*4),colorup='green',colordown='red')

#******Plot points for object picking.******
ax.plot(df['Dates'].tail(no_of_bars).values,high.values,'o',picker=5)

ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))

ax.annotate('A bad day at the office', xy=(dt, 24), xytext=(dt, 25),
            arrowprops=dict(facecolor='black', shrink=0.05),
           )

def onpick(event):
    thisline = event.artist
    xdata = thisline.get_xdata()
    ydata = thisline.get_ydata()
    ind = event.ind
    points = tuple(zip(xdata[ind], ydata[ind]))
    print('onpick points:', points)

f1.canvas.mpl_connect('pick_event', onpick)


plt.show()