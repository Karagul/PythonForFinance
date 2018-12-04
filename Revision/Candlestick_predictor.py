# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 14:36:35 2018

@author: Tim George Kabue
@email: timkabue@yahoo.com
@phone_number: +254706762054
@title: Prediction of bias of next candlestick.
"""

import pandas as pd
import numpy as np
import talib
import matplotlib.dates as mdates

#Retrieve data.
#*****************************************************************************

location = "C:\\Users\\User\\Projects\\MT4\\Data\\EURUSDhourly4_12_2018.csv"
location_daily = "C:\\Users\\User\\Projects\\MT4\\Data\\EURUSDdaily4_12_2018.csv"
df = pd.read_csv(location,names=['Date','Time','Open','High','Low','Close','Volume'])
df_daily = pd.read_csv(location_daily,names=['Date','Time','Open','High','Low','Close','Volume'])

#******Retrieve financial data.******
df1 = pd.read_csv(location,
                 names=['Date','Time','Open','High','Low','Close','Volume'],                
                 parse_dates=[['Date','Time']],
                 index_col=['Date_Time'])

#******Preprocess date format for candlestick_ohlc.****** 
df1['Dates'] = df1.index.map(mdates.date2num)
df1 = df1.reset_index() 

df['Dates'] = df1['Dates']

#******************************************************************************

#Create columns.
#******************************************************************************
'''
1. Bias (1 column)
2. EMA 10 and 20 (2 columns)
3. Size of candlestick: range,body,top-wick,bottom-wick (4 columns)
4. Pivotpoints (7 columns)
5. Pivotpoints distances: Open,Close,High.Low (7*4=28 columns)
'''
# 1. Bias
condition = df['Close'] > df['Open']
df['Bias'] = np.where(condition,1,0)

#2. Exponential Moving Averages: 10period and 20period
close = np.asarray(df['Close'])
df['EMA_10'] = talib.EMA(close,10)
df['EMA_20'] = talib.EMA(close,20)

#3. Size of candlestick.
#========================
#Range.
df['Range'] = df['High'] - df['Low']

#Body.
index = 0
while index <= df.iloc[-1].name:
    if df.iloc[index]['Bias'] == 1:
        df.at[index,'Body'] = df.iloc[index]['Close'] - df.iloc[index]['Open']
    else:
        df.at[index,'Body'] = df.iloc[index]['Open'] - df.iloc[index]['Close']
    index += 1
    
#Top wick.
index = 0
while index <= df.iloc[-1].name:
    if df.iloc[index]['Bias'] == 1:
        df.at[index,'TopWick'] = df.iloc[index]['High'] - df.iloc[index]['Close']
    else:
        df.at[index,'TopWick'] = df.iloc[index]['High'] - df.iloc[index]['Open']
    index += 1

#Bottom wick.
index = 0
while index <= df.iloc[-1].name:
    if df.iloc[index]['Bias'] == 1:
        df.at[index,'BottomWick'] = df.iloc[index]['Open'] - df.iloc[index]['Low']
    else:
        df.at[index,'BottomWick'] = df.iloc[index]['Close'] - df.iloc[index]['Low']
    index += 1
#========================
    
#4. Pivotpoints.
#=========================================
#Calculate pivotpoints from daily charts.
index = 1

while index <= df_daily.iloc[-1].name:
    df_daily.at[index,'PivotPoint'] = (df_daily.iloc[index-1]['High'] + 
            df_daily.iloc[index-1]['Low'] + 
            df_daily.iloc[index-1]['Close']) / 3
    df_daily.at[index,'Resistance1'] = (2 * df_daily.iloc[index]['PivotPoint']) - df_daily.iloc[index-1]['Low']
    df_daily.at[index,'Support1'] = (2 * df_daily.iloc[index]['PivotPoint']) - df_daily.iloc[index-1]['High']
    df_daily.at[index,'Resistance2'] = df_daily.iloc[index]['PivotPoint'] + (df_daily.iloc[index-1]['High'] - df_daily.iloc[index-1]['Low'])
    df_daily.at[index,'Support2'] = df_daily.iloc[index]['PivotPoint'] - (df_daily.iloc[index-1]['High'] - df_daily.iloc[index-1]['Low'])
    df_daily.at[index,'Resistance3'] = df_daily.iloc[index-1]['High'] + 2*(df_daily.iloc[index]['PivotPoint'] - df_daily.iloc[index-1]['Low'])
    df_daily.at[index,'Support3'] = df_daily.iloc[index-1]['Low'] - 2*(df_daily.iloc[index-1]['High'] - df_daily.iloc[index]['PivotPoint'])
    index += 1

#Assign daily pivotpoints to hourly candlesticks.
index = 1

while index <= df_daily.iloc[-1].name:
    temp_df = df.loc[df['Date'] == df_daily.iloc[index]['Date']] #Temporary dataframe containing rows matching dates.
    if len(temp_df.index) != 0:
        for row_index in temp_df.index:
            df.at[row_index,'PivotPoint'] = df_daily.iloc[index]['PivotPoint']
            df.at[row_index,'Resistance1'] = df_daily.iloc[index]['Resistance1']
            df.at[row_index,'Support1'] = df_daily.iloc[index]['Support1']
            df.at[row_index,'Resistance2'] = df_daily.iloc[index]['Resistance2']
            df.at[row_index,'Support2'] = df_daily.iloc[index]['Support2']
            df.at[row_index,'Resistance3'] = df_daily.iloc[index]['Resistance3']
            df.at[row_index,'Support3'] = df_daily.iloc[index]['Support3']
    index += 1

#=========================================
    
#5. Pivotpoint distances.
index = 0
column_names = [['H_PP','H_R1','H_R2','H_R3','H_S1','H_S2','H_S3'],
                ['L_PP','L_R1','L_R2','L_R3','L_S1','L_S2','L_S3'],
                ['O_PP','O_R1','O_R2','O_R3','O_S1','O_S2','O_S3'],
                ['C_PP','C_R1','C_R2','C_R3','C_S1','C_S2','C_S3']]
pivotpoint_names = ['PivotPoint','Resistance1','Resistance2','Resistance3','Support1','Support2','Support3']
candlestick_values = ['High','Low','Open','Close']
while index <= df.iloc[-1].name:
    print("****I am calculating distances at: ",index)
    group = 0
    while group <= len(column_names)-1:
        print("I am calculating group: ",group)
        for count,pivotpoint_name in enumerate(pivotpoint_names,0):
            df.at[index,column_names[group][count]] = df.iloc[index][candlestick_values[group]] - df.iloc[index][pivotpoint_name]
        group += 1    
    index += 1
#*******************************************************************************************************************************
    
#Sample labels.
df['Label'] = 0
index = 0
point = 0.0001

while index < df.iloc[-1].name:
    if df.iloc[index]['Bias'] == df.iloc[index+1]['Bias']:
        if df.iloc[index]['Bias'] == 1 and df.iloc[index+1]['TopWick'] + df.iloc[index+1]['Body'] >= 10*point and df.iloc[index+1]['BottomWick'] <= 5*point:
            df.at[index,'Label'] = 1
        elif df.iloc[index]['Bias'] == 0 and df.iloc[index+1]['BottomWick'] + df.iloc[index+1]['Body'] >= 10*point and df.iloc[index+1]['TopWick'] <= 5*point:
            df.at[index,'Label'] = 1
    index += 1
            
#******************************************************************************
    
#Save dataframe.
df_labels = df['Label']

df = df[['Dates','Open','High','Low','Close','Volume','Bias',
         'EMA_10', 'EMA_20','Range','Body','TopWick','BottomWick',
         'PivotPoint','Resistance1','Support1', 'Resistance2','Support2','Resistance3','Support3',
         'H_PP','H_R1','H_R2','H_R3','H_S1', 'H_S2','H_S3',
         'L_PP','L_R1','L_R2','L_R3','L_S1','L_S2','L_S3',
         'O_PP','O_R1','O_R2', 'O_R3','O_S1','O_S2','O_S3',
         'C_PP','C_R1','C_R2','C_R3','C_S1','C_S2','C_S3']]

dataset = np.asarray(df)
datalabels = np.asarray(df_labels)

np.save('latest_dataset',dataset)
np.save('latest_datalabels',datalabels)