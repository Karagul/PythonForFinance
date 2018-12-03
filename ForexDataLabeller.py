# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np

#Data location.
location = "C:\\Users\\User\\Projects\\MT4\Data\\EURUSDhourly26_10_2018.csv"
location_daily= "C:\\Users\\User\\Projects\\MT4\Data\\EURUSDdaily26_10_2018.csv"

#Read file into dataframe.
df_hourly = pd.read_csv(location,names=['Date','Time','Open','High','Low','Close','Volume'],
                 parse_dates=[['Date','Time']],index_col='Date_Time')

df_daily = pd.read_csv(location_daily,names=['Date','Time','Open','High','Low','Close','Volume'],
                       parse_dates=['Date'],index_col='Date')
                       
#Create 'Bias' column.
df_hourly['Bias'] = np.where(df_hourly['Close'] > df_hourly['Open'],1,0)
df_daily['Bias'] = np.where(df_daily['Close'] > df_daily['Open'],1,0) 

#Create pivot points columns.
df_daily = df_daily.reset_index()

df_daily['PivotPoint'] = 0.0
df_daily['Resistance1'] = 0.0
df_daily['Resistance2'] = 0.0
df_daily['Resistance3'] = 0.0
df_daily['Support1'] = 0.0
df_daily['Support2'] = 0.0
df_daily['Support3'] = 0.0
#df_daily['PivotPoint'].astype(np.float64)

count = 1

while count <= df_daily.iloc[-1].name:
    df_daily.at[count,'PivotPoint'] = (df_daily.iloc[count-1]['High'] + 
            df_daily.iloc[count-1]['Low'] + 
            df_daily.iloc[count-1]['Close']) / 3
    df_daily.at[count,'Resistance1'] = (2 * df_daily.iloc[count]['PivotPoint']) - df_daily.iloc[count-1]['Low']
    df_daily.at[count,'Support1'] = (2 * df_daily.iloc[count]['PivotPoint']) - df_daily.iloc[count-1]['High']
    df_daily.at[count,'Resistance2'] = df_daily.iloc[count]['PivotPoint'] + (df_daily.iloc[count-1]['High'] - df_daily.iloc[count-1]['Low'])
    df_daily.at[count,'Support2'] = df_daily.iloc[count]['PivotPoint'] - (df_daily.iloc[count-1]['High'] - df_daily.iloc[count-1]['Low'])
    df_daily.at[count,'Resistance3'] = df_daily.iloc[count-1]['High'] + 2*(df_daily.iloc[count]['PivotPoint'] - df_daily.iloc[count-1]['Low'])
    df_daily.at[count,'Support3'] = df_daily.iloc[count-1]['Low'] - 2*(df_daily.iloc[count-1]['High'] - df_daily.iloc[count]['PivotPoint'])
    count += 1

   
'''
Pivot point (PP) = (High + Low + Close) / 3.
First resistance (R1) = (2 x PP) – Low.
First support (S1) = (2 x PP) – High.
Second resistance (R2) = PP + (High – Low)
Second support (S2) = PP – (High – Low)
Third resistance (R3) = High + 2(PP – Low)
Third support (S3) = Low – 2(High – PP)
'''