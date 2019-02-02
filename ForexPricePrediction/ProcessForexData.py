# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 23:00:28 2019

@author: Tim george Kabue
@email: timkabue@yahoo.com
@phone_number: +2540706762054
@task: Process Forex data adding features e.g moving averages,ATR,pivot points etc
"""

import pandas as pd
import numpy as np
import talib

#Load data.
location = "C:\\Users\\User\\Projects\\Forex\\PythonForFinance\\ForexPricePrediction\\GBPNZD60.csv"
location_daily = "C:\\Users\\User\\Projects\\Forex\\PythonForFinance\\ForexPricePrediction\\GBPNZD1440.csv"
df = pd.read_csv(location,names=['Date','Time','Open','High','Low','Close','Volume'])
df_daily = pd.read_csv(location_daily,names=['Date','Time','Open','High','Low','Close','Volume'])

#Create exponential moving average columns.
close = np.asarray(df['Close'])
df['EMA_10'] = talib.EMA(close,10)
df['EMA_20'] = talib.EMA(close,20)

#Create EMA difference column
df['EMA_diff'] = df['EMA_10'] - df['EMA_20']

#Create Pivot point column
df_daily['PivotPoint'] = np.nan
index = 1
while index <= df_daily.index[-1]:
    df_daily.at[index,'PivotPoint'] = (df_daily.iloc[index-1]['High'] + 
            df_daily.iloc[index-1]['Low'] + 
            df_daily.iloc[index-1]['Close']) / 3
    index += 1
df_daily = df_daily.dropna()
df_daily = df_daily.set_index('Date'); df_daily = df_daily.drop('Time',axis=1);

#Assign PivotPoints to hourly chart.
grouped = df.groupby('Date')

for date,group in grouped:
    if date in df_daily.index:
        df.at[group.index,'PivotPoint'] = df_daily.loc[date,'PivotPoint']

#Create PivotPoint distance column.
df['PP_Close'] = df['Close'] - df['PivotPoint']

#Find moving average crossovers.
crossing,crossing_up,crossing_down = [],[],[]
index = 1
while index <= df.index[-1]:
    if df.iloc[index]['EMA_diff'] > 0.0 and df.iloc[index-1]['EMA_diff'] < 0.0:
        df.at[index,'Crossover'] = 1
        crossing_up.append(index)
        crossing.append(index)
    elif df.iloc[index]['EMA_diff'] < 0.0 and df.iloc[index-1]['EMA_diff'] > 0.0:
        df.at[index,'Crossover'] = 0
        crossing_down.append(index)
        crossing.append(index)
    else:
        df.at[index,'Crossover'] = np.nan
    index += 1 
    
#Labelling.
df['Label'] = np.nan
index = 1
for counter,value in enumerate(crossing):
    if counter == 0 or counter == len(crossing)-1:
        pass
    else:
        if value in crossing_up:
            down_rows = df.loc[crossing[counter-1]:value]
            end_down_index = down_rows.loc[down_rows['Close'] == down_rows['Close'].min()].index[0]
            up_rows = df.loc[value:crossing[counter+1]]
            end_up_index = up_rows.loc[up_rows['Close'] == up_rows['Close'].max()].index[0]
            
            df.at[end_down_index:end_up_index+1,'Label'] = 1
        
        elif value in crossing_down:
            up_rows = df.loc[crossing[counter-1]:value]
            end_up_index = up_rows.loc[up_rows['Close'] == up_rows['Close'].max()].index[0]
            down_rows = df.loc[value:crossing[counter+1]]
            end_down_index = down_rows.loc[down_rows['Close'] == down_rows['Close'].min()].index[0]
            
            df.at[end_up_index:end_down_index+1,'Label'] = 0

#Save dataframe.
save_location = "C:\\Users\\User\\Projects\\Forex\\PythonForFinance\\ForexPricePrediction\\GBPNZD60_processed.csv"
df.to_csv(save_location,header=True,index=False)