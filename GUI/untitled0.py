# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 17:24:15 2018

@author: User
"""

import pandas as pd


#File location.
location = "C:\\Users\\User\\Projects\\MT4\\Data\\EURUSDdaily26_10_2018.csv"

#Lambda function to take care of Date-Time format.
dateparse = lambda x : pd.datetime.strptime(x,'%Y-%m-%d %H:%M:%S')

#Create dataframe.
df = pd.read_csv(location,names=['Date','Time','Open','High','Low','Close','Volume'],parse_dates=[['Date','Time']],date_parser=dateparse)
