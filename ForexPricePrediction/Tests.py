# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 17:17:24 2019

@author: User
"""

import pandas as pd

location = "C:\\Users\\User\\Projects\\Forex\\PythonForFinance\\ForexPricePrediction\\GBPNZD60_processed.csv"
df = pd.read_csv(location,parse_dates=[['Date','Time']],index_col='Date_Time')
