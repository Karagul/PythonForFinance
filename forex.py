# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 16:28:25 2018

@author: User
"""

pp = lambda high,low,close:(high+low+close)/3
r1 = lambda pivot,low:(2*pivot)-low
s1 = lambda pivot,high:(2*pivot)-high
r2 = lambda pivot,high,low:pivot+(high-low)
s2 = lambda pivot,high,low:pivot-(high-low)
r3 = lambda pivot,high,low:high+2(pivot-low)
s3 = lambda pivot,high,low:low-2(high-pivot)
