# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 13:50:39 2018

@author: User
"""

import tkinter as tk

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg,NavigationToolbar2Tk
#Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler

import numpy as np

#Tkinter root window.
root = tk.Tk()
root.title("Embedding in Tk")

#Create figure and add subplot.
fig = plt.figure(1)
fig.suptitle("Forex price movement.")

axes = fig.add_subplot(1,1,1)
y = np.random.randint(1,50,size=50); x = np.arange(len(y)) #Data.
axes.plot(x,y,'green')

#Convert matplotlib figure into tk widget???
canvax = FigureCanvasTkAgg(fig,master=root) #A tk.DrawingArea
canvax.draw()
canvax.get_tk_widget().pack(side=tk.TOP,fill=tk.BOTH,expand=1)

toolbar = NavigationToolbar2Tk(canvax,root)
toolbar.update()
canvax.get_tk_widget().pack(side=tk.TOP,fill=tk.BOTH,expand=1)
'''
def on_key_press(event):
    print("---You pressed {}".format(event.key))
    key_press_handler(event,canvax,toolbar)
'''
def onpick(event):
    thisline = event.artist
    xdata = thisline.get_xdata()
    ydata = thisline.get_ydata()
    ind = event.ind
    points = tuple(zip(xdata[ind], ydata[ind]))
    print('onpick points:', points)
    
canvax.mpl_connect("***key_press_event***",onpick)

def _quit():
    root.quit() #Stops mainloop.
    root.destroy() #Necessary in Windows!
    
button = tk.Button(root,text="Quit program",command=_quit)
button.pack(side=tk.BOTTOM)

tk.mainloop()
