# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 18:07:30 2018

@author: User
"""

import tkinter as tk

#******Root window.******
root = tk.Tk()

#******Frame container.******
topframe = tk.Frame()
topframe.pack() #Display

bottomframe = tk.Frame()
bottomframe.pack(side=tk.BOTTOM)

#******Widgets.******
button1 = tk.Button(topframe,text="Button 1",fg="red")
button2 = tk.Button(topframe,text="Button 2",fg="red")
button3 = tk.Button(topframe,text="Button 3",fg="red")
button4 = tk.Button(bottomframe,text="Button 4",fg="purple")

#******Display widgets.******
button1.pack(side=tk.LEFT)
button2.pack(side=tk.LEFT)
button3.pack(side=tk.LEFT)
button4.pack()

#******Display everything.******
root.mainloop()