# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 10:47:11 2018

@author: User
"""

import tkinter as tk

#Root.
root = tk.Tk()

#Widgets.
tk.Label(root,text="Username:").grid(row=0,sticky=tk.W)
tk.Label(root,text="Password:").grid(row=1,sticky=tk.W)

tk.Entry(root).grid(row=0,column=1,sticky=tk.E)
tk.Entry(root).grid(row=1,column=1,sticky=tk.E)

tk.Button(root,text="Enter").grid(row=2,column=1,sticky=tk.E)

#Display.
root.mainloop()