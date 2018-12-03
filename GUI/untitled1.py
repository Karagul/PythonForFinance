# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 12:19:06 2018

@author: User
"""

import tkinter as tk

#Root window.
root = tk.Tk()

#Widgets.
frame = tk.Frame(root)
tk.Label(frame,text="Demo: side & fill---").pack()
tk.Button(frame,text="A").pack(side=tk.LEFT,fill=tk.Y)
tk.Button(frame,text="B").pack(side=tk.TOP,fill=tk.X)
tk.Button(frame,text="C").pack(side=tk.RIGHT,fill=tk.Y)
tk.Button(frame,text="D").pack(side=tk.TOP,fill=tk.BOTH)
tk.Button(frame,text="e").pack(side=tk.BOTTOM,fill=tk.X)

tk.Label(root,text="Demo: expand---").pack()
tk.Button(root,text="I do not expand.").pack()
tk.Button(root,text="I expand but do not fill X.").pack(expand=1)
tk.Button(root,text="I EXPAND AND FILL X").pack(fill=tk.X,expand=1)

#Geometry.
frame.pack()


#Display.
root.mainloop()