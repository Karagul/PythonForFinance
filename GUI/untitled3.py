# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 13:58:12 2018

@author: User
"""

import tkinter as tk

#Root window and it's title.
root = tk.Tk()
root.title("Find & Replace")

#Widgets.
tk.Label(root,text="Find:").grid(row=0,sticky=tk.E)
tk.Label(root,text="Replace:").grid(row=1,sticky=tk.W)

tk.Entry(root).grid(row=0,column=1,columnspan=9,sticky='we',padx=2,pady=2)
tk.Entry(root).grid(row=1,column=1,columnspan=9,sticky='we',padx=2,pady=2)

tk.Button(root,text="Find").grid(row=0,column=10,sticky=tk.W)
tk.Button(root,text="Find All").grid(row=1,column=10,sticky=tk.W)
tk.Button(root,text="Replace").grid(row=2,column=10,sticky=tk.W)
tk.Button(root,text="Replace All").grid(row=3,column=10,sticky=tk.W)

tk.Checkbutton(root,text="Match whole word only").grid(row=2,column=1,columnspan=4,sticky=tk.W)
tk.Checkbutton(root,text="Match Case").grid(row=3,column=1,columnspan=4,sticky=tk.W)
tk.Checkbutton(root,text="Wrap Around").grid(row=4,column=1,columnspan=4,sticky=tk.W)

tk.Label(root,text="Direction:").grid(row=2,column=6,sticky=tk.W)
tk.Radiobutton(root,text="Up",value=1).grid(row=3,column=6,sticky=tk.W)
tk.Radiobutton(root,text="Down",value=2).grid(row=3,column=7,sticky=tk.E)

#Display.
root.mainloop()