# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 15:38:58 2018

@author: User
"""

import sys
from PyQt4 import QtGui

def window():
    #Create application object.
    app = QtGui.QApplication(sys.argv)
    #Create top level window.
    w = QtGui.QWidget()
    #Add label object to window.
    b = QtGui.QLabel(w)
    #Set caption of label.
    b.setText("Hello World!")
    #Size and position of window
    w.setGeometry(100,100,200,50)
    b.move(50,20)
    w.setWindowTitle("PyQt")
    w.show()
    sys.exit(app.exec_())
    
if __name__ == '__main__':
    window()