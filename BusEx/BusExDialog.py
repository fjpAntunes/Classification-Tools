"""
/***************************************************************************
Name			 	 : BusEX
Description          : Trabalho SIG
Date                 : 17/Jul/17 
copyright            : (C) 2017 by Ronaldo Martins
email                : ronaldo.rmsjr@gmail.com 
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
"""
from PyQt4 import QtCore, QtGui 
from Ui_BusEx import Ui_BusEx
# create the dialog for BusEx
class BusExDialog(QtGui.QDialog):
  def __init__(self,Busex):
    QtGui.QDialog.__init__(self) 
    # Set up the user interface from Designer. 
    self.ui = Ui_BusEx (Busex)
    self.ui.setupUi(self)