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
 This script initializes the plugin, making it known to QGIS.
"""
def name(): 
  return "BusEX" 
def description():
  return "Trabalho SIG"
def version(): 
  return "Version 0.1" 
def qgisMinimumVersion():
  return "1.0"
def classFactory(iface): 
  # load BusEx class from file BusEx
  from BusEx import BusEx 
  return BusEx(iface)
