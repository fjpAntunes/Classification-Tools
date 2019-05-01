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
# Import the PyQt and QGIS libraries
from PyQt4.QtCore import * 
from PyQt4.QtGui import *
from qgis.core import *
# Initialize Qt resources from file resources.py
import resources
# Import the code for the dialog
from BusExDialog import BusExDialog
from obia.Obia3 import Obia3

class BusEx: 

  def __init__(self, iface):
    # Save reference to the QGIS interface
    self.iface = iface

  def initGui(self):  
    # Create action that will start plugin configuration
    self.action = QAction(QIcon(":/plugins/BusEx/icon.png"), \
        "BusEX", self.iface.mainWindow())
    # connect the action to the run method
    QObject.connect(self.action, SIGNAL("activated()"), self.run) 

    # Add toolbar button and menu item
    self.iface.addToolBarIcon(self.action)
    self.iface.addPluginToMenu("&BusEX", self.action)

  def unload(self):
    # Remove the plugin menu item and icon
    self.iface.removePluginMenu("&BusEX",self.action)
    self.iface.removeToolBarIcon(self.action)

  # run method that performs all the real work
  def run(self):
    # create and show the dialog 
    dlg = BusExDialog(self)
    self.OBIA = Obia3()
    # show the dialog
    dlg.show()
    result = dlg.exec_()
    # See if OK was pressed
    if result == 1:
      # do something useful (delete the line containing pass and
      # substitute with your code
      pass
      # Carrega a imagem de interesse

  def loadImage(self,path):
    fileName = path
    # fileName = "C:/Users/Ronaldo/Documents/Carto- 2017.2/PFC/Imagens/rapieye_95906/2328520_2015-07-12_RE1_3A_313875_CR.tif"
    fileInfo = QFileInfo(fileName)
    baseName = fileInfo.baseName()
    rlayer = QgsRasterLayer(fileName, baseName)

    if not rlayer.isValid():
      print("Layer failed to load!")
      QgsMessageLog.logMessage("Layer failed to load!", "BusEX")

    #  Set the layer's transparency to 50 percent:
    # rlayer.renderer().setOpacity(0.5)

    # Add the layer
    QgsMapLayerRegistry.instance().addMapLayer(rlayer)

    # Another way to add the layer
    # self.iface.addRasterLayer(fileName, baseName)
    pass

    # Seleciona as regioes de interesse para classificacao
  def loadLayers(self,comboBox):
    comboBox.clear()
    layers = self.iface.legendInterface().layers()
    layer_list = []
    if len(layers) != 0:
      for layer in layers:
        layer_list.append(layer.name())
      comboBox.addItems(layer_list)
    else:
      comboBox.addItems(["Nenhuma camada carregada."])

  def loadAvailableLayers(self):
    return self.iface.legendInterface().layers()

  def selectLayers(self):
    layers = self.iface.legendInterface().layers()
    layer_list = []
    for layer in layers:
      fileName = layer
      fileInfo = QFileInfo(fileName)
      baseName = fileInfo.baseName()
      print(fileName)
      print(fileInfo)
      print(baseName)
      layer_list.append(layer.name())
    pass

    # Remove regioes de interesse escolhidas para classificacao

  def removeLayers(self):
    pass

    # Seleciona o tipo de classificador que sera utilizado

  def selectClassifier(self):
    pass

  def loadClassifiedImage(self, path, transparency):
    fileName = path
    # fileName = "C:/Users/Ronaldo/Documents/Carto- 2017.2/PFC/Imagens/rapieye_95906/2328520_2015-07-12_RE1_3A_313875_CR.tif"
    fileInfo = QFileInfo(fileName)
    baseName = fileInfo.baseName()
    rlayer = QgsRasterLayer(fileName, baseName)

    if not rlayer.isValid():
      print("Layer failed to load!")
      QgsMessageLog.logMessage("Layer failed to load!", "BusEX")

    #  Set the layer's transparency to 50 percent:
    rlayer.renderer().setOpacity(transparency)

    # Add the layer
    QgsMapLayerRegistry.instance().addMapLayer(rlayer)

  def setTransparency(self, layer, transparency):
    rlayer.renderer().setOpacity(0.5)
    rlayer.triggerRepaint()

  # Script de classificacao
  def classify(self, image_path, classified_image_path, show_output, classifierType, classifier, segmenter, selected_clf_layers=None,selected_valitation_layers=None):

    # Supervisionado - Preciso enviar segmentos de treinamento
    if classifierType == "Supervisionado":
      self.OBIA.setParametersQGIS(image_path, classified_image_path, show_output, classifierType, classifier, segmenter,
                                  selected_clf_layers, selected_valitation_layers)
      self.OBIA.QgisClassification()

      # Se deseja que seja liberado um output
      if show_output:
        self.loadClassifiedImage(classified_image_path,0.3)

      # Se foram selecionada os arquivos para validacao
      if selected_valitation_layers is not None:
        # cm, classificationAccuracy, classificationReport = self.OBIA.validateClassification()
        cm, erros_vetorizacao, total_pixels= self.OBIA.validateClassification()
        # return cm, classificationAccuracy, classificationReport
        return cm, erros_vetorizacao, total_pixels
        # Desejo mostrar a imagem classificada

    # Nao Supervisionado - Preciso enviar segmentos de treinamento
    else:
      self.OBIA.setParametersQGIS(self, image_path, classified_image_path, show_output, classifierType, classifier,segmenter)
      self.OBIA.QgisClassification()
      # Desejo mostrar a imagem classificada
      if show_output:
        self.loadClassifiedImage(classified_image_path,0.3)
      pass

    pass

  def showResults(self):
    pass
