# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file Ui_BusEx.ui
# Created with: PyQt4 UI code generator 4.4.4
# WARNING! All changes made in this file will be lost!
from PyQt4 import QtCore, QtGui
from PyQt4.QtCore import *
from PyQt4.QtGui import *
import os
import sys
import datetime

class Ui_BusEx(object):

    def __init__(self,BusEx):
        self.BusEx = BusEx
        self.available_clf_layers = {}
        self.available_validation_layers = {}
        self.selected_clf_layers = {}
        self.selected_training_layers = {}
        self.selected_validation_layers = {}
        self.classifierType = None
        self.classifier = None
        self.segmenter = None

    def setupUi(self, BusExDialog):
        self.BusExDialog = BusExDialog
        BusExDialog.setObjectName("BusEx")
        BusExDialog.resize(400, 600)

        # self.MainPage = QtGui.QWidget()

        self.vbox = QVBoxLayout()

        self.selectImageUi()
        # self.selectClassifierTypeUI()
        self.selectClassifierUi()
        self.selectClassificationLayersUi()
        self.selectTrainingLayers()
        # self.selectValidationLayers()
        self.selectSegmenterUi()
        self.b2 = QPushButton("Classificar")
        self.b2.clicked.connect(self.classify2)
        self.vbox.addStretch()
        self.vbox.addWidget(self.b2)

        # self.MainPage.setLayour(self.vbox)
        BusExDialog.setLayout(self.vbox)

        self.retranslateUi(BusExDialog)

    def retranslateUi(self, BusExDialog):
        BusExDialog.setWindowTitle(QtGui.QApplication.translate("BusEx", "BusEx", None, QtGui.QApplication.UnicodeUTF8))

    def selectClassifierTypeUI(self):
        ltc = QLabel("Tipo de classificador :")
        myFont = QFont()
        myFont.setBold(True)
        ltc.setFont(myFont)
        self.r_tc1 = QRadioButton("Supervisionado")
        self.r_tc2 = QRadioButton("Nao supervisionado")
        self.tc_group = QtGui.QButtonGroup()
        self.tc_group.addButton(self.r_tc1)
        self.tc_group.addButton(self.r_tc2)
        self.tc_group.name = "tc"
        self.tc_group.buttonClicked.connect(lambda: self.buttonGroupMethod(self.tc_group.name, self.tc_group.checkedButton()))
        self.vbox.addWidget(ltc)
        self.vbox.addWidget(self.r_tc1)
        self.vbox.addWidget(self.r_tc2)
        pass

    def selectClassifierUi(self):
        lc = QLabel("Classificador :")
        myFont = QFont()
        myFont.setBold(True)
        lc.setFont(myFont)
        self.vbox.addWidget(lc)

        teste = True
        self.c_group = QtGui.QButtonGroup()
        self.c_group.name = "c"
        names = [
            'Neural Net',
            'Linear SVM',
            'RBF SVM',
            'Gaussian Process',
            'Decision Tree',
            'Random Forest',
            'Nearest Neighbors',
            'AdaBoost',
            'Naive Bayes',
            'Quadratic Discrimant']
        r = [None] * len(names)
        for idx, name in enumerate(names):
            r[idx] = QRadioButton(name)
            self.c_group.addButton(r[idx])
            self.vbox.addWidget(r[idx])
        self.c_group.buttonClicked.connect(
            lambda: self.buttonGroupMethod(self.c_group.name, self.c_group.checkedButton()))
        #
        # if teste:
        #     r = [None] * len(names)
        #     for idx,name in enumerate(names):
        #         r[idx] = QRadioButton(name)
        #         self.c_group.addButton(r[idx])
        #         self.vbox.addWidget(r[idx])
        #     self.c_group.buttonClicked.connect(
        #         lambda: self.buttonGroupMethod(self.c_group.name, self.c_group.checkedButton()))
        # else:
        #     self.r_c1 = QRadioButton("Random Florest")
        #     self.r_c2 = QRadioButton("Outro")
        #     self.r_c2 = QRadioButton("Outro")
        #     self.r_c2 = QRadioButton("Outro")
        #     self.r_c2 = QRadioButton("Outro")
        #     self.r_c2 = QRadioButton("Outro")
        #     self.r_c2 = QRadioButton("Outro")
        #     self.r_c2 = QRadioButton("Outro")
        #     self.r_c2 = QRadioButton("Outro")
        #     self.r_c2 = QRadioButton("Outro")
        #     self.c_group = QtGui.QButtonGroup()
        #     self.c_group.addButton(self.r_c1)
        #     self.c_group.addButton(self.r_c2)
        #     self.c_group.name = "c"
        #     self.c_group.buttonClicked.connect(lambda: self.buttonGroupMethod(self.c_group.name, self.c_group.checkedButton()))
        #     self.vbox.addWidget(self.r_c1)
        #     self.vbox.addWidget(self.r_c2)
        pass

    def selectTrainingLayers(self):
        pass

    def selectValidationLayers(self):
        l2 = QLabel("Camadas disponiveis:")
        self.vbox.addWidget(l2)

        availableLayerBox = QHBoxLayout()
        self.b_refreshLayer = QPushButton()
        self.b_refreshLayer.setFixedWidth(25)
        self.b_addLayer = QPushButton("Adicionar")
        self.b_addLayer.setFixedWidth(70)
        self.b_addLayer.clicked.connect(self.addClfLayer)
        self.cb_available_clf_layers = QComboBox()
        self.loadAvailableLayers()
        # self.cb_available_clf_layers.addItem("C")
        # self.cb_available_clf_layers.addItems(["Java", "C#", "Python"])
        # self.cb_available_clf_layers.currentIndexChanged.connect(self.selectionchange)
        # self.b_refreshLayer.setIcon(QIcon(QPixmap("ajax-refresh-icon.gif")))
        self.b_refreshLayer.setIcon(QIcon(QPixmap(self.getIconPath())))
        # self.b_refreshLayer.clicked.connect(lambda: self.BusEx.loadLayers(self.cb_available_clf_layers))
        self.b_refreshLayer.clicked.connect(self.loadAvailableLayers)
        availableLayerBox.addWidget(self.b_refreshLayer)
        availableLayerBox.addWidget(self.cb_available_clf_layers)
        # availableLayerBox.addWidget(self.b_addLayer)
        self.vbox.addLayout(availableLayerBox)

        l3 = QLabel("Camadas selecionadas:")
        self.vbox.addWidget(l3)

        selectedLayerBox = QHBoxLayout()
        self.b_removeLayer = QPushButton("Remover")
        self.b_removeLayer.setFixedWidth(70)
        self.b_removeLayer.clicked.connect(self.removeClfLayer)
        self.cb_selected_clf_layers = QComboBox()
        self.cb_selected_clf_layers.addItem("Nenhuma camada selecionada.")
        # self.cb_selected_clf_layers.currentIndexChanged.connect(self.selectionchange)
        # self.b_refreshLayer.setIcon(QIcon(QPixmap("ajax-refresh-icon.gif")))
        selectedLayerBox.addWidget(self.cb_selected_clf_layers)
        selectedLayerBox.addWidget(self.b_removeLayer)
        self.vbox.addLayout(selectedLayerBox)
        pass

    def selectSegmenterUi(self):
        ls = QLabel("Segmentador:")
        myFont = QFont()
        myFont.setBold(True)
        ls.setFont(myFont)
        self.r_s1 = QRadioButton("Quick")
        self.r_s2 = QRadioButton("Felz")
        # button group
        self.ts_group = QtGui.QButtonGroup()
        self.ts_group.addButton(self.r_s1)
        self.ts_group.addButton(self.r_s2)
        self.ts_group.name = "ts"
        self.ts_group.buttonClicked.connect(lambda: self.buttonGroupMethod(self.ts_group.name, self.ts_group.checkedButton()))
        self.vbox.addWidget(ls)
        self.vbox.addWidget(self.r_s1)
        self.vbox.addWidget(self.r_s2)
        pass

    def selectImageUi(self):
        l1 = QLabel("Imagem para classificacao:")
        myFont = QFont()
        myFont.setBold(True)
        l1.setFont(myFont)
        selectFile_box = QHBoxLayout()
        self.line_filePath = QLineEdit()
        b_search = QPushButton("Procurar")
        b_search.setFixedWidth(70)
        b_search.clicked.connect(self.selectImage)
        b_loadImage = QPushButton("Carregar imagem")
        b_loadImage.clicked.connect(self.loadImage)
        # b_loadImage.clicked.connect(lambda: self.BusEx.loadImage(self.imagePath))
        selectFile_box.addWidget(self.line_filePath)
        selectFile_box.addWidget(b_search)
        self.vbox.addWidget(l1)
        self.vbox.addLayout(selectFile_box)
        self.vbox.addWidget(b_loadImage)

    def selectClassificationLayersUi(self):
        #
        l1 = QLabel("Selecione as camadas para classificacao:")
        myFont = QFont()
        myFont.setBold(True)
        l1.setFont(myFont)
        self.vbox.addWidget(l1)

        l2 = QLabel("Camadas disponiveis:")
        self.vbox.addWidget(l2)

        availableLayerBox = QHBoxLayout()
        self.b_refreshLayer = QPushButton()
        self.b_refreshLayer.setFixedWidth(25)
        self.b_addLayer = QPushButton("Adicionar")
        self.b_addLayer.setFixedWidth(70)
        self.b_addLayer.clicked.connect(self.addClfLayer)
        self.cb_available_clf_layers = QComboBox()
        self.loadAvailableLayers()
        self.b_refreshLayer.setIcon(QIcon(QPixmap(self.getIconPath())))
        self.b_refreshLayer.clicked.connect(self.loadAvailableLayers)
        availableLayerBox.addWidget(self.b_refreshLayer)
        availableLayerBox.addWidget(self.cb_available_clf_layers)
        availableLayerBox.addWidget(self.b_addLayer)
        self.vbox.addLayout(availableLayerBox)

        l3 = QLabel("Camadas selecionadas:")
        self.vbox.addWidget(l3)
        selectedLayerBox = QHBoxLayout()
        self.b_removeLayer = QPushButton("Remover")
        self.b_removeLayer.setFixedWidth(70)
        self.b_removeLayer.clicked.connect(self.removeClfLayer)
        self.cb_selected_clf_layers = QComboBox()
        self.cb_selected_clf_layers.addItem("Nenhuma camada selecionada.")
        selectedLayerBox.addWidget(self.cb_selected_clf_layers)
        selectedLayerBox.addWidget(self.b_removeLayer)
        self.vbox.addLayout(selectedLayerBox)
        pass

    def loadImage(self):
        imagePath = self.line_filePath.text()
        self.BusEx.loadImage(imagePath)

    def loadAvailableLayersForValidation(self,layerName_cb_loop):
        # print((self.cb_available_validation_layers[layerName_cb_loop]))
        (self.cb_available_validation_layers[layerName_cb_loop]).clear()
        self.available_validation_layers.clear()
        availableLayers = self.BusEx.loadAvailableLayers()
        layer_list = []
        if len(availableLayers) != 0:
            for layer in availableLayers:
                QgsRasterLayer = layer  # QgsRasterLayer object
                LayerPath = layer.source()
                LayerName = layer.name()
                layer_list.append(LayerName)
                self.available_validation_layers[LayerName] = LayerPath

            print(layer_list)
            print((self.cb_available_validation_layers[layerName_cb_loop]))
            (self.cb_available_validation_layers[layerName_cb_loop]).addItems(layer_list)
        else:
            (self.cb_available_validation_layers[layerName_cb_loop]).addItems(["Nenhuma camada carregada."])
        pass

    def loadAvailableLayers(self):
        self.cb_available_clf_layers.clear()
        self.available_clf_layers.clear()
        availableLayers = self.BusEx.loadAvailableLayers()
        layer_list = []
        if len(availableLayers) != 0:
            for layer in availableLayers:
                QgsRasterLayer = layer # QgsRasterLayer object
                LayerPath = layer.source()
                LayerName = layer.name()
                layer_list.append(LayerName)
                self.available_clf_layers[LayerName] = LayerPath
            self.cb_available_clf_layers.addItems(layer_list)
        else:
            self.cb_available_clf_layers.addItems(["Nenhuma camada carregada."])

    def addClfLayer(self):
        layer_to_add = self.cb_available_clf_layers.currentText()
        # Checa se o layer já foi adicionado
        if layer_to_add in self.selected_clf_layers:
            pass
        else:
            self.selected_clf_layers[layer_to_add] = self.available_clf_layers[layer_to_add]
            # Checa se já tem algum layer selecionado
            if self.cb_selected_clf_layers.currentText() == "Nenhuma camada selecionada.":
                self.cb_selected_clf_layers.clear()
                self.cb_selected_clf_layers.addItem(layer_to_add)
            else:
                self.cb_selected_clf_layers.addItem(layer_to_add)
        # self.selected_clf_layers.addItem()
        pass

    def removeClfLayer(self):
        layer_to_remove = self.cb_selected_clf_layers.currentText()
        # Checa se o layer para remover está entre os selecionados
        if layer_to_remove in self.selected_clf_layers:
            del self.selected_clf_layers[layer_to_remove]
            self.cb_selected_clf_layers.removeItem(self.cb_selected_clf_layers.currentIndex())
            # Caso eu tenha deletado todos os layers selecionados
            if len(self.selected_clf_layers) == 0:
                self.cb_selected_clf_layers.addItem("Nenhuma camada selecionada.")
        else:
            pass
        # self.selected_clf_layers.addItem()
        pass

    def addValidationLayer(self):
        layer_to_add = self.cb_available_validation_layers.currentText()
        if layer_to_add in self.selected_validation_layers:
            pass
        else:
            self.selected_validation_layers[layer_to_add] = "layer_path"
        # self.selected_clf_layers.addItem()
        pass

    def removeValidationLayer(self):
        layer_to_remove = self.cb_selected_validation_layers.currentText()
        if layer_to_remove in self.selected_validation_layers:
            del self.selected_validation_layers[layer_to_remove]
            self.cb_selected_validation_layers.removeItem(self.cb_selected_validation_layers.currentIndex())
        else:
            pass
        # self.selected_clf_layers.addItem()
        pass

    def addTrainingLayer(self):
        layer_to_add = self.cb_available_training_layers.currentText()
        if layer_to_add in self.selected_training_layers:
            pass
        else:
            self.selected_training_layers[layer_to_add] = "layer_path"
        # self.selected_clf_layers.addItem()
        pass

    def removeTrainingLayer(self):
        layer_to_remove = self.cb_selected_training_layers.currentText()
        if layer_to_remove in self.selected_training_layers:
            del self.selected_training_layers[layer_to_remove]
            self.cb_selected_training_layers.removeItem(self.cb_selected_training_layers.currentIndex())
        else:
            pass
        # self.selected_clf_layers.addItem()
        pass

    def setButtonBox(self):
        self.buttonBox.setGeometry(QtCore.QRect(30, 240, 341, 32))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Cancel | QtGui.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")

        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL("accepted()"), BusEx.accept)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL("rejected()"), BusEx.reject)
        QtCore.QMetaObject.connectSlotsByName(BusEx)

    def selectImage(self):
        # C:/Users/Ronaldo/Documents/Carto- 2017.2/PFC/Imagens/
        fname = QFileDialog.getOpenFileName(self.BusExDialog, 'Open file', 'C:\\Users\\Ronaldo\\Documents\\Carto- 2017.2\\PFC\\Imagens\\', "Image files (*.jpg *.jpeg *.tif *.tiff)")
        # fname = QFileDialog.getOpenFileName(self.BusEx, 'Open file', 'c:\\', "Image files (*.jpg *.jpeg)")
        self.imagePath = fname
        self.line_filePath.setText(fname)
        print(fname)

    def getIconPath(self):
        currentFile = __file__
        realPath = os.path.realpath(currentFile)  # /home/user/test/my_script.py
        dirPath = os.path.dirname(realPath)  # /home/user/test
        path = dirPath + "\\ajax-refresh-icon.gif"
        return path

    def checkSelection(self):
        ready = True
        msg = "Por favor, selecione os seguintes campos: "
        imagePath = self.line_filePath.text()
        if not imagePath:
            msg = msg + "\nImagem para classificacao"
            ready = False
        if not self.segmenter:
            msg = msg + "\nSegmentador"
            ready = False
            pass
        # if not self.classifierType:
        #     msg = msg + "\nTipo de classificador"
        #     ready = False
        #     if (self.classifierType == "Supervisionado"  and len(self.selected_clf_layers) == 0):
        #         msg = msg + "\nCamadas de treinamento"
        #         ready = False
        #         pass
        if not self.classifier:
            msg = msg + "\nClassificador"
            ready = False
            pass
        return ready, msg
        pass

    def classify2(self):
        # Checo se todos os campos foram selecionados
        ready, msg = self.checkSelection()

        # Checo se será necessária a validação

        # Todos os campos foram preenchidos
        self.classifierType = "Supervisionado"
        if ready:
            self.checkToShowValidationDialog()
        else:
            pass
            self.showdialog(msg)
            # self.checkToShowValidationDialog()
        pass

    def classify(self):
        ready, msg = self.checkSelection()
        # Todos os campos foram preenchidos
        self.classifierType = "Supervisionado"
        if ready:
            # Start Classification
            # Classificador supervisionado -> select clf layers (training)
            image_path = self.line_filePath.text()
            imageName = image_path.split('/')[-1]
            imageDir = image_path[:-len(imageName)]
            classified_image_path = imageDir + "Imagem Classificada_" +  datetime.datetime.today().strftime('%Ih%M%p-%d_%m_%Y') + ".tiff"
            pass
            if self.classifierType == "Supervisionado":
                print("Esta no supervisionado")
                self.BusEx.classify(image_path, classified_image_path, True, self.classifierType,self.classifier, self.segmenter, self.selected_clf_layers)

                # Classificador supervisionado -> sem clf layers (training)
                self.BusEx.classify(image_path, classified_image_path, True, self.classifierType, self.classifier,
                                self.segmenter)
            else:
                print("Esta no nao supervisionado")
                self.BusEx.classify(image_path, classified_image_path, True, self.classifierType,self.classifier, self.segmenter)
            pass
        # Faltam campos para serem preenchidos
        else:
            # self.showdialog(msg)
            # self.checkToShowValidationDialog()
            self.checkToShowValidationDialog()

    def selectionchange(self, i):
        # print "Items in the list are :"
        # for count in range(self.cb.count()):
        #     print self.cb.itemText(count)
        # currentFile = __file__  # May be 'my_script', or './my_script' or
        # # '/home/user/test/my_script.py' depending on exactly how
        # # the script was run/loaded.
        # realPath = os.path.realpath(currentFile)  # /home/user/test/my_script.py
        # dirPath = os.path.dirname(realPath)  # /home/user/test
        # dirName = os.path.basename(dirPath)  # test
        # print(realPath)
        # print(dirPath)
        # print(dirName)
        # print(os.getcwd())
        print("Current index", i, "selection changed ", self.cb_available_clf_layers.currentText())

    def buttonGroupMethod(self, radioGroup, selected):
        if radioGroup =="tc":
            self.classifierType = selected.text()
            print(selected.text())
            pass
        elif radioGroup =="c":
            self.classifier = selected.text()
            print(selected.text())
            pass
        elif radioGroup =="ts":
            self.segmenter = selected.text()
            print(selected.text())
            pass
        # print(radioGroup)
        pass

    def comboManager(self, group, option, selected):

        pass

    def btnstate(self, b):
        if b.text() == "Quick":
            if b.isChecked() == True:
                print(b.text() + " is selected")
            else:
                print(b.text() + " is deselected")
        if b.text() == "Felz":
            if b.isChecked() == True:
                print(b.text() + " is selected")
            else:
                print(b.text() + " is deselected")

    def validationSelectionDialog(self):
        self.cb_available_validation_layers = {}
        self.d_validationSelection = QtGui.QDialog()
        self.d_validationSelection.setWindowTitle("Validacao")

        layout = QVBoxLayout(self.d_validationSelection)

        l1 = QtGui.QLabel("Selecione as camadas de validacao para cada classe:")
        myFont = QFont()
        myFont.setBold(True)
        l1.setFont(myFont)
        layout.addWidget(l1)
        layout.addSpacing(10)

        for layer in self.selected_clf_layers.keys():
            pergunta = "Selecione a camada de validacao para o layer: " + layer
            l = QtGui.QLabel(pergunta)
            self.cb_available_validation_layers[layer] = QComboBox()
            self.loadAvailableLayersForValidation(layer)
            layout.addWidget(l)
            layout.addWidget(self.cb_available_validation_layers[layer])
            layout.addSpacing(10)

            # print(layer)
        # print(self.selected_clf_layers.values())
        P = QtGui.QPushButton("Camadas de validacao selecionadas")
        P.clicked.connect(self.getTestDataPathAndClassify)
        layout.addWidget(P)

        self.d_validationSelection.exec_()

        pass

    def getTestDataPath(self):
        # Here we get the name of the selected validation layers path and add to a variable
        for combobox in self.cb_available_validation_layers.values():
            layerSelected = combobox.currentText()
            print(layerSelected)
            self.selected_validation_layers[layerSelected] = self.available_clf_layers[layerSelected]
        pass

    def getTestDataPathAndClassify(self):

        # Adicionamos os caminhos e nome dos layers de validação ao dict selected_validation_layers
        self.getTestDataPath()

        # Now we start classification with validation
        image_path = self.line_filePath.text()
        imageName = image_path.split('/')[-1]
        imageDir = image_path[:-len(imageName)]
        classified_image_path = imageDir + "Imagem Classificada_" + datetime.datetime.today().strftime(
            '%Hh%M-%d_%m_%Y') + ".tiff"

        # Classificador supervisionado
        if self.classifierType == "Supervisionado":
            self.cm, self.classificationAccuracy, self.classificationReport = self.BusEx.classify(image_path, classified_image_path, True, self.classifierType, self.classifier,
                                self.segmenter, self.selected_clf_layers,self.selected_validation_layers)
            self.validationResultsDialog(self.cm,self.classificationAccuracy,self.classificationReport)

        else:
            pass
            # print("Esta no nao supervisionado")
            # self.BusEx.classify(image_path, classified_image_path, True, self.classifierType, self.classifier,
            #                     self.segmenter)
        pass


        # Faço a requisição para obter o retorno do Obia3 da validação


        pass

    def validationResultsDialog(self,cm,classificationAccuracy,classificationReport):
        self.d_validationResults = QtGui.QDialog()
        self.d_validationResults.setWindowTitle("Resultados da Validacao")

        layout = QVBoxLayout(self.d_validationResults)

        l1 = QtGui.QLabel("Matriz de confusao")
        myFont = QFont()
        myFont.setBold(True)
        l1.setFont(myFont)
        layout.addWidget(l1)

        # labels = ["Mar","Vegetacao","Urbana"]
        labels = self.selected_clf_layers.keys()
        # cm = [[747, 33, 15], [20, 757, 0], [1114, 327, 2489]]
        layout.addWidget(self.addTable(labels,cm))

        l2 = QtGui.QLabel("Acuracia de classificacao")
        # classificationAccuracy = 0.950745
        l2.setFont(myFont)
        layout.addWidget(l2)
        l2_r = QtGui.QLabel(str(classificationAccuracy))
        layout.addWidget(l2_r)

        l3 = QtGui.QLabel("Report de classificacao")
        l3.setFont(myFont)
        layout.addWidget(l3)
        # classificationReport = '             precision    recall  f1-score   support\n\n          A       0.93      0.97      0.95      3930\n          B       0.81      0.62      0.70       795\n          C       1.00      1.00      1.00       777\n\navg / total       0.92      0.92      0.92      5502\n'
        l3_r = QtGui.QLabel(classificationReport)
        layout.addWidget(l3_r)

        # l4 = QtGui.QLabel("Exportar validacao: ")
        # l4.setFont(myFont)
        # exportFile_box = QHBoxLayout()
        # self.export_filepath = QLineEdit()
        # expB_search = QPushButton("Procurar")
        # expB_search.setFixedWidth(70)
        # expB_search.clicked.connect(self.selectDirectory)
        # exportFile_box.addWidget(self.export_filepath)
        # exportFile_box.addWidget(expB_search)
        # pb_export = QtGui.QPushButton("Exportar validacao")
        # layout.addWidget(l4)
        # layout.addWidget(QtGui.QLabel("Selecione o diretorio: "))
        # layout.addLayout(exportFile_box)
        # layout.addWidget(pb_export)

        self.d_validationResults.exec_()

        pass

    def selectDirectory(self):
        # C:/Users/Ronaldo/Documents/Carto- 2017.2/PFC/Imagens/
        self.exp_directory = QFileDialog.getExistingDirectory(self.d_validationResults, 'Selecione a pasta',
                                            'C:\\')
        # fname = QFileDialog.getOpenFileName(self.BusEx, 'Open file', 'c:\\', "Image files (*.jpg *.jpeg)")
        self.export_filepath.setText(self.exp_directory)
        print(self.exp_directory)

    def saveFile(self):
        name = self.exp_directory + "\\Validacao:" + datetime.datetime.today().strftime('%Ih%M%p-%d_%m_%Y') + ".txt"
        # I - hora
        # M - minuto
        # p - AM ou PM
        pass

    def addTable(self, labels,cm):
        self.table = QTableWidget()
        # tableItem = QTableWidgetItem()

        # initiate table
        size = len(labels)
        self.table.setRowCount(size)
        self.table.setColumnCount(size)

        # set label
        self.table.setHorizontalHeaderLabels(labels)
        self.table.setVerticalHeaderLabels(labels)

        # set data
        for i, label1 in enumerate(labels):
            # print("%{0}s".format(columnwidth) % label1, end="\t")
            for j in range(len(labels)):
                print(i)
                print(j)
                self.table.setItem(int(i), int(j), QTableWidgetItem(str(cm[int(i)][int(j)])))
                self.table.item(int(i),int(j)).setTextAlignment (QtCore.Qt.AlignVCenter|QtCore.Qt.AlignHCenter)
        self.table.resizeColumnsToContents()

        header_horizontal = self.table.horizontalHeader()
        header_horizontal.setStretchLastSection(True)
        header_vertical = self.table.verticalHeader()
        header_vertical.setStretchLastSection(True)

        return self.table

    def checkToShowValidationDialog(self):
        self.showValidationDialog = QtGui.QDialog()
        self.showValidationDialog.setWindowTitle("Validacao")
        layout = QVBoxLayout(self.showValidationDialog)
        # nice widget for editing the date

        l1 = QtGui.QLabel("Deseja efetuar a validacao apos a classificacao")
        # OK and Cancel buttons
        buttons = QtGui.QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            Qt.Horizontal)
        # print(buttons.button(QDialogButtonBox.Ok))
        (buttons.button(QDialogButtonBox.Ok)).setText("Sim")
        (buttons.button(QDialogButtonBox.Cancel)).setText("Nao")
        buttons.accepted.connect(self.testAccept)
        buttons.rejected.connect(self.testReject)
        layout.addWidget(l1)
        layout.addWidget(buttons)
        self.showValidationDialog.exec_()
        pass

    def testAccept(self):
        self.doValidate = True

        # Vamos para a seleção das áreas de teste e classificação
        self.validationSelectionDialog()



    def testReject(self):
        self.doValidate = False
        self.showValidationDialog.close()

        # Now we start classification without validation
        image_path = self.line_filePath.text()
        imageName = image_path.split('/')[-1]
        imageDir = image_path[:-len(imageName)]
        classified_image_path = imageDir + "Imagem Classificada_" + datetime.datetime.today().strftime(
            '%Ih%M%p-%d_%m_%Y') + ".tiff"
        pass
        if self.classifierType == "Supervisionado":
            self.BusEx.classify(image_path, classified_image_path, True, self.classifierType, self.classifier,
                                self.segmenter, self.selected_clf_layers)

        else:
            print("Esta no nao supervisionado")
            self.BusEx.classify(image_path, classified_image_path, True, self.classifierType, self.classifier,
                                self.segmenter)
        pass

    def startClassification(self):
        if self.doValidate == True:
            pass
        else:
            pass
        pass

    def showdialog(self, text_msg,title=None):
        title = "Atencao"
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText(text_msg)
        # msg.setInformativeText("This is additional information")
        msg.setWindowTitle(title)
        # msg.setDetailedText("The details are as follows:")
        # msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.buttonClicked.connect(self.msgbtn)
        retval = msg.exec_()

    def msgbtn(self,i):
        print("Button pressed is:", i.text())

class Page1Widget(QtGui.QWidget):
    def __init__(self, parent=None):
        super(Page1Widget, self).__init__(parent)
        layout = QtGui.QHBoxLayout()
        self.label = QtGui.QLabel('This is Page 1')
        self.button = QtGui.QPushButton('Go Page 2!')
        parent.connectPageChange(self.button, "Page2Widget")
        layout.addWidget(self.label)
        layout.addWidget(self.button)
        self.setLayout(layout)
        # you might want to do self.button.click.connect(self.parent().login) here

class Page2Widget(QtGui.QWidget):
    def __init__(self, parent=None):
        super(Page2Widget, self).__init__(parent)
        layout = QtGui.QHBoxLayout()
        self.label = QtGui.QLabel('This is Page 2!')
        self.button = QtGui.QPushButton('Go Page 3')
        parent.connectPageChange(self.button, "Page3Widget")
        layout.addWidget(self.label)
        layout.addWidget(self.button)
        self.setLayout(layout)