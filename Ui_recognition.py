# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Ui_recognition.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 537)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.imageLabel = QtWidgets.QLabel(self.centralwidget)
        self.imageLabel.setGeometry(QtCore.QRect(120, 110, 200, 200))
        self.imageLabel.setText("")
        self.imageLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.imageLabel.setObjectName("imageLabel")
        self.title = QtWidgets.QLabel(self.centralwidget)
        self.title.setGeometry(QtCore.QRect(250, 10, 291, 51))
        font = QtGui.QFont()
        font.setPointSize(40)
        self.title.setFont(font)
        self.title.setAlignment(QtCore.Qt.AlignCenter)
        self.title.setObjectName("title")
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(180, 340, 431, 81))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.openButton = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.openButton.setObjectName("openButton")
        self.horizontalLayout.addWidget(self.openButton)
        self.recButton = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.recButton.setObjectName("recButton")
        self.horizontalLayout.addWidget(self.recButton)
        self.exitButton = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.exitButton.setObjectName("exitButton")
        self.horizontalLayout.addWidget(self.exitButton)
        self.resultLabel = QtWidgets.QLabel(self.centralwidget)
        self.resultLabel.setGeometry(QtCore.QRect(490, 140, 81, 31))
        self.resultLabel.setObjectName("resultLabel")
        self.result = QtWidgets.QLineEdit(self.centralwidget)
        self.result.setGeometry(QtCore.QRect(490, 180, 191, 21))
        self.result.setReadOnly(True)
        self.result.setObjectName("result")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.title.setText(_translate("MainWindow", "测试图片识别"))
        self.openButton.setText(_translate("MainWindow", "选择图片"))
        self.recButton.setText(_translate("MainWindow", "识别"))
        self.exitButton.setText(_translate("MainWindow", "退出"))
        self.resultLabel.setText(_translate("MainWindow", "识别结果："))
