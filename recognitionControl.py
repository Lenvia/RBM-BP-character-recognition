import sys
import joblib
import cv2
import os
import numpy as np
import tensorflow as tf
from Ui_recognition import *
from PyQt5.QtWidgets import QApplication, QMainWindow

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QStandardItemModel, QStandardItem, QPixmap
from PyQt5.QtWidgets import QTableView, QHeaderView, QMessageBox


def main():
    # 程序的开始，所有的窗口都由登陆界面（w1）衍生
    app = QApplication(sys.argv)
    w1 = recognitionWindow()  # w1表示登录窗口的对象
    w1.show()
    app.exec_()


# 取反色
def inverse_color(image):
    # 获取高度、宽度
    height, width, temp = image.shape
    img2 = image.copy()

    for i in range(height):
        for j in range(width):
            img2[i, j] = (255-image[i, j][0], 255-image[i, j][1], 255-image[i, j][2])  # 对255去反色
    return img2


def inverse(arr):
    for i in range(0, len(arr)):
        arr[i] = 255 - arr[i]
    return arr


class recognitionWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(recognitionWindow, self).__init__(parent)
        self.setupUi(self)

        # 占位
        self._a = [None] * 5
        self._w = [None] * 5
        self._b = [None] * 5
        self.imagePath = ""
        self.tempPath = ""
        self.savePath = "picData.csv"
        self.result.hide()
        # 用于归一化 很关键！
        self.trainData = joblib.load('./trainData.pkl')
        self.trX = self.trainData[:, 1:]
        self.min_trX = np.min(self.trX, axis=0)
        self.deno_trX = np.max(self.trX, axis=0) - np.min(self.trX, axis=0) + 0.001

        # 读取模型
        for i in range(4):
            dirs = 'BP_result'
            self._w[i] = joblib.load(dirs + '/_w[' + str(i) + '].pkl')
            self._b[i] = joblib.load(dirs + '/_b[' + str(i) + '].pkl')

        self.openButton.clicked.connect(self.loadPicture)
        self.recButton.clicked.connect(self.recPicture)
        self.exitButton.clicked.connect(self.closeWindow)

    def closeWindow(self):
        if self.tempPath != "":
            os.remove(self.tempPath)
        self.close()

    def loadPicture(self):
        if self.tempPath != "":
            os.remove(self.tempPath)
            self.tempPath = ""
        # print("打开文件")
        pathInfo = QtWidgets.QFileDialog.getOpenFileName(self, "选择一张图片", ".")
        self.imagePath = pathInfo[0]  # 获取绝对路径
        self.tempPath = self.imagePath
        if self.imagePath != "":  # 如果路径不为空
            fileFormat = self.imagePath.split('.')[-1]  # 则取文件的后缀名，并判断是否为图片
            # print(fileFormat)
            if fileFormat not in ['jpg', 'JPG', 'jpeg', 'JPEG', 'png', 'PNG', 'bmp', 'BMP']:
                errInfo = QMessageBox.critical(self, "操作反馈", "请选择以下格式的图片：\n jpg, jpeg, png, bmp")
                self.imagePath = ""
            else:  # 是图片的格式，修改大小并显示
                # 修改大小
                tempPic = cv2.imread(self.imagePath)
                tempPic = cv2.resize(tempPic, (200, 200))
                # 获取不含格式的文件名，并写入本地
                splitList = self.imagePath.split('.')
                relative = splitList[0].split('/')[-1]
                self.tempPath = "temp"+relative+'.'+splitList[1]
                cv2.imwrite(self.tempPath, tempPic)
                # 显示
                pix = QPixmap(self.tempPath)
                self.imageLabel.setPixmap(pix)
        self.result.clear()
        self.result.hide()

    def recPicture(self):
        # print("识别图片")
        if self.imagePath == "":
            errInfo = QMessageBox.critical(self, "操作反馈", "您还未选择图片！")
        else:
            # 读取原图
            img_origin = cv2.imread(self.imagePath)
            # # 取反色
            # img_origin = inverse_color(img_origin)
            # 为图片重新指定尺寸
            img_resize = cv2.resize(img_origin, (28, 28))
            # 转为灰度图
            img_gray = cv2.cvtColor(img_resize, cv2.COLOR_RGB2GRAY)

            # 转成一维数组 并保存到本地 必须要写入csv再读取出来，因为写入时是uint8编码，直接转float32值会改变
            img = img_gray.reshape(1, 784)

            # 根据背景判断是否给图片取反色
            count_0 = 0
            count_255 = 0
            for x in img[0]:
                if str(x) == '0':
                    count_0 += 1
                if str(x) == '255':
                    count_255 += 1
            if count_0 < count_255:  # 白色背景
                inverse(img[0])
            # print(count_0, count_255)

            np.savetxt(self.savePath, img, delimiter=',')
            # 读取测试图片
            img = np.genfromtxt(self.savePath, delimiter=",", dtype='float32')
            img = img.reshape(1, 784)
            # 使用训练集的最小值和差值进行归一化
            img = (img - self.min_trX) / self.deno_trX

            # 把图片当作输入
            self._a[0] = img
            # 前向传播
            for i in range(1, 5):
                self._a[i] = tf.nn.sigmoid(tf.matmul(self._a[i - 1], self._w[i - 1]) + self._b[i - 1])  # 每一层的输入，第0层的输入是 ?*784的矩阵
            # 预测
            predict_op = tf.argmax(self._a[-1], 1)
            # 运行
            with tf.compat.v1.Session() as sess:
                # 初始化变量
                sess.run(tf.compat.v1.global_variables_initializer())
                prob = sess.run(self._a[-1])
                result = sess.run(predict_op)

            print("预测结果为：", result[0])
            print("各概率为：", prob)
            text = str(result[0])
            print(text)
            self.result.setText(text)
            self.result.setFocus()
            self.result.show()


if __name__ == '__main__':
    main()
