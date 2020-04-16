import math
import tensorflow as tf
import numpy as np
from sklearn.externals import joblib

# 加载数据并保存
trainPath = "mnist_train.csv"
trainData = np.genfromtxt(trainPath, delimiter=",", dtype='float32')

testPath = "mnist_test.csv"
testData = np.genfromtxt(testPath, delimiter=",", dtype='float32')

joblib.dump(trainData, './trainData.pkl')
joblib.dump(testData, './testData.pkl')

# trainData = joblib.load('./trainData.pkl')
# testData = joblib.load('./testData.pkl')
#
# trX = trainData[:, 1:]
# trX = (trX - np.min(trX, axis=0)) / (np.max(trX, axis=0) - np.min(trX, axis=0) + 0.001)  # 归一化
# trY = trainData[:, 0]
# trY = np.array(trY, dtype=int)
# trY = np.eye(10)[trY]  # one-hot
#
# teX = testData[:, 1:]
# teX = (teX - np.min(teX, axis=0)) / (np.max(teX, axis=0) - np.min(teX, axis=0) + 0.001)  # 归一化
# teY = testData[:, 0]
# teY = np.array(teY, dtype=int)
# teY = np.eye(10)[teY]  # one-hot
