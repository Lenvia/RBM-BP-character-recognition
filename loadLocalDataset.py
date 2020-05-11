import math
import tensorflow as tf
import numpy as np
import joblib

# 加载数据并保存
trainPath = "./dataset/mnist_train.csv"
trainData = np.genfromtxt(trainPath, delimiter=",", dtype='float32')

testPath = "./dataset/mnist_test.csv"
testData = np.genfromtxt(testPath, delimiter=",", dtype='float32')

joblib.dump(trainData, './trainData.pkl')
joblib.dump(testData, './testData.pkl')
