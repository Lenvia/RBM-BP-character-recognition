import os
import csv
import re
import cv2
import numpy as np
import joblib


path = '../A_Z Handwritten Data.csv'
print("正在读取...")
Data = np.genfromtxt(path, delimiter=",", dtype='float32')
# 这里是保存全部的数据，如果需要扩大训练规模，只要加载这个就行了，不用再读取csv了（把上面的注释掉）
joblib.dump(Data, '../totalData.pkl')


np.random.shuffle(Data)

trainData = Data[0:119999, :]
testData = Data[120000:179999, :]

print(trainData)
print(testData)
# 保存为pkl格式
# dump训练集和测试集
joblib.dump(trainData, '../trainData2.pkl')
print("训练集数据已dump")
joblib.dump(testData, '../testData2.pkl')
print("测试集数据已dump")

