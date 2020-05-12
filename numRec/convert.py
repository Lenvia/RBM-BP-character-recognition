import os
import math
import tensorflow as tf
import numpy as np
import joblib


def convert(imgf, labelf, outf, n):
    f = open(imgf, "rb")
    o = open(outf, "w")
    l = open(labelf, "rb")

    f.read(16)
    l.read(8)
    images = []

    for i in range(n):
        image = [ord(l.read(1))]
        for j in range(28*28):
            image.append(ord(f.read(1)))
        images.append(image)

    for image in images:
        o.write(",".join(str(pix) for pix in image)+"\n")
    f.close()
    o.close()
    l.close()


dir0 = 'MNIST_data/'
dirs = 'dataset/'

if not os.path.exists(dirs):
    os.makedirs(dirs)


convert(dir0 + "train-images-idx3-ubyte", dir0 + "train-labels-idx1-ubyte",
        dirs + "mnist_train.csv", 60000)
convert(dir0 + "t10k-images-idx3-ubyte", dir0 + "t10k-labels-idx1-ubyte",
        dirs + "mnist_test.csv", 10000)

print("Convert Finished!")


# 加载数据
trainPath = dirs + "mnist_train.csv"
trainData = np.genfromtxt(trainPath, delimiter=",", dtype='float32')

testPath = dirs + "mnist_test.csv"
testData = np.genfromtxt(testPath, delimiter=",", dtype='float32')

# 保存为pkl格式
joblib.dump(trainData, '../trainData.pkl')
joblib.dump(testData, '../testData.pkl')
print("数据集dump完毕！")
