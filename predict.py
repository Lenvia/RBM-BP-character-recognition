import tensorflow as tf
import joblib
import numpy as np

# 用于归一化
# trainData = joblib.load('./trainData.pkl')
# trX = trainData[:, 1:]
# min_trX = np.min(trX, axis=0)
# deno_trX = np.max(trX, axis=0) - np.min(trX, axis=0) + 0.001

_a = [None] * 5
_w = [None] * 5
_b = [None] * 5


for i in range(4):
    dirs = 'BP_result'
    _w[i] = joblib.load(dirs + '/_w[' + str(i) + '].pkl')
    _b[i] = joblib.load(dirs + '/_b[' + str(i) + '].pkl')

testData = joblib.load('./testData.pkl')
teX = testData[:, 1:]
teX = (teX - np.min(teX, axis=0)) / (np.max(teX, axis=0) - np.min(teX, axis=0) + 0.001)  # 归一化

teY = testData[:, 0]
teY = np.array(teY, dtype=int)
teY = np.eye(10)[teY]  # one-hot

# _a[0] = tf.compat.v1.placeholder("float", [None, 784])  # 输入数据
_a[0] = teX

for i in range(1, 5):
    _a[i] = tf.nn.sigmoid(tf.matmul(_a[i - 1], _w[i - 1]) + _b[i - 1])  # 每一层的输入，第0层的输入是 ?*784的矩阵

predict_op = tf.argmax(_a[-1], 1)

with tf.compat.v1.Session() as sess:
    # 初始化变量
    sess.run(tf.compat.v1.global_variables_initializer())
    result = sess.run(predict_op)

# print("预测结果为：", result)

    print("Accuracy : " + str(np.mean(np.argmax(teY, axis=1) == sess.run(predict_op))))
