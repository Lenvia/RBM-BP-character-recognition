import math
import tensorflow as tf
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os
import sys

trainData = joblib.load('../trainData.pkl')

trX = trainData[:, 1:]
trX = (trX - np.min(trX, axis=0)) / (np.max(trX, axis=0) - np.min(trX, axis=0) + 0.001)  # 归一化
trY = trainData[:, 0]
trY = np.array(trY, dtype=int)
trY = np.eye(10)[trY]  # one-hot
nh_sizes = [500, 200, 100]  # RBM隐藏层的数目


class BP(object):
    def __init__(self, sizes, X, Y):  # 依次放入 nh_sizes，训练集样本，训练集标签
        self._sizes = sizes
        self._X = X  # 训练数据
        self._Y = Y  # 训练标签
        self.w_list = []  # 权重矩阵列表
        self.b_list = []  # 偏置量列表
        self.eta = 0.2  # 学习率
        self.momentum = 0  # 动量
        self.epochs = 1000
        self.batchsize = 100
        # 分成的块数，self.nblock * self.batchsize = ns
        self.nblock = int((self._X.shape[0] + self.batchsize - 1) / self.batchsize)

        # 载入RBM训练数据
        dirs = 'RBM_model'
        for i in range(len(self._sizes)):
            self.w_list.append(joblib.load(dirs+'/W'+str(i)+'.pkl'))
            self.b_list.append(joblib.load(dirs+'/hbias'+str(i)+'.pkl'))

        # 第四层，使用随机生成的权值，偏置为0
        self.w_list.append(
            np.random.uniform(-0.1, 0.1, [self._sizes[-1], Y.shape[1]]).astype(np.float32))
        self.b_list.append(np.zeros([Y.shape[1]], np.float32))

    def train(self):
        # [[] for _ in k] 表示声明含有k个空数组的数组
        _a = [[] for _ in range(len(self._sizes) + 2)]  # 多出来的一个是最开头的输入数据，一个是BP网络的输出
        _w = [[] for _ in range(len(self._sizes) + 1)]  # 3层RBM+1层BP
        _b = [[] for _ in range(len(self._sizes) + 1)]  # 3层RBM+1层BP

        _a[0] = tf.compat.v1.placeholder("float", [None, self._X.shape[1]])  # 训练数据
        y = tf.compat.v1.placeholder("float", [None, self._Y.shape[1]])  # 真实标签
        keep_prob = tf.compat.v1.placeholder("float", None)  # 保留率（用于dropout泛化）

        # 需要进行优化的变量
        for i in range(len(self._sizes)+1):
            _w[i] = tf.Variable(self.w_list[i])
            _b[i] = tf.Variable(self.b_list[i])

        # 向前传播
        for i in range(1, len(self._sizes)+2):
            _a[i] = tf.nn.sigmoid(tf.matmul(_a[i - 1], _w[i - 1]) + _b[i-1])
            if i != len(self._sizes)+1:  # 除了最后的输出层
                _a[i] = tf.nn.dropout(_a[i], rate=1-keep_prob)  # 每一个中间隐藏层都有rate的可能失活

        # 计算代价
        loss = tf.reduce_mean(tf.square(_a[-1] - y))
        # 使用动量优化器进行优化
        train_op = tf.compat.v1.train.MomentumOptimizer(self.eta, self.momentum).minimize(loss)
        predict_op = tf.argmax(_a[-1], 1)  # tf.argmax(x,1)输出的是列向量， 每一行的值为x所有列中最大的 下标
        # 代价列表
        costList = []
        # 开始训练
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            for epoch in range(self.epochs):
                for iteration in range(1, self.nblock + 1):  # 从第1到第nblock遍，每次取X中的一块
                    # 每一块的起始位置
                    start = (iteration - 1) * self.batchsize
                    end = start + self.batchsize
                    inputs = self._X[start:end]  # 取一块数据。self.batchsize*784
                    labels = self._Y[start:end]
                    # 取每一轮最后一次迭代的error
                    if iteration == self.nblock:
                        error = sess.run(loss, feed_dict={_a[0]: inputs, y: labels, keep_prob: 0.7})
                        costList.append(error)
                    sess.run(train_op, feed_dict={_a[0]: inputs, y: labels, keep_prob: 0.7})

                # 更新变量
                for j in range(len(self._sizes) + 1):
                    self.w_list[j] = sess.run(_w[j])
                    self.b_list[j] = sess.run(_b[j])

                # 输出在训练集上的正确率 和 代价
                # 此时keep_prob设置为1，即所有神经元都保持活性
                print("Accuracy for epoch " + str(epoch) + ": " +
                      str(np.mean(
                          np.argmax(self._Y, axis=1) == sess.run(predict_op, feed_dict={_a[0]: self._X, y: self._Y, keep_prob: 1}))),
                      'reconstruction error: %f' % costList[-1])
                # 每20轮保存训练模型
                if (epoch + 1) % 20 == 0:
                    dirs = 'BP_result'
                    if not os.path.exists(dirs):
                        os.makedirs(dirs)
                    for k in range(4):
                        joblib.dump(self.w_list[k], dirs+'/_w[' + str(k) + '].pkl')
                        joblib.dump(self.b_list[k], dirs+'/_b[' + str(k) + '].pkl')
                sys.stdout.flush()

        # 绘制代价曲线
        plt.plot(costList)
        plt.xlabel("Batch Number")
        plt.ylabel("loss")
        plt.show()


# 启动
bp = BP(nh_sizes, trX, trY)
bp.train()
