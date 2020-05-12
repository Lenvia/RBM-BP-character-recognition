import numpy as np
import numpy.matlib
import tensorflow as tf
import matplotlib.pyplot as plt
import joblib
import pdb
import os
np.set_printoptions(threshold=np.inf)

trainData = joblib.load('../trainData.pkl')

trX = trainData[:, 1:]
trX = (trX - np.min(trX, axis=0)) / (np.max(trX, axis=0) - np.min(trX, axis=0) + 0.001)  # 归一化


nh_sizes = [500, 200, 100]  # RBM隐藏层的数目
RBM_list = []
input_X = trX  # 初始输入数据
input_size = trX.shape[1]  # 可见层元素数量

global number  # 用来区分存储的RBM模型
number = 0


class RBM(object):
    def __init__(self, input_size, output_size):
        # 归一化最佳学习率为0.01
        self.eta = 0.01  # 学习率
        self.nv = input_size  # 可见层单元数目
        self.nh = output_size  # 隐藏层单元数目
        self.epochs = 100  # 轮数
        self.batchsize = 100  # 每一块的元素数量
        # 分成的块数，self.nblock * self.batchsize = ns
        self.nblock = int((self.nv + self.batchsize - 1) / self.batchsize)

        # 初始化原始参数
        self.hbias = np.zeros([self.nh], np.float32)  # 前向传播偏置量
        self.vbias = np.zeros([self.nv], np.float32)  # 反向重构偏置量
        self.W = np.zeros((self.nv, self.nh), np.float32)  # 权重矩阵
        # dropout泛化
        # self.keep_prob = 0.7

    def train(self, X):
        global number  # 用来区分保存模型序号
        errors =[]  # 存放每一次训练的error，用来绘图
        input_X = X
        # 占位符，下面用sess.run来填充
        _w = tf.compat.v1.placeholder("float", [self.nv, self.nh])  # 尺寸：nv * nh
        _hb = tf.compat.v1.placeholder("float", [self.nh])  # 尺寸：nh * 1
        _vb = tf.compat.v1.placeholder("float", [self.nv])  # 尺寸：nv * 1
        # 初始化超参数
        var_w = np.zeros([self.nv, self.nh], np.float32)
        var_hb = np.zeros([self.nh], np.float32)
        var_vb = np.zeros([self.nv], np.float32)

        # batchsize * input_size
        v0 = tf.compat.v1.placeholder("float", [None, self.nv])  # 直接放进去一块，而不是一行。行数是batch_size

        # sample_h_given_v
        # 在已知V的情况下，在隐藏层H0中第j个隐藏层神经元取值为1的概率为 pH0_j
        ph0 = tf.nn.sigmoid(tf.matmul(v0, _w) + _hb)  # 公式(3.30)。 pH0是行向量
        tempH0 = ph0 - tf.random.uniform(tf.shape(ph0))  # 产生[0,1]上的随机数，并作差
        # 如果ph0>随机数，则对应的h0为1；否则为0
        h0 = tf.maximum((tf.sign(tempH0)), 0)
        # dropout
        # h0 = tf.nn.dropout(h0, rate=1-self.keep_prob)

        # sample_v_given_h
        # 第一次反向重构，在已知H0的情况下，第i个可见层神经元取值为1的概率为 pV1_i
        pv1 = tf.nn.sigmoid(tf.matmul(h0, tf.transpose(_w)) + _vb)  # 公式(3.31)
        tempV1 = pv1 - tf.random.uniform(tf.shape(pv1))
        v1 = tf.maximum((tf.sign(tempV1)), 0)  # 反向重构产生的V1
        # v1 = tf.nn.dropout(v1, rate=1-self.keep_prob)

        ph1 = tf.nn.sigmoid(tf.matmul(v1, _w) + _hb)  # 再次前向传播

        # 计算梯度
        # lnP(v0)对w求偏导 = positive_grad - negative_grad
        positive_grad = tf.matmul(tf.transpose(v0), ph0)
        negative_grad = tf.matmul(tf.transpose(v1), ph1)

        # 更新参数
        update_w = _w + self.eta * (positive_grad - negative_grad) / tf.cast(tf.shape(v0)[0], dtype=tf.float32)
        # lnP(v0)对vbias求偏导 tf.reduce_mean(v0 - v1, 0)
        # lnP(v0)对hbias求偏导 tf.reduce_mean(ph0 - ph1, 0)
        update_vb = _vb + self.eta * tf.reduce_mean(v0 - v1, 0)
        update_hb = _hb + self.eta * tf.reduce_mean(ph0 - ph1, 0)

        # 求方差（二阶矩）
        err = tf.reduce_mean(tf.square(v0 - v1))  # tf.square是对每一个元素求平方，直接reduce_mean是取所有元素和的平均值
        # 循环训练
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            # 每一轮就是对整个数据集的过了一遍
            for epoch in range(self.epochs):
                for iteration in range(1, self.nblock + 1):  # 从第1到第nblock遍，每次取X中的一块
                    # sess.run(tf.compat.v1.global_variables_initializer())
                    # 每一块的起始位置
                    start = (iteration - 1) * self.batchsize
                    end = start + self.batchsize
                    V0 = input_X[start:end, :]  # 取一块数据。self.batchsize*784

                    var_w = sess.run(update_w, feed_dict={v0: V0, _w: var_w, _hb: var_hb, _vb: var_vb})
                    var_hb = sess.run(update_hb, feed_dict={v0: V0, _w: var_w, _hb: var_hb, _vb: var_vb})
                    var_vb = sess.run(update_vb, feed_dict={v0: V0, _w: var_w, _hb: var_hb, _vb: var_vb})

                    error = sess.run(err, feed_dict={v0: X, _w: var_w, _vb: var_vb, _hb: var_hb})
                    errors.append(error)
                print('Epoch: %d' % epoch, 'reconstruction error: %f' % errors[-1])

        self.W = var_w
        self.hbias = var_hb
        self.vbias = var_vb

        plt.plot(errors)
        plt.xlabel("Batch Number")
        plt.ylabel("Error")
        plt.show()

        dirs = 'RBM_model'
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        joblib.dump(self.W, dirs+'/W'+str(number)+'.pkl')
        joblib.dump(self.hbias, dirs + '/hbias'+str(number)+'.pkl')
        number = number + 1

        input_X = tf.constant(X)
        w = tf.constant(self.W)
        hb = tf.constant(self.hbias)
        out = tf.nn.sigmoid(tf.matmul(input_X, w) + hb)

        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            return sess.run(out)  # 使用sess.run输出的是np类型的，因为tensor类型的无法作为下一个的输入


# 叠加RBM层
for i, output_size in enumerate(nh_sizes):  # enumerate 枚举
    RBM_list.append(RBM(input_size, output_size))  # 第二个参数size表示output_size
    input_size = output_size

for rbm in RBM_list:
    # 逐层训练RBM
    output_X = rbm.train(input_X)  # 上一层训练输出为下一层的输入
    print("-----------****----------")
    input_X = output_X

