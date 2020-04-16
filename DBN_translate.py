import math
import tensorflow as tf
import numpy as np
import joblib
from PIL import Image


trainData = joblib.load('./trainData.pkl')
testData = joblib.load('./testData.pkl')

# 训练集
trX = trainData[:, 1:]
trX = (trX - np.min(trX, axis=0)) / (np.max(trX, axis=0) - np.min(trX, axis=0) + 0.001)  # 归一化
trY = trainData[:, 0]
trY = np.array(trY, dtype=int)
trY = np.eye(10)[trY]  # one-hot

# 测试集
teX = testData[:, 1:]
teX = (teX - np.min(teX, axis=0)) / (np.max(teX, axis=0) - np.min(teX, axis=0) + 0.001)  # 归一化
teY = testData[:, 0]
teY = np.array(teY, dtype=int)
teY = np.eye(10)[teY]  # one-hot


# RBM
class RBM(object):

    def __init__(self, input_size, output_size):
        # Defining the hyperparameters
        self._input_size = input_size  # 可见层（输入层）神经元数
        self._output_size = output_size  # 隐藏层（输出层）神经元数
        self.epochs = 10  # 轮数
        self.learning_rate = 1.0  # 学习率
        self.batchsize = 100  # 块大小

        # 初始化权重矩阵和偏置量
        self.w = np.zeros([input_size, output_size], np.float32)
        self.hb = np.zeros([output_size], np.float32)
        self.vb = np.zeros([input_size], np.float32)

    # 条件概率 P(h=1|v)
    def prob_h_given_v(self, visible, w, hb):
        # Sigmoid
        return tf.nn.sigmoid(tf.matmul(visible, w) + hb)

    # 条件概率 P(v=1|h)
    def prob_v_given_h(self, hidden, w, vb):
        return tf.nn.sigmoid(tf.matmul(hidden, tf.transpose(w)) + vb)

    # 根据概率采样
    def sample_prob(self, probs):
        return tf.nn.relu(tf.sign(probs - tf.random.uniform(tf.shape(probs))))

    # 对每一个batch进行训练
    def train(self, X):
        # placeholder
        _w = tf.compat.v1.placeholder("float", [self._input_size, self._output_size])  # nv * nh
        _hb = tf.compat.v1.placeholder("float", [self._output_size])  # nh * 1
        _vb = tf.compat.v1.placeholder("float", [self._input_size])  # nv * 1

        # 初始化参数
        prv_w = np.zeros([self._input_size, self._output_size], np.float32)
        prv_hb = np.zeros([self._output_size], np.float32)
        prv_vb = np.zeros([self._input_size], np.float32)

        cur_w = tf.compat.v1.placeholder("float", [self._input_size, self._output_size])
        cur_hb = tf.compat.v1.placeholder("float", [self._output_size])
        cur_vb = tf.compat.v1.placeholder("float", [self._input_size])

        # batchsize * input_size
        v0 = tf.compat.v1.placeholder("float", [None, self._input_size])  # 直接放进去一块，而不是一行。行数是batch_size

        # 计算条件概率和采样
        ph0 = self.prob_h_given_v(v0, _w, _hb)
        h0 = self.sample_prob(ph0)
        pv1 = self.prob_v_given_h(h0, _w, _vb)
        v1 = self.sample_prob(pv1)
        ph1 = self.prob_h_given_v(v1, _w, _hb)  # 再次前向传播
        h1 = self.sample_prob(ph1)

        # 计算梯度
        positive_grad = tf.matmul(tf.transpose(v0), ph0)
        negative_grad = tf.matmul(tf.transpose(v1), ph1)

        # 更新参数
        update_w = _w + self.learning_rate * (positive_grad - negative_grad) / tf.cast(tf.shape(v0)[0], dtype=tf.float32)
        update_vb = _vb + self.learning_rate * tf.reduce_mean(v0 - v1, 0)
        update_hb = _hb + self.learning_rate * tf.reduce_mean(ph0 - ph1, 0)

        # 求方差（二阶矩）
        err = tf.reduce_mean(tf.square(v0 - v1))  # tf.square是对每一个元素求平方，直接reduce_mean是取所有元素和的平均值

        # 循环训练
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            # 每一轮就是对整个数据集的过了一遍
            for epoch in range(self.epochs):
                # 对于每一个batch
                for start, end in zip(range(0, len(X), self.batchsize), range(self.batchsize, len(X), self.batchsize)):
                    batch = X[start:end]  # 每次取从start到end这几行
                    # Update the rates
                    cur_w = sess.run(update_w, feed_dict={v0: batch, _w: prv_w, _hb: prv_hb, _vb: prv_vb})
                    cur_hb = sess.run(update_hb, feed_dict={v0: batch, _w: prv_w, _hb: prv_hb, _vb: prv_vb})
                    cur_vb = sess.run(update_vb, feed_dict={v0: batch, _w: prv_w, _hb: prv_hb, _vb: prv_vb})

                    prv_w = cur_w
                    prv_hb = cur_hb
                    prv_vb = cur_vb
                error = sess.run(err, feed_dict={v0: X, _w: cur_w, _vb: cur_vb, _hb: cur_hb})
                print('Epoch: %d' % epoch, 'reconstruction error: %f' % error)
            # RBM训练结果
            self.w = prv_w
            self.hb = prv_hb
            self.vb = prv_vb

    # 训练输出是下一层的输入
    def rbm_outpt(self, X):
        input_X = tf.constant(X)
        _w = tf.constant(self.w)
        _hb = tf.constant(self.hb)
        out = tf.nn.sigmoid(tf.matmul(input_X, _w) + _hb)

        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            # print(sess.run(out))
            return sess.run(out)  # 使用sess.run输出的是np类型的，因为tensor类型的无法作为下一个的输入


# 建立DBN
RBM_hidden_sizes = [500, 200, 100]
rbm_list = []
inpX = trX
# 可见层的神经元数量
input_size = inpX.shape[1]

# 叠加RBM层
for i, size in enumerate(RBM_hidden_sizes):  # enumerate 枚举
    print('RBM: ', i, ' ', input_size, '->', size)
    rbm_list.append(RBM(input_size, size))  # 第二个参数size表示output_size
    input_size = size


# 神经网络
class NN(object):
    # nNet = NN(RBM_hidden_sizes, trX, trY)
    # RBM_hidden_sizes = [500, 200, 50]
    def __init__(self, sizes, X, Y):  # RBM的隐藏层神经元数量，训练集数据，训练集标签
        # Initialize hyperparameters
        self._sizes = sizes
        self._X = X
        self._Y = Y
        self.w_list = []
        self.b_list = []
        self._learning_rate = 1.0  # 学习率
        self._momentum = 0.0  # 动量
        self._epoches = 50  # 轮
        self._batchsize = 100
        input_size = X.shape[1]  # 输入层神经元数，这里为784

        # 初始化的三个等一会都会被覆盖，第四个保留了！！！！！！！！！！！！！所以这个循环的目的主要是为了初始化第四个矩阵
        # Y是经过one-hot编码的，所以Y.shape[1] = 10
        # [500, 200, 50] + [[10]] = [500, 200, 50, 10]
        for size in self._sizes + [Y.shape[1]]:  # [500, 200, 50, 10]
            # Define upper limit for the uniform distribution range
            max_range = 4 * math.sqrt(6. / (input_size + size))

            # 初始化w为 input_size * size 的随机均匀分布矩阵
            self.w_list.append(np.random.uniform(-max_range, max_range, [input_size, size]).astype(np.float32))
            # self.w_list.append(np.zeros([input_size, size], np.float32))

            # 初始化bias
            self.b_list.append(np.zeros([size], np.float32))
            # 输出纬度作为下一个的输入纬度
            input_size = size

    # 从之前训练好的RBM中载入数据
    def load_from_rbms(self, rbm_list):  # load_from_rbms(RBM_hidden_sizes, rbm_list)
        # 接收所有之前RBM训练出来的权重矩阵和偏置量
        for i in range(len(self._sizes)):
            self.w_list[i] = rbm_list[i].w
            self.b_list[i] = rbm_list[i].hb

    # 训练DBN
    def train(self):
        # self.sizes = RBM_hidden_sizes（数组）
        _a = [None] * (len(self._sizes) + 2)  # 多出来的一个是最开头的输入数据，一个是BP网络的输出
        _w = [None] * (len(self._sizes) + 1)  # 多出来的一个是RBM后的BP网络
        _b = [None] * (len(self._sizes) + 1)

        _a[0] = tf.compat.v1.placeholder("float", [None, self._X.shape[1]])  # 输入数据

        y = tf.compat.v1.placeholder("float", [None, self._Y.shape[1]])  # 真实标签

        for i in range(len(self._sizes) + 1):
            _w[i] = tf.Variable(self.w_list[i])
            _b[i] = tf.Variable(self.b_list[i])
        for i in range(1, len(self._sizes) + 2):
            _a[i] = tf.nn.sigmoid(tf.matmul(_a[i - 1], _w[i - 1]) + _b[i - 1])  # 每一层的输入，第0层的输入是 ?*784的矩阵

        # 代价函数
        cost = tf.reduce_mean(tf.square(_a[-1] - y))  # 预测值和真实标签的代价

        # 使用动量优化器
        # minimize在返回是会更新计算图中的var_list，而在之前我们用tf.Variable声明了w和b是变量
        train_op = tf.compat.v1.train.MomentumOptimizer(
            self._learning_rate, self._momentum).minimize(cost)

        # 预测
        predict_op = tf.argmax(_a[-1], 1)  # tf.argmax(x,1)输出的是列向量， 每一行的值为x所有列中最大的 下标
        currentCostList = []
        with tf.compat.v1.Session() as sess:
            # 初始化变量
            sess.run(tf.compat.v1.global_variables_initializer())

            for i in range(self._epoches):
                for start, end in zip(range(0, len(self._X), self._batchsize), range(self._batchsize, len(self._X), self._batchsize)):
                    inputs = self._X[start:end]
                    labels = self._Y[start:end]
                    currentCost = sess.run(cost, feed_dict={_a[0]: inputs, y: labels})
                    currentCostList.append(currentCost)
                    # 优化权重矩阵和偏置量
                    sess.run(train_op, feed_dict={_a[0]: inputs, y: labels})

                for j in range(len(self._sizes) + 1):
                    self.w_list[j] = sess.run(_w[j])
                    self.b_list[j] = sess.run(_b[j])
                # print(self.w_list[3])

                print("Accuracy for epoch " + str(i) + ": " +
                      str(np.mean(np.argmax(self._Y, axis=1) == sess.run(predict_op, feed_dict={_a[0]: self._X, y: self._Y}))))

                # 保存训练模型
                # if i == self._epoches - 1:
                #     for k in range(4):
                #         joblib.dump(self.w_list[k], './train_result/_w[' + str(k) + '].pkl')
                #         joblib.dump(self.b_list[k], './train_result/_b[' + str(k) + '].pkl')
        print(currentCostList)


if __name__ == '__main__':
    # 训练
    for rbm in rbm_list:
        print('New RBM:')
        # 逐层训练RBM
        rbm.train(inpX)  # 训练结果放在self.w, self.hb, self.vb
        # 当前RBM的输出作为下一个RBM的输入
        inpX = rbm.rbm_outpt(inpX)  # 将inpX与上一层RBM的w,hb,vb运算结果当作下一层RBM输入

    # 第三层RBM输出被丢弃，因为直接拿原数据集做为整个DBN的输入
    # RBM_hidden_sizes = [500, 200, 50]
    print("正在训练")
    nNet = NN(RBM_hidden_sizes, trX, trY)
    nNet.load_from_rbms(rbm_list)
    nNet.train()
