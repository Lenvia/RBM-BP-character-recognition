# RBM_BP_net
RBM+BP神经网络识别手写数字、英文字符



## 文件目录树

.
├── README.md（使用前必读）
├── Ui_recognition.py
├── Ui_recognition.ui（手写数字识别ui）
├── Ui_recognition2.py
├── Ui_recognition2.ui（英文字母识别ui）
├──**charRec（英文字符项目文件夹）**
│  ├── BP2.py
│  ├── BP_result2
│  ├── Custom picture2
│  ├── RBM_model2
│  ├── convert2.py
│  ├── myRBM2.py
│  └── predict2.py
├── **numRec（数字项目文件夹）**
│  ├── BP.py（BP网络）
│  ├── BP_result（BP训练结果）
│  ├── Custom picture（单个样本测试图片，数字0-9）
│  ├── MNIST_data（MNIST原始数据集）
│  ├── RBM_model（RBM训练结果）
│  ├── convert.py（数据集预加载）
│  ├── myRBM.py（RBM网络）
│  └── predict.py（MNIST测试集）
├── <font color=red>recognitionControl.py</font>（数字识别GUI，系统入口）
├── recognitionControl2.py（字符识别GUI，系统入口）
├── <font color=gray>testData.pkl</font>
├── <font color=gray>testData2.pkl</font>
├── <font color=gray>trainData.pkl</font>
├── <font color=gray>trainData2.pkl</font>
└── tree.txt



**注：**

**若要运行recognitionControl.py，需要按下方步骤生成trainData.pkl和testData.pkl**

**若要运行recognitionControl2.py，需要按下方步骤生成trainData2.pkl和testData2.pkl**



由于英文字符数据集过大，生成trainData2.pkl和testData2.pkl的时间较长，可以直接下载已保存的pkl文件供测试：

链接: https://pan.baidu.com/s/1HmT3oLz2bwlo0YVBfbCkXw  密码: 7d2j



## Requirements

- **joblib** 0.14.1
- **matplotlib** 3.1.3
- **numpy** 1.16.2
- **pyqt5** 5.14.2
- **tensorflow** 1.14.0rc0
- **opencv3**



## 数据集获取及处理

#### MNSIT数据集

官网下载：http://yann.lecun.com/exdb/mnist/

百度网盘：https://pan.baidu.com/s/1EAimqd4yai8sLRcwiVBjUw  密码: m180



使用方法：

1. 将四个压缩包各个解压后放在./numRec/MNIST_data目录下。
2. 运行./numRec/convert.py



#### A-Z Handwritten Alphabets 数据集

下载地址：https://www.kaggle.com/sachinpatel21/az-handwritten-alphabets-in-csv-format

百度网盘：https://pan.baidu.com/s/1MBl0ftiqI_b_YQRCvLqZoQ  密码: k7v7



使用方法：

1. 解压后，将其中一个（解压后有两个，但是完全一样）A_Z Handwritten Data.csv文件放在项目根目录下
2. 运行./charRec/convert2.py （有370K行数据，读取时间较长，可能需要15分钟以上，并且本次系统只使用180K行）



## 运行方法（数字识别）

### 方法一：使用已有的模型

2. 运行./numRec/convert.py预处理数据集（因为在启动窗口时需要读取原数据集进行归一化）

3. 运行./recognitionControl.py

   ```
   python recognitionControl.py
   ```

4. 点击“选择”，弹出的窗口选择一张图片

5. 点击“识别”



### 方法二：自行训练模型

1. 进入项目根目录
2. 预处理数据集
3. 若需自定义RBM层，请修改myRBM.py
4. 若需自定义BP，请修改BP.py
5. 修改完毕后，按方法一的步骤运行



## 运行方法（英文字符）

### 方法一：使用已有的模型

1. 运行./charRec/convet2.py预处理数据集

2. 运行recognitionControl2.py

   ```
   python recognitionControl2.py
   ```

3. 点击“选择”，弹出的窗口选择一张图片

4. 点击“识别”



### 方法二：自行训练模型

1. 进入项目根目录
2. 预处理数据集
3. 若需自定义RBM层，请修改myRBM2.py
4. 若需自定义BP，请修改BP2.py
5. 修改完毕后，按方法一的步骤运行