# RBM_BP_net

RBM+BP神经网络识别手写数字、英文字符



## 注意事项

- 项目存放路径请勿出现中文

  请将RBM_BP_net文件夹提取出来并放在英文路径下，否则会导致图片读取失败


- 不同操作系统中界面显示可能会有差异



## 文件目录树
```
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
├── recognitionControl.py（数字识别GUI，系统入口）
├── recognitionControl2.py（字符识别GUI，系统入口）
├── testData.pkl（经过处理的MNIST测试集）
├── testData2.pkl（经过处理的英文字符测试集）
├── trainData.pkl（经过处理的MNIST训练集）
├── trainData2.pk（经过处理的英文字符训练集）
└── tree.txt
```


**注：**

**若要运行recognitionControl.py，需要trainData.pkl和testData.pkl文件（获取/生成方式见后文）**

**若要运行recognitionControl2.py，需要trainData2.pkl和testData2.pkl文件（获取/生成方式见后文）**



## Requirements

- **joblib** 0.14.1
- **matplotlib** 3.1.3
- **numpy** 1.16.2
- **pyqt5** 5.14.2
- **tensorflow** 1.14.0rc0
- **opencv** 4.1.2



## 数据集获取及处理

英文字符数据集过大，生成trainData2.pkl和testData2.pkl的时间较长（约15min+），可以直接下载已保存的pkl文件供测试，下载完成后放在根目录下：

链接: https://pan.baidu.com/s/1HmT3oLz2bwlo0YVBfbCkXw  密码: 7d2j



**若使用以上链接进行下载，可以跳过本节其余内容。**



#### MNSIT数据集

官网下载：http://yann.lecun.com/exdb/mnist/

百度网盘：https://pan.baidu.com/s/1EAimqd4yai8sLRcwiVBjUw  密码: m180



使用方法：

1. 将四个压缩包各个解压后放在./numRec/MNIST_data目录下。
2. 运行./numRec/convert.py自动生成trainData.pkl和testData.pkl



#### A-Z Handwritten Alphabets 数据集

下载地址：https://www.kaggle.com/sachinpatel21/az-handwritten-alphabets-in-csv-format

百度网盘：https://pan.baidu.com/s/1MBl0ftiqI_b_YQRCvLqZoQ  密码: k7v7



使用方法：

1. 解压后，将其中一个（解压后有两个，但是完全一样）A_Z Handwritten Data.csv文件放在项目根目录下
2. 运行./charRec/convert2.py 自动生成trainData2.pkl和testData2.pkl



## 运行方法（数字识别）

### 方法一：使用已有的模型

1. 下载或自行生成trainData.pkl和testData.pkl

2. 运行./recognitionControl.py

   ```
   python recognitionControl.py
   ```

3. 点击“选择”，弹出的窗口选择一张图片

4. 点击“识别”



### 方法二：自行训练模型

1. 下载或自行生成trainData.pkl和testData.pkl
3. 若需自定义RBM层，请修改./numRec/myRBM.py
4. 若需自定义BP，请修改./numRec/BP.py
5. 修改完毕后，按方法一的步骤运行



## 运行方法（英文字符）

### 方法一：使用已有的模型

1. 下载或自行生成trainData2.pkl和testData2.pkl

2. 在根目录运行recognitionControl2.py

   ```
   python recognitionControl2.py
   ```

3. 点击“选择”，弹出的窗口选择一张图片

4. 点击“识别”



### 方法二：自行训练模型

1. 下载或自行生成trainData2.pkl和testData2.pkl
3. 若需自定义RBM层，请修改./charRec/myRBM2.py
4. 若需自定义BP，请修改./charRec/BP2.py
5. 修改完毕后，按方法一的步骤运行



## 其他事项

./numRec/Custom picture 和 ./charRec/Custiom picture2文件夹内已经存放少量示例图像供测试识别。

在运行recognitionControl.py或recognitionControl2.py时选择其中的图片即可。

