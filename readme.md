# tensorflow 学习笔记
This notebook is write in chinese.if you want to see the English version, please click the original author's link:
[TensorFlow Examples](https://github.com/aymericdamien/TensorFlow-Examples#tutorial-index)

Note: The original authors of the following codes is Aymeric Damien.

注意：以下代码的原作者均为 Aymeric Damien。


该项目主要来源于Github上的开源项目
[TensorFlow Examples](https://github.com/aymericdamien/TensorFlow-Examples#tutorial-index) ，由于原文主要是英文版，因此本项目主要是对原来项目的翻译，以及添加一些自己写的Demo便于理解与记忆。
## 环境配置
+ Pycharm Community Edition 2018.3.3
+ Anaconda3
+ Python 3.6
+ NVIDIA GPU
+ tensorflow 1.12.0
+ tensorflow-base 1.12.0
+ tensorflow-gpu 1.12.0
## 章节列表
运行方式：在main.py文件中import想要运行的代码的run()方法即可，main.py中给出了运行第一章第一节代码的方法。
+ Chapter 1
    + ["HELLO WORLD"](https://github.com/D-Hsueh/TensorflowLearn/blob/master/chapter1/HelloWorld.py)
    + [基础的操作](https://github.com/D-Hsueh/TensorflowLearn/blob/master/chapter1/BasicOperations.py)
    + [动态图API](https://github.com/D-Hsueh/TensorflowLearn/blob/master/chapter1/BasicEagerAPI.py)
+ Chapter 2
    + [线性回归模型](https://github.com/D-Hsueh/TensorflowLearn/blob/master/chapter2/LinearRegression.py)
    + [基于动态图实现的线性回归模型](https://github.com/D-Hsueh/TensorflowLearn/blob/master/chapter2/LinearRegressionWithEagerAPI.py)
    + [逻辑回归模型](https://github.com/D-Hsueh/TensorflowLearn/blob/master/chapter2/LogisticRegression.py)
    + [最近邻模型](https://github.com/D-Hsueh/TensorflowLearn/blob/master/chapter2/NearestNeighbor.py)
    + [K-Means聚类模型](https://github.com/D-Hsueh/TensorflowLearn/blob/master/chapter2/Kmeans.py)
    + [随机森林](https://github.com/D-Hsueh/TensorflowLearn/blob/master/chapter2/RandomForest.py)
    + [梯度提升决策树](https://github.com/D-Hsueh/TensorflowLearn/blob/master/chapter2/GradientBoostedDecisionTree.py)
    + [单词（字）的向量表示](https://github.com/D-Hsueh/TensorflowLearn/blob/master/chapter2/Word2Vec.py)
+ chapter 3
    + [监督学习](https://github.com/D-Hsueh/TensorflowLearn/tree/master/chapter3/Supervised)
        + [一个神经网络的例子](https://github.com/D-Hsueh/TensorflowLearn/blob/master/chapter3/Supervised/NeuralNetworkExample.py)
        + [利用 Tensorflow 提供的 API 更快的实现神经网络](https://github.com/D-Hsueh/TensorflowLearn/blob/master/chapter3/Supervised/NeuralNetworkExamplewithTFAPI.py)
        + [卷积神经网络](https://github.com/D-Hsueh/TensorflowLearn/blob/master/chapter3/Supervised/ConvolutionalNeuralNetwork.py)
        + [利用 Tensorflow 提供的 API 更快的实现卷积神经网络](https://github.com/D-Hsueh/TensorflowLearn/blob/master/chapter3/Supervised/ConvolutionalNeuralNetworkwithTFAPI.py)
        + [循环神经网络（LSTM实现）](https://github.com/D-Hsueh/TensorflowLearn/blob/master/chapter3/Supervised/LSTM.py)
        + [双向递归神经网络（LSTM实现）](https://github.com/D-Hsueh/TensorflowLearn/blob/master/chapter3/Supervised/bidirectionalLSTM.py)
    + [无监督学习](https://github.com/D-Hsueh/TensorflowLearn/tree/master/chapter3/Unsupervised)
        + [自动编码器](https://github.com/D-Hsueh/TensorflowLearn/tree/master/chapter3/Unsupervised/AutoEncoder.py)
        + [变分自动编码器（VAE）](https://github.com/D-Hsueh/TensorflowLearn/tree/master/chapter3/Unsupervised/VariationalAutoEncoder.py)
        + [生成对抗网络（GAN）](https://github.com/D-Hsueh/TensorflowLearn/tree/master/chapter3/Unsupervised/GAN.py)
        + [深度卷积生成对抗网络（DCGAN）](https://github.com/D-Hsueh/TensorflowLearn/tree/master/chapter3/Unsupervised/DCGAN.py)
+ chapter 4
    + [模型的保存和加载](https://github.com/D-Hsueh/TensorflowLearn/tree/master/chapter4/SaveRestoreModel.py)
    + [Tensorboard基础](https://github.com/D-Hsueh/TensorflowLearn/tree/master/chapter4/TensorboardBasic.py)
    