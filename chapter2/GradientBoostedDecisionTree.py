# -*- coding: utf-8 -*-
"""
@Time        : 2019/1/16 20:25
@Author      : Dong Hsueh
@Email       : frozenhsueh@gmail.com
@File        : GradientBoostedDecisionTree.py
@Software    : PyCharm
@introduction: 实现了一个基于 Tensorflow 的梯度提升决策树
"""

import tensorflow as tf
from tensorflow.contrib.boosted_trees.estimator_batch.estimator import GradientBoostedDecisionTreeClassifier
from tensorflow.contrib.boosted_trees.proto import learner_pb2 as gbdt_learner
import os
from tensorflow.examples.tutorials.mnist import input_data

def run():
    # 忽略所有的 GPU, 当前的 TF GBDT 不支持 GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


    # 设置 Tensorflow 的 log 只显示 error
    tf.logging.set_verbosity(tf.logging.ERROR)
    # 导入 MNIST 数据集
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=False,
                                      source_url='http://yann.lecun.com/exdb/mnist/')

    # 超参数设置
    batch_size = 4096       # 每批用来训练的数据的大小
    num_classes = 10        # 最终结果一共有基类
    num_features = 784      # 输入数据的特征数——每张图片的大小是 28 * 28 像素
    max_steps = 10000

    # GBDT 超参数
    learning_rate = 0.1             # 学习率
    l1_regul = 0.                   # L1 正则化
    l2_regul = 1.                   # L2 正则化
    examples_per_layer = 1000
    num_trees = 10                  # 树的数量
    max_depth = 16                  # 最大深度

    # 设置 GBDT 的超参数
    learner_config = gbdt_learner.LearnerConfig()
    learner_config.learning_rate_tuner.fixed.learning_rate = learning_rate
    learner_config.regularization.l1 = l1_regul
    learner_config.regularization.l2 = l2_regul / examples_per_layer
    learner_config.constraints.max_tree_depth = max_depth
    growing_mode = gbdt_learner.LearnerConfig.LAYER_BY_LAYER
    learner_config.growing_mode = growing_mode
    run_config = tf.contrib.learn.RunConfig(save_checkpoints_secs=300)
    learner_config.multi_class_strategy = (
        gbdt_learner.LearnerConfig.DIAGONAL_HESSIAN)

    # 创建TensorFlor GBDT 模型
    gbdt_model = GradientBoostedDecisionTreeClassifier(
        model_dir=None,  # 不指定保存位置
        learner_config=learner_config,
        n_classes=num_classes,
        examples_per_layer=examples_per_layer,
        num_trees=num_trees,
        center_bias=False,
        config=run_config)

    # 展示 log 的 info 信息
    tf.logging.set_verbosity(tf.logging.INFO)

    # 定义用来训练的输入数据
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'images': mnist.train.images}, y=mnist.train.labels,
        batch_size=batch_size, num_epochs=None, shuffle=True)

    # 拟合模型
    gbdt_model.fit(input_fn=input_fn, max_steps=max_steps)

    # 评估拟合后的模型
    # 定义用于评估的输入
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'images': mnist.test.images}, y=mnist.test.labels,
        batch_size=batch_size, shuffle=False)

    # 使用 评估 方法来进行评估
    e = gbdt_model.evaluate(input_fn=input_fn)
    print("测试准确度为:", e['accuracy'])


    return