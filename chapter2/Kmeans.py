# -*- coding: utf-8 -*-
"""
@Time        : 2019/1/16 14:37
@Author      : Dong Hsueh
@Email       : frozenhsueh@gmail.com
@File        : Kmeans.py
@Software    : PyCharm
@introduction: 我们将借助 Tensorflow 来实现一个 K-Means 聚类的程序
"""

import numpy as np
import tensorflow as tf
from tensorflow.contrib.factorization import KMeans
from tensorflow.examples.tutorials.mnist import input_data
import os

def run():
    # 忽略所有的 GPU, 因为随机森林不会从中受益
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    # 使用mnist数据
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
    full_data_x = mnist.train.images

    # 设置超参数
    # num_steps     训练次数
    # batch_size    每次训练用的数据集大小
    # k             分类的数量
    # num_classes   类的数量
    # num_features  特征的数量
    num_steps = 50
    batch_size = 1024
    k = 25
    num_classes = 10
    num_features = 784

    # 定义输入数据
    X = tf.placeholder(tf.float32, shape=[None, num_features])
    # 定义标签
    Y = tf.placeholder(tf.float32, shape=[None, num_classes])

    # 利用 Tensorflow 提供的KMeans来进行计算
    # KMeans(inputs,                                    输入张量或者张量列表
    #       num_clusters,                               一个整数张量，指定簇的数量
    #       initial_clusters=RANDOM_INIT,               指定初始化期间使用的集群。
    #       distance_metric=SQUARED_EUCLIDEAN_DISTANCE, 用于聚类的距离度量。 支持的选项：“平方欧几里得”，“余弦”。
    #       use_mini_batch=False,                       如果为true，请使用小批量k-means算法。 否则假设使用完整批次。
    #       mini_batch_steps_per_iteration=1,           更新后的簇中心同步到主副本的最小步数
    #       random_seed=0,                              PRNG的种子用于初始化种子
    #       kmeans_plus_plus_num_retries=2,             对于在kmeans ++初始化期间采样的每个点，
    #                                                   此参数指定在选择最佳值之前从当前分布中绘制的附加点的数量。
    #                                                   如果指定负值，则使用启发式对O（log（num_to_sample））个附加点进行采样。
    #       kmc2_chain_length=200)                      确定k-MC2算法使用多少个候选点来生成一个新的聚类中心。
    #                                                   如果（小）批次包含较少的点，则从（小）批次生成一个新的集群中心。
    kmeans = KMeans(inputs=X, num_clusters=k, distance_metric='cosine',
                    use_mini_batch=True)

    # 建立 KMeans 计算图
    (all_scores, cluster_idx, scores, cluster_centers_initialized, init_op, train_op) = kmeans.training_graph()
    cluster_idx = cluster_idx[0]
    avg_distance = tf.reduce_mean(scores)

    # 初始化所有变量
    init_vars = tf.global_variables_initializer()

    # 获取计算图
    sess = tf.Session()

    # 初始化变量
    sess.run(init_vars, feed_dict={X: full_data_x})
    sess.run(init_op, feed_dict={X: full_data_x})

    # 训练
    for i in range(1, num_steps + 1):
        _, d, idx = sess.run([train_op, avg_distance, cluster_idx],
                             feed_dict={X: full_data_x})
        if i % 10 == 0 or i == 1:
            print("步数 %i, 平均距离: %f" % (i, d))

    # 给每个中心分配标签
    # 使用每次训练的标签计算每个中心心的标签总数
    # 把样本分配到他们最近的中心
    counts = np.zeros(shape=(k, num_classes))
    for i in range(len(idx)):
        counts[idx[i]] += mnist.train.labels[i]
    # 将最频率最高的标签分配给中心
    labels_map = [np.argmax(c) for c in counts]
    labels_map = tf.convert_to_tensor(labels_map)

    # 评价模型
    cluster_label = tf.nn.embedding_lookup(labels_map, cluster_idx)
    # 计算准确性
    correct_prediction = tf.equal(cluster_label, tf.cast(tf.argmax(Y, 1), tf.int32))
    accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 测试数据
    test_x, test_y = mnist.test.images, mnist.test.labels
    print("在测试集上的准确率:", sess.run(accuracy_op, feed_dict={X: test_x, Y: test_y}))

    return