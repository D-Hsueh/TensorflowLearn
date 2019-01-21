# -*- coding: utf-8 -*-
"""
@Time        : 2019/1/21 16:25
@Author      : Dong Hsueh
@Email       : frozenhsueh@gmail.com
@File        : DCGAN.py
@Software    : PyCharm
@introduction: 实现一个深度卷积生成对抗网络
"""
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def run():
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

    # 训练超参数
    num_steps = 10000
    batch_size = 128
    lr_generator = 0.002
    lr_discriminator = 0.002

    # 网络参数
    image_dim = 784
    noise_dim = 100

    # 网络的输入
    noise_input = tf.placeholder(tf.float32, shape=[None, noise_dim])
    real_image_input = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
    # 用一个 bool 值来表示是否在训练时间。
    is_training = tf.placeholder(tf.bool)

    # LeakyReLU 激活函数
    def leakyrelu(x, alpha=0.2):
        return 0.5 * (1 + alpha) * x + 0.5 * (1 - alpha) * abs(x)

    # 生成网络
    # 输入: 随机噪音点, 输出: 图像
    # 请注意，批量标准化在训练和推理时具有不同的行为，
    # 因此我们使用占位符来指示我们是否正在训练的图层。
    def generator(x, reuse=False):
        with tf.variable_scope('Generator', reuse=reuse):
            # TensorFlow图层根据输入自动创建变量并计算其形状。
            x = tf.layers.dense(x, units=7 * 7 * 128)
            x = tf.layers.batch_normalization(x, training=is_training)
            x = tf.nn.relu(x)
            # 变形成四维数据格式: (batch, height, width, channels)
            # 新的格式: (batch, 7, 7, 128)
            x = tf.reshape(x, shape=[-1, 7, 7, 128])
            # 解卷积, 图像格式: (batch, 14, 14, 64)
            x = tf.layers.conv2d_transpose(x, 64, 5, strides=2, padding='same')
            x = tf.layers.batch_normalization(x, training=is_training)
            x = tf.nn.relu(x)
            # 解卷积, 图像格式: (batch, 28, 28, 1)
            x = tf.layers.conv2d_transpose(x, 1, 5, strides=2, padding='same')
            # 利用 tan 来获得更好地稳定性
            x = tf.nn.tanh(x)
            return x

    # 对抗网络
    # 输入: 图像, 输出: 预测真假
    def discriminator(x, reuse=False):
        with tf.variable_scope('Discriminator', reuse=reuse):
            # 利用卷积神经网络对图像进行分类
            x = tf.layers.conv2d(x, 64, 5, strides=2, padding='same')
            x = tf.layers.batch_normalization(x, training=is_training)
            x = leakyrelu(x)
            x = tf.layers.conv2d(x, 128, 5, strides=2, padding='same')
            x = tf.layers.batch_normalization(x, training=is_training)
            x = leakyrelu(x)
            # 把多维数据一维化 ———— Flatten
            x = tf.reshape(x, shape=[-1, 7 * 7 * 128])
            x = tf.layers.dense(x, 1024)
            x = tf.layers.batch_normalization(x, training=is_training)
            x = leakyrelu(x)
            # 输出是两类
            x = tf.layers.dense(x, 2)
        return x

    # 建立生成网络
    gen_sample = generator(noise_input)

    # 建立两个对抗网络
    disc_real = discriminator(real_image_input)
    disc_fake = discriminator(gen_sample, reuse=True)

    # 建立堆叠的对抗/生成
    stacked_gan = discriminator(gen_sample, reuse=True)

    # 损失函数 (真的为1，假的为0)
    disc_loss_real = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=disc_real, labels=tf.ones([batch_size], dtype=tf.int32)))
    disc_loss_fake = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=disc_fake, labels=tf.zeros([batch_size], dtype=tf.int32)))
    # 损失函数求和
    disc_loss = disc_loss_real + disc_loss_fake
    # 生成网络损失函数（试图让对抗网络认为他的生成图是真）
    gen_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=stacked_gan, labels=tf.ones([batch_size], dtype=tf.int32)))

    # 建立优化器
    optimizer_gen = tf.train.AdamOptimizer(learning_rate=lr_generator, beta1=0.5, beta2=0.999)
    optimizer_disc = tf.train.AdamOptimizer(learning_rate=lr_discriminator, beta1=0.5, beta2=0.999)

    # 需要训练的变量
    # 对抗网络变量
    gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')
    # 生成网络变量
    disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')

    # 创建训练操作
    # TensorFlow 的 UPDATE_OPS 集合 保存所有的批处理操作来更新变量
    gen_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Generator')
    # control_dependencies 确保 gen_update_ops 将在 minimize 操作（反省传播）之前运行
    with tf.control_dependencies(gen_update_ops):
        train_gen = optimizer_gen.minimize(gen_loss, var_list=gen_vars)
    disc_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Discriminator')
    with tf.control_dependencies(disc_update_ops):
        train_disc = optimizer_disc.minimize(disc_loss, var_list=disc_vars)

    init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init)
    # 训练
    for i in range(1, num_steps + 1):
        # 准备输入数据
        batch_x, _ = mnist.train.next_batch(batch_size)
        batch_x = np.reshape(batch_x, newshape=[-1, 28, 28, 1])
        # 调整到[-1, 1] ———— 对抗网络识别的部分
        batch_x = batch_x * 2. - 1.

        # 对抗训练
        # 产生随机噪声点来给生成网络
        z = np.random.uniform(-1., 1., size=[batch_size, noise_dim])
        _, dl = sess.run([train_disc, disc_loss],
                         feed_dict={real_image_input: batch_x, noise_input: z, is_training: True})

        # 生成训练
        # 产生随机噪声点来给生成网络
        z = np.random.uniform(-1., 1., size=[batch_size, noise_dim])
        _, gl = sess.run([train_gen, gen_loss], feed_dict={noise_input: z, is_training: True})

        if i % 500 == 0 or i == 1:
            print('步数 %i: 生成损失: %f, 对抗损失: %f' % (i, gl, dl))

    # 测试
    n = 6
    canvas = np.empty((28 * n, 28 * n))
    for i in range(n):
        z = np.random.uniform(-1., 1., size=[n, noise_dim])
        g = sess.run(gen_sample, feed_dict={noise_input: z, is_training: False})
        g = (g + 1.) / 2.
        g = -1 * (g - 1)
        for j in range(n):
            canvas[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = g[j].reshape([28, 28])

    plt.figure(figsize=(n, n))
    plt.imshow(canvas, origin="upper", cmap="gray")
    plt.show()


    return