# -*- coding: utf-8 -*-
"""
@Time        : 2019/1/21 15:39
@Author      : Dong Hsueh
@Email       : frozenhsueh@gmail.com
@File        : VariationalAutoEncoder.py
@Software    : PyCharm
@introduction: 实现一个变分自动编码器
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def run():
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

    # 超参数
    learning_rate = 0.001
    num_steps = 30000
    batch_size = 64

    # 网络参数
    image_dim = 784
    hidden_dim = 512
    latent_dim = 2

    # 自定义初始化 (Xavier初始化)
    def glorot_init(shape):
        return tf.random_normal(shape=shape, stddev=1. / tf.sqrt(shape[0] / 2.))

    # 权重和偏置值
    weights = {
        'encoder_h1': tf.Variable(glorot_init([image_dim, hidden_dim])),
        'z_mean': tf.Variable(glorot_init([hidden_dim, latent_dim])),
        'z_std': tf.Variable(glorot_init([hidden_dim, latent_dim])),
        'decoder_h1': tf.Variable(glorot_init([latent_dim, hidden_dim])),
        'decoder_out': tf.Variable(glorot_init([hidden_dim, image_dim]))
    }
    biases = {
        'encoder_b1': tf.Variable(glorot_init([hidden_dim])),
        'z_mean': tf.Variable(glorot_init([latent_dim])),
        'z_std': tf.Variable(glorot_init([latent_dim])),
        'decoder_b1': tf.Variable(glorot_init([hidden_dim])),
        'decoder_out': tf.Variable(glorot_init([image_dim]))
    }

    # 建立编码器
    input_image = tf.placeholder(tf.float32, shape=[None, image_dim])
    encoder = tf.matmul(input_image, weights['encoder_h1']) + biases['encoder_b1']
    encoder = tf.nn.tanh(encoder)
    z_mean = tf.matmul(encoder, weights['z_mean']) + biases['z_mean']
    z_std = tf.matmul(encoder, weights['z_std']) + biases['z_std']

    # 采样器：正常（高斯）随机分布
    eps = tf.random_normal(tf.shape(z_std), dtype=tf.float32, mean=0., stddev=1.0,
                           name='epsilon')
    z = z_mean + tf.exp(z_std / 2) * eps

    # 建立解码器
    decoder = tf.matmul(z, weights['decoder_h1']) + biases['decoder_b1']
    decoder = tf.nn.tanh(decoder)
    decoder = tf.matmul(decoder, weights['decoder_out']) + biases['decoder_out']
    decoder = tf.nn.sigmoid(decoder)

    # 定义 VAE 损失
    def vae_loss(x_reconstructed, x_true):
        # 重建损失
        encode_decode_loss = x_true * tf.log(1e-10 + x_reconstructed) \
                             + (1 - x_true) * tf.log(1e-10 + 1 - x_reconstructed)
        encode_decode_loss = -tf.reduce_sum(encode_decode_loss, 1)
        # KL散度损失
        kl_div_loss = 1 + z_std - tf.square(z_mean) - tf.exp(z_std)
        kl_div_loss = -0.5 * tf.reduce_sum(kl_div_loss, 1)
        return tf.reduce_mean(encode_decode_loss + kl_div_loss)

    loss_op = vae_loss(decoder, input_image)
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)

    # 初始化
    init = tf.global_variables_initializer()

    # 开始训练
    sess = tf.Session()
    sess.run(init)
    for i in range(1, num_steps + 1):
        # 准备数据
        batch_x, _ = mnist.train.next_batch(batch_size)
        # 训练
        feed_dict = {input_image: batch_x}
        _, l = sess.run([train_op, loss_op], feed_dict=feed_dict)
        if i % 1000 == 0 or i == 1:
            print('步数 %i, 损失: %f' % (i, l))

    # 测试
    noise_input = tf.placeholder(tf.float32, shape=[None, latent_dim])
    # 为了从噪声中创建图片来重建编码器
    decoder = tf.matmul(noise_input, weights['decoder_h1']) + biases['decoder_b1']
    decoder = tf.nn.tanh(decoder)
    decoder = tf.matmul(decoder, weights['decoder_out']) + biases['decoder_out']
    decoder = tf.nn.sigmoid(decoder)

    # 构建生成数字的流形
    n = 20
    x_axis = np.linspace(-3, 3, n)
    y_axis = np.linspace(-3, 3, n)

    canvas = np.empty((28 * n, 28 * n))
    for i, yi in enumerate(x_axis):
        for j, xi in enumerate(y_axis):
            z_mu = np.array([[xi, yi]] * batch_size)
            x_mean = sess.run(decoder, feed_dict={noise_input: z_mu})
            canvas[(n - i - 1) * 28:(n - i) * 28, j * 28:(j + 1) * 28] = \
                x_mean[0].reshape(28, 28)

    plt.figure(figsize=(8, 10))
    Xi, Yi = np.meshgrid(x_axis, y_axis)
    plt.imshow(canvas, origin="upper", cmap="gray")
    plt.show()


    return