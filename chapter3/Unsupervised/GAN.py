# -*- coding: utf-8 -*-
"""
@Time        : 2019/1/21 15:54
@Author      : Dong Hsueh
@Email       : frozenhsueh@gmail.com
@File        : GAN.py
@Software    : PyCharm
@introduction: 实现一个生成对抗网络
"""
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def run():
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

    # 训练超参数
    num_steps = 70000
    batch_size = 128
    learning_rate = 0.0002

    # 网络超参数
    image_dim = 784
    gen_hidden_dim = 256
    disc_hidden_dim = 256
    noise_dim = 100         # 噪音点

    # Xavier 自定义初始化
    def glorot_init(shape):
        return tf.random_normal(shape=shape, stddev=1. / tf.sqrt(shape[0] / 2.))

    # 权重和偏置值
    weights = {
        'gen_hidden1': tf.Variable(glorot_init([noise_dim, gen_hidden_dim])),
        'gen_out': tf.Variable(glorot_init([gen_hidden_dim, image_dim])),
        'disc_hidden1': tf.Variable(glorot_init([image_dim, disc_hidden_dim])),
        'disc_out': tf.Variable(glorot_init([disc_hidden_dim, 1])),
    }
    biases = {
        'gen_hidden1': tf.Variable(tf.zeros([gen_hidden_dim])),
        'gen_out': tf.Variable(tf.zeros([image_dim])),
        'disc_hidden1': tf.Variable(tf.zeros([disc_hidden_dim])),
        'disc_out': tf.Variable(tf.zeros([1])),
    }

    # 生成网络
    def generator(x):
        hidden_layer = tf.matmul(x, weights['gen_hidden1'])
        hidden_layer = tf.add(hidden_layer, biases['gen_hidden1'])
        hidden_layer = tf.nn.relu(hidden_layer)
        out_layer = tf.matmul(hidden_layer, weights['gen_out'])
        out_layer = tf.add(out_layer, biases['gen_out'])
        out_layer = tf.nn.sigmoid(out_layer)
        return out_layer

    # 对抗网络
    def discriminator(x):
        hidden_layer = tf.matmul(x, weights['disc_hidden1'])
        hidden_layer = tf.add(hidden_layer, biases['disc_hidden1'])
        hidden_layer = tf.nn.relu(hidden_layer)
        out_layer = tf.matmul(hidden_layer, weights['disc_out'])
        out_layer = tf.add(out_layer, biases['disc_out'])
        out_layer = tf.nn.sigmoid(out_layer)
        return out_layer

    # 网络输入数据：
    # 生成网络的输入为随机的噪音点
    # 对方网络的输入是生成网络生成的图片
    gen_input = tf.placeholder(tf.float32, shape=[None, noise_dim], name='input_noise')
    disc_input = tf.placeholder(tf.float32, shape=[None, image_dim], name='disc_input')

    # 建立生成网络
    gen_sample = generator(gen_input)

    # 建立两个对抗网络，一个是真实的样本，一个是来源于生成网络的输入
    disc_real = discriminator(disc_input)
    disc_fake = discriminator(gen_sample)

    # 建立损失
    gen_loss = -tf.reduce_mean(tf.log(disc_fake))
    disc_loss = -tf.reduce_mean(tf.log(disc_real) + tf.log(1. - disc_fake))

    # 建立优化器
    optimizer_gen = tf.train.AdamOptimizer(learning_rate=learning_rate)
    optimizer_disc = tf.train.AdamOptimizer(learning_rate=learning_rate)

    # 每个优化器的需要更新的变量
    # 默认情况下，在TensorFlow中，每个优化器都会更新所有变量，
    # 因此我们需要为每个变量精确更新要更新的特定变量。
    # 生成网络
    gen_vars = [weights['gen_hidden1'], weights['gen_out'],
                biases['gen_hidden1'], biases['gen_out']]
    # 对抗网络
    disc_vars = [weights['disc_hidden1'], weights['disc_out'],
                 biases['disc_hidden1'], biases['disc_out']]

    # 创建训练操作
    train_gen = optimizer_gen.minimize(gen_loss, var_list=gen_vars)
    train_disc = optimizer_disc.minimize(disc_loss, var_list=disc_vars)

    init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init)
    # 训练
    for i in range(1, num_steps + 1):
        batch_x, _ = mnist.train.next_batch(batch_size)
        # 随机生成噪音点
        z = np.random.uniform(-1., 1., size=[batch_size, noise_dim])
        feed_dict = {disc_input: batch_x, gen_input: z}
        _, _, gl, dl = sess.run([train_gen, train_disc, gen_loss, disc_loss],
                                feed_dict=feed_dict)
        if i % 2000 == 0 or i == 1:
            print('步数 %i: 生成网络损失: %f, 对抗网络损失: %f' % (i, gl, dl))

    # 测试
    n = 6
    canvas = np.empty((28 * n, 28 * n))
    for i in range(n):
        # 以随机点做输入
        z = np.random.uniform(-1., 1., size=[n, noise_dim])
        g = sess.run(gen_sample, feed_dict={gen_input: z})
        # 反色以更好地显示
        g = -1 * (g - 1)
        for j in range(n):
            canvas[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = g[j].reshape([28, 28])

    plt.figure(figsize=(n, n))
    plt.imshow(canvas, origin="upper", cmap="gray")
    plt.show()
    return