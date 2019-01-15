# -*- coding: utf-8 -*-
"""
@Time        : 2019/1/15 16:54
@Author      : Dong Hsueh
@Email       : frozenhsueh@gmail.com
@File        : LogisticRegression.py
@Software    : PyCharm
@introduction: 在本部分，我们将实现一个简单的逻辑回归模型
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def run():
    # 导入 MNIST 数据集
    # 该数据集包含60,000个用于培训的示例和10,000个用于测试的示例。
    # 这些数字已经过尺寸标准化，并且以固定大小的图像（28x28像素）为中心，其值为0到1。
    # 为简单起见，每个图像都被展平并转换为784个特征的1-D numpy数组（28 * 28））。
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

    # 超参数初始化：
    # learning_rate     学习率
    # training_epochs   训练次数
    # batch_size        每次训练集大小
    # display_step      展示间隔
    learning_rate = 0.01
    training_epochs = 25
    batch_size = 100
    display_step = 1

    # 定义计算图的输入数据
    # 输入数据 X ： MNIST 数据集是一个 28 * 28 的数组，因此可以转换成一个 28 * 28 = 784 长度的一维向量
    # 输入数据 y ： 由于输出数据是 0 - 9 十个数字的分类，因此是一个长度为 10 的一维向量
    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 10])

    # 设置模型的参数变量，其中 W , b 的 shape 由 X , y 共同决定
    # 初始化时将其全部初始化为 0 .
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    # 构造模型 采用 softmax 对输出的结果进行 softmax 激活
    # tf.nn.softmax(    计算softmax激活。
    #     logits,       一个非空张量,必须是以下类型之一：half, float32, float64
    #     axis=None,    将被执行的softmax维度,默认值是-1，表示最后一个维度。
    #     name=None,    操作的名称（可选）。
    #     dim=None      弃用，axis的别名。
    # )
    # Softmax简单的说就是把一个N*1的向量归一化为（0，1）之间的值。
    # 由于其中采用指数运算，使得向量中数值较大的量特征更加明显。
    # 在本例中， softmax 的输出向量的含义就是结果属于各个类的概率。
    # 例如对 A = [1.0,2.0,3.0,4.0,5.0,6.0] 执行 softmax 的结果为：
    #   [ 0.00426978 0.01160646 0.03154963 0.08576079 0.23312201 0.63369131]
    pred = tf.nn.softmax(tf.matmul(x, W) + b)

    # 使用交叉熵最小化损失函数
    # tf.reduce_mean(input_tensor,              输入的待降维的tensor
    #                 axis=None,                指定的轴，如果不指定，则计算所有元素的均值
    #                 keep_dims=False,          是否降维度，设置为True，输出的结果保持输入tensor的形状，
    #                                                      设置为False，输出结果会降低维度
    #                 name=None,
    #                 reduction_indices=None)   在以前版本中用来指定轴，已弃用
    # 该函数函数用于计算张量tensor沿着指定的数轴（tensor的某一维度）上的的平均值。
    # 主要用作降维或者计算tensor（图像）的平均值。
    cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=1))

    # 采用梯度下降来进行优化
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    # 初始化变量
    init = tf.global_variables_initializer()

    # 开始训练
    with tf.Session() as sess:
        # 变量初始化
        sess.run(init)
        for epoch in range(training_epochs):
            avg_cost = 0.
            # 计算需要训练几批
            total_batch = int(mnist.train.num_examples / batch_size)
            # 循环遍历所有批次
            for i in range(total_batch):
                # mnist.train.next_batch(
                # batch_size,           本次所需的数据大小
                # fake_data=False,      是否由假数据
                # shuffle=True)         每次从头开始提供训练数据时，是否打乱数据
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                # 利用训练数据进行训练
                # 注意：以下两个语句 [此处期待一个更加简洁明了的解释]
                # (1)sess.run([a,b])
                # (2)sess.run(a) sess.run(b)
                # 这两个语句初看时没有任何区别，
                # 但是如果a,b函数恰好是读取example_batch和label_batch这种需要使用到数据批次输入输出函数时
                # 例如(tf.train.shuffle_batch.tf.reader.read).
                # (1)式只会调用一次输入数据函数，则得到的example_batch和label_batch来自同一批次。
                # (2)式会单独调用两次输入数据函数，则得到的example_batch来自上一批次而label_batch来自下一批次。
                # 这个需要十分注意，因为如果我们想要实时打印出label_batch和inference(example_batch)时，
                # 即将输入数据的标签和经过模型预测推断的结果进行比较时.如果我们使用(2)中的写法，
                # 则label_batch和inference(example_batch)并不是来自与同一批次数据。
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                              y: batch_ys})
                # 计算平均损失，由于我们每步都要分批输入数据，因此我们需要计算平均的损失
                avg_cost += c / total_batch
            # 展示每步的日志
            if (epoch + 1) % display_step == 0:
                print("步数:", '%04d' % (epoch + 1), "损失=", "{:.9f}".format(avg_cost))
        print("训练完成!")

        # 测试模型
        # tf.argmax(vector, 1)：
        #   返回的是vector中的最大值的索引号，
        #   如果vector是一个向量，那就返回一个值，
        #   如果是一个矩阵，那就返回一个向量，这个向量的每一个维度都是相对应矩阵行的最大值元素的索引号。
        # tf.equal(A, B)：
        #   是对比这两个矩阵或者向量的相等的元素，
        #   如果是相等的那就返回True，反正返回False，
        #   返回的值的矩阵维度和A是一样的
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        # 计算3000个测试数据的准确度
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # tf.Tensor.eval(
        # feed_dict=None,   将张量对象映射到Feed值的字典。
        # session=None)     用于评估此张量的session。 如果没有，将使用默认session。
        print("准确性:", accuracy.eval({x: mnist.test.images[:3000], y: mnist.test.labels[:3000]}))


    return
