# -*- coding: utf-8 -*-
"""
@Time        : 2019/1/15 14:29
@Author      : Dong Hsueh
@Email       : frozenhsueh@gmail.com
@File        : BasicOperations.py
@Software    : PyCharm
@introduction: 本节主要介绍一些tensorflow的基础操作
"""
import tensorflow as tf

def run():

    # ------------------------------ PART ONE ------------------------------
    # tf.constant
    # 我们可以利用tf.constant来声明一个常量
    # 此方法的返回值表示 constant 的输出
    # 我们生成两个常量a和b，其中a = 2 b = 3
    a = tf.constant(2)
    b = tf.constant(3)

    # tf.Session
    # Session封装了Tensor计算的环境，可以用来执行运算并返回结果
    # 在接下来我们通过sess.run()方法来展示a,b的值以及a,b之间的基础运算
    # sess.run()的返回值就是运算结果
    # 我们采用python的 with ... as ... 来让代码更简洁，同时方便的处理异常与自动关闭
    with tf.Session() as sess:
        print("a: %i" % sess.run(a), "b: %i" % sess.run(b))
        print("常量相加的结果是: %i" % sess.run(a + b))
        print("常量相乘的结果是: %i" % sess.run(a * b))
    # ------------------------------ PART ONE END ------------------------------


    # ------------------------------ PART TWO ------------------------------
    # tf.placeholder
    # 大部分时候，我们的函数输入可能需要在运行的时候在输入具体的值。
    # tf.placeholder 提供的占位符机制很好的解决了这样一个问题
    # 这个函数的输入是指定常量的类型
    # 接下来我们声明两个类型为int的常量
    a = tf.placeholder(tf.int32)
    b = tf.placeholder(tf.int32)

    # *操作*
    # 有了常量，我们还要有能够供session执行的操作
    # 我们可以定义两个操作，加和乘，返回值是加与乘运算的结果
    # 其中tf.add() 与 + 的结果基本相同，返回的值是a+b的结果，类型与a相同
    # tf.multiply 与 * 相同，该方法实现为元素级别的相乘
    # 在计算精度上，二者并没有差别。
    # 需要注意的是，两个变量的数据类型必须相同
    add = tf.add(a, b)
    mul = tf.multiply(a, b)

    # feed_dict
    # 由于a,b的值是由占位符提供，我们需要在运行计算图时给出具体的值
    # sess.run()函数内通过feed_dict参数来输入一个dict数据结构，tensorflow通过索引在执行时给出a,b的具体值
    with tf.Session() as sess:
        print("常量相加的结果是: %i" % sess.run(add, feed_dict={a: 2, b: 3}))
        print("常量相乘的结果是: %i" % sess.run(mul, feed_dict={a: 2, b: 3}))
    # ------------------------------ PART TWO END ------------------------------


    # ------------------------------ PART THREE ------------------------------

    # 关于矩阵
    # 在更适应的情况下，我们需要考虑的是矩阵计算
    # 下面这个例子将声明一个 1 * 2 的矩阵与一个 2 * 1 的矩阵
    matrix1 = tf.constant([[3., 3.]])
    matrix2 = tf.constant([[2.], [2.]])

    # tf.matmul
    # 与 tf.multiply 不同，该方法实现的是矩阵相乘，其输入参数如下所示
    # tf.matmul
    # (a, b,                两个矩阵
    # transpose_a=False,    是否在乘法运算前转置 a 矩阵
    # transpose_b=False,    是否在乘法运算前转置 b 矩阵
    # adjoint_a=False,      是否在乘法运算前共轭和转置 a 矩阵
    # adjoint_b=False,      是否在乘法运算前共轭和转置 b 矩阵
    # a_is_sparse=False,    是否把 a 处理为稀疏矩阵
    # b_is_sparse=False,    是否把 b 处理为稀疏矩阵
    # name=None)
    # 返回值 product 是矩阵相乘的结果，在本例中是一个 1 * 1 的矩阵
    product = tf.matmul(matrix1, matrix2)

    # 调用 sess.run() 方法来执行计算得到结果
    # 通过执行我们发现，运算的结果的数据类型是 numpy.ndarray
    with tf.Session() as sess:
        result = sess.run(product)
        print("矩阵乘法的结果是: %i ；其数据类型为： %s" % (result,type(result)))
    # ------------------------------ PART THREE END ------------------------------

    return