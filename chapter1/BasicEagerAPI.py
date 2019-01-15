# -*- coding: utf-8 -*-
"""
@Time        : 2019/1/15 15:32
@Author      : Dong Hsueh
@Email       : frozenhsueh@gmail.com
@File        : BasicEagerAPI.py
@Software    : PyCharm
@introduction: 本节主要介绍了 tensorflow 的动态图 -- eager API
               我们发现，在有些时候，我们也需要定义之后立马得到结果，而不是在之后的计算图中一起运算，因此我们需要 tensorflow 提供的动态图概念。
"""
import tensorflow as tf
import numpy as np

# from __future__ import absolute_import, division, print_function
# 由于原文的运行环境基于 Python2 ,因此原文中利用 __future__ 包将 python3中的一些特性导入进来，本文开发环境本身基于python3，因此不需要这个导入操作了

def run():
    # 设置为动态图模式，我们需要注意的是，如果不是在动态图模式下，本部分的代码会产生错误
    tf.enable_eager_execution()
    tfe = tf.contrib.eager

    # 定义常量张量，并直接 print 出来，而不需要在计算图中调用 run() 方法
    a = tf.constant(2)
    print("a = %i ; a的类型为： %s" % (a,type(a)))
    b = tf.constant(3)
    print("b = %i ; b的类型为： %s" % (b,type(b)))

    # 不需要 tf.Session() 也可以直接进行运算并得出结果
    c = a + b
    print("a + b = %i" % c)
    d = a * b
    print("a * b = %i" % d)

    # 我们可以发现，Tensors 和 Numpy 是完全兼容的
    # 我们定义一个 tensor 常量a，和一个 numpy 常量b
    # 我们发现，a 和 b是可以在一起计算的
    a = tf.constant([[2., 1.],
                     [1., 0.]], dtype=tf.float32)
    print("Tensor:\n a = %s" % a)
    b = np.array([[3., 0.],
                  [5., 1.]], dtype=np.float32)
    print("NumpyArray:\n b = %s" % b)
    c = a + b
    print("a + b = %s" % c)
    d = tf.matmul(a, b)
    print("a * b = %s" % d)

    # 我们也可以很方便的迭代矩阵 a
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            print(a[i][j])
    return