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
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
    return
