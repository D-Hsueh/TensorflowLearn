# -*- coding: utf-8 -*-
"""
@Time        : 2019/1/15 12:59
@Author      : Dong Hsueh
@Email       : frozenhsueh@gmail.com
@File        : HelloWorld.py
@Software    : PyCharm
@introduction: 作为第一个程序，自然需要实现tensorflow的hello world
               具体的的程序解释在之后给出
"""
import tensorflow as tf

def run():
    hello = tf.constant("Hello World")
    sess = tf.Session()
    result = sess.run(hello)
    print(result)