# -*- coding: utf-8 -*-
"""
@Time        : 2019/1/16 20:56
@Author      : Dong Hsueh
@Email       : frozenhsueh@gmail.com
@File        : Word2Vec.py
@Software    : PyCharm
@introduction: 实现了一种 Word2Vec 的算法
"""
import collections
import os
import random
import urllib
import zipfile

import numpy as np
import tensorflow as tf

def run():
    # 预定义训练的超参数
    learning_rate = 0.1     # 学习率
    batch_size = 128        # 输入数据批次的大小
    num_steps = 3000000     # 总计训练次数
    display_step = 10000    # 展示步间隔
    eval_step = 200000      # 评估步间隔

    # 评估参数
    eval_words = [b'five',b'of', b'going', b'hardware', b'american', b'britain']

    # Word2Vec 模型的参数
    embedding_size = 200            # 向量的维数
    max_vocabulary_size = 50000     # 词汇表中不同单词的总数
    min_occurrence = 10             # 删除至少n次不出现的所有单词
    skip_window = 3                 # 左右要考虑多少个单词
    num_skips = 2                   # 重复使用输入生成标签的次数
    num_sampled = 64                # 抽样的负面例子数量

    # 下载一小部分维基百科文章集
    url = 'http://mattmahoney.net/dc/text8.zip'
    data_path = 'text8.zip'
    if not os.path.exists(data_path):
        print("下载数据集... (这可能要花费一些时间)")
        filename, _ = urllib.request.urlretrieve(url, data_path)
        print("完成!")
    # 解压缩文本已处理完毕的数据集文件。
    with zipfile.ZipFile(data_path) as f:
        text_words = f.read(f.namelist()[0]).lower().split()
    # 构建字典并用UNK替换罕见的单词
    count = [('UNK', -1)]
    # 检索最常见的单词
    count.extend(collections.Counter(text_words).most_common(max_vocabulary_size - 1))
    # 删除少于'min_occurrence'次数的样本
    for i in range(len(count) - 1, -1, -1):
        if count[i][1] < min_occurrence:
            count.pop(i)
        else:
            # 该集合是有序的，因此在达到'min_occurrence'时就可以停止
            break
    # 计算词汇量大小
    vocabulary_size = len(count)
    # 为每个单词分配一个id
    word2id = dict()
    for i, (word, _) in enumerate(count):
        word2id[word] = i

    data = list()
    unk_count = 0
    for word in text_words:
        # 检索单词id，或者如果不在字典中则为其指定索引0（'UNK'）
        index = word2id.get(word, 0)
        if index == 0:
            unk_count += 1
        data.append(index)
    count[0] = ('UNK', unk_count)
    id2word = dict(zip(word2id.values(), word2id.keys()))

    print("单词数量:", len(text_words))
    print("独特的单词:", len(set(text_words)))
    print("词汇的数量:", vocabulary_size)
    print("最常见的词:", count[:10])

    data_index = 0

    # 为skip-gram模型生成训练批次
    def next_batch(batch_size, num_skips, skip_window,data_index):
        assert batch_size % num_skips == 0
        assert num_skips <= 2 * skip_window
        batch = np.ndarray(shape=(batch_size), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        # 获取窗口大小（左右两个字+当前一个）
        span = 2 * skip_window + 1
        buffer = collections.deque(maxlen=span)
        if data_index + span > len(data):
            data_index = 0
        buffer.extend(data[data_index:data_index + span])
        data_index += span
        for i in range(batch_size // num_skips):
            context_words = [w for w in range(span) if w != skip_window]
            words_to_use = random.sample(context_words, num_skips)
            for j, context_word in enumerate(words_to_use):
                batch[i * num_skips + j] = buffer[skip_window]
                labels[i * num_skips + j, 0] = buffer[context_word]
            if data_index == len(data):
                buffer.extend(data[0:span])
                data_index = span
            else:
                buffer.append(data[data_index])
                data_index += 1
        # 回溯一点，以避免在批处理结束时跳过单词
        data_index = (data_index + len(data) - span) % len(data)
        return batch, labels,data_index

    # 输入数据
    X = tf.placeholder(tf.int32, shape=[None])
    # 输入标签
    Y = tf.placeholder(tf.int32, shape=[None, 1])

    # 确保在CPU上分配以下ops＆var
    # （某些操作在GPU上不兼容）
    with tf.device('/cpu:0'):
        # 创建嵌入变量（每行代表一个单词嵌入向量）
        embedding = tf.Variable(tf.random_normal([vocabulary_size, embedding_size]))
        # 在X中查找每个样本的相应嵌入向量
        X_embed = tf.nn.embedding_lookup(embedding, X)

        # 构造NCE损失的变量
        nce_weights = tf.Variable(tf.random_normal([vocabulary_size, embedding_size]))
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    # 计算批次的平均NCE损失
    loss_op = tf.reduce_mean(
        tf.nn.nce_loss(weights=nce_weights,
                       biases=nce_biases,
                       labels=Y,
                       inputs=X_embed,
                       num_sampled=num_sampled,
                       num_classes=vocabulary_size))

    # 定义优化函数
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(loss_op)

    # 评估
    # 计算输入数据嵌入和每个嵌入向量之间的余弦相似度
    X_embed_norm = X_embed / tf.sqrt(tf.reduce_sum(tf.square(X_embed)))
    embedding_norm = embedding / tf.sqrt(tf.reduce_sum(tf.square(embedding), 1, keepdims=True))
    cosine_sim_op = tf.matmul(X_embed_norm, embedding_norm, transpose_b=True)

    # 初始化全部变量
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        # 测试数据
        x_test = np.array([word2id[w] for w in eval_words])

        average_loss = 0
        for step in range(1, num_steps + 1):
            # 获取一批新数据
            batch_x, batch_y,data_index = next_batch(batch_size, num_skips, skip_window,data_index)
            # 运行优化函数训练
            _, loss = sess.run([train_op, loss_op], feed_dict={X: batch_x, Y: batch_y})
            average_loss += loss

            if step % display_step == 0 or step == 1:
                if step > 1:
                    average_loss /= display_step
                print("步数 " + str(step) + ", 平均损失= " + \
                      "{:.4f}".format(average_loss))
                average_loss = 0

            # 评估
            if step % eval_step == 0 or step == 1:
                print("评估中...")
                sim = sess.run(cosine_sim_op, feed_dict={X: x_test})
                for i in range(len(eval_words)):
                    top_k = 8  # number of nearest neighbors
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                    log_str = '"%s" 含义最相近的是:' % eval_words[i]
                    for k in range(top_k):
                        log_str = '%s %s,' % (log_str, id2word[nearest[k]])
                    print(log_str)
    return
