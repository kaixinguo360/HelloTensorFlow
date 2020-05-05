# coding=utf-8
import tensorflow as tf
import numpy as np

# 首先，创建一个TensorFlow常量=>2
const = tf.constant(2.0, name='const')

# 创建placeholder
b = tf.placeholder(tf.float32, [None, 1], name='b')
# 创建TensorFlow变量c
c = tf.Variable(1.0, dtype=tf.float32, name='c')

# 创建operation
d = tf.add(b, c, name='d')
e = tf.add(c, const, name='e')
a = tf.multiply(d, e, name='a')

# 1. 定义init operation
init_op = tf.global_variables_initializer()

# session
with tf.Session() as sess:
    # 2. 运行init operation
    sess.run(init_op)
    # 计算
    a_out = sess.run(a, feed_dict={b: np.arange(0, 10)[:, np.newaxis]})
    print("Variable a is {}".format(a_out))

print("Hello,world!")