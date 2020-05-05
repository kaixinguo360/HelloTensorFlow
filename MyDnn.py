# coding=utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def get_weight(my_shape):
    return tf.Variable(tf.truncated_normal(my_shape, stddev=0.1))


def get_bias(my_shape):
    return tf.Variable(tf.constant(0.1, shape=my_shape))


def get_conv2d(my_input, my_filter):
    return tf.nn.conv2d(
        my_input,
        filter=my_filter,
        strides=[1, 1, 1, 1],
        padding="SAME"
    )


def get_conv2d_with_bias(my_input, my_shape):
    return tf.nn.relu(
        get_conv2d(my_input, get_weight(my_shape))+
        get_bias([my_shape[3]])
    )


def get_pool2d(my_input, key_size):
    return tf.nn.max_pool(
        my_input,
        ksize=[1, key_size, key_size, 1],
        strides=[1, key_size, key_size, 1],
        padding='SAME'
    )


def get_full_connect(my_input, my_shape):
    return tf.nn.relu(
        tf.matmul(my_input, get_weight(my_shape)) +
        get_bias([my_shape[1]])
    )


def get_output(my_input, my_shape):
    return tf.nn.softmax(
        tf.matmul(my_input, get_weight(my_shape)) +
        get_bias([my_shape[1]])
    )


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])
x_image = tf.reshape(x, [-1, 28, 28, 1])  # (? x 28 x 28 x 1)

# Conv Layer

conv1 = get_conv2d_with_bias(x_image, [5, 5, 1, 32])  # (? x 28 x 28 x 32)
pool1 = get_pool2d(conv1, 2)  # (? x 14 x 14 x 32)

conv2 = get_conv2d_with_bias(pool1, [5, 5, 32, 64])  # (? x 14 x 14 x 64)
pool2 = get_pool2d(conv2, 2)  # (? x 7 x 7 x 64)

# Full Connect Layer
fc1 = get_full_connect(
    tf.reshape(pool2, [-1, 7 * 7 * 64]),
    [7 * 7 * 64, 1024]
)

# Dropout Layer
keep_prob = tf.placeholder("float")
fc1_drop = tf.nn.dropout(fc1, keep_prob)

# Output Layer
y_conv = get_output(fc1_drop, [1024, 10])


## 评估 ##

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

with sess.as_default():
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    tf.summary.scalar('loss', cross_entropy)
    tf.summary.scalar('accuracy', accuracy)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter("Log/MyDnn", tf.get_default_graph())

    sess.run(tf.initialize_all_variables())
    for i in range(20000):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            summary, acc = sess.run(
                [merged, train_step],
                feed_dict={
                    x: batch[0],
                    y_: batch[1],
                    keep_prob: 0.5
                },
                options=run_options,
                run_metadata=run_metadata
            )
            train_accuracy = accuracy.eval(
                feed_dict={
                    x: batch[0],
                    y_: batch[1],
                    keep_prob: 1.0
                }
            )
            train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
            train_writer.add_summary(summary, i)
            print("step %d, training accuracy %g" % (i, train_accuracy))
            if train_accuracy > 0.99:
                break
        else:
            summary, _ = sess.run(
                [merged, train_step],
                feed_dict={
                    x: batch[0],
                    y_: batch[1],
                    keep_prob: 0.5
                }
            )
            train_writer.add_summary(summary, i)

    print("test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
    train_writer.close()
