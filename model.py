import numpy as np
# import os
import tensorflow as tf
# from PIL import Image
# import utility as Utility
# import argparse


class InfoGAN():
    def __init__(self, noise_unit_num, img_channel, seed, base_channel, keep_prob):
        self.NOISE_UNIT_NUM = noise_unit_num  # 74
        self.IMG_CHANNEL = img_channel  # 1
        self.SEED = seed
        np.random.seed(seed=self.SEED)
        self.BASE_CHANNEL = base_channel  # 64
        self.KEEP_PROB = keep_prob
        self.CATEGORICAL_NUM = 10
        self.CONTINUOUS_NUM = 2

    def leaky_relu(self, x, alpha):
        return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

    def gaussian_noise(self, input, std):  # used at discriminator
        noise = tf.random_normal(shape=tf.shape(input), mean=0.0, stddev=std, dtype=tf.float32, seed=self.SEED)
        return input + noise

    def conv2d(self, input, in_channel, out_channel, k_size, stride, seed):
        w = tf.get_variable('w', [k_size, k_size, in_channel, out_channel],
                            initializer=tf.random_normal_initializer
                            (mean=0.0, stddev=0.02, seed=seed), dtype=tf.float32)
        b = tf.get_variable('b', [out_channel], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(input, w, strides=[1, stride, stride, 1], padding="SAME", name='conv') + b
        return conv

    def conv2d_transpose(self, input, in_channel, out_channel, k_size, stride, seed):
        w = tf.get_variable('w', [k_size, k_size, out_channel, in_channel],
                            initializer=tf.random_normal_initializer
                            (mean=0.0, stddev=0.02, seed=seed), dtype=tf.float32)
        b = tf.get_variable('b', [out_channel], initializer=tf.constant_initializer(0.0))
        out_shape = tf.stack(
            [tf.shape(input)[0], tf.shape(input)[1] * 2, tf.shape(input)[2] * 2, tf.constant(out_channel)])
        deconv = tf.nn.conv2d_transpose(input, w, output_shape=out_shape, strides=[1, stride, stride, 1],
                                        padding="SAME") + b
        return deconv

    def batch_norm(self, input):
        shape = input.get_shape().as_list()
        n_out = shape[-1]
        scale = tf.get_variable('scale', [n_out], initializer=tf.constant_initializer(1.0))
        beta = tf.get_variable('beta', [n_out], initializer=tf.constant_initializer(0.0))
        batch_mean, batch_var = tf.nn.moments(input, [0])
        bn = tf.nn.batch_normalization(input, batch_mean, batch_var, beta, scale, 0.0001, name='batch_norm')
        return bn

    def fully_connect(self, input, in_num, out_num, seed):
        w = tf.get_variable('w', [in_num, out_num], initializer=tf.random_normal_initializer
        (mean=0.0, stddev=0.02, seed=seed), dtype=tf.float32)
        b = tf.get_variable('b', [out_num], initializer=tf.constant_initializer(0.0))
        fc = tf.matmul(input, w, name='fc') + b
        return fc

    def generator(self, z, reuse=False, is_training=False):  # z is expected [n, 74] : 74 = cat10 + 2continuous + 62
        with tf.variable_scope('generator', reuse=reuse):
            with tf.variable_scope("layer1"):  # layer1 fc nx200 -> nx1024
                fc1 = self.fully_connect(z, self.NOISE_UNIT_NUM, 1024, self.SEED)
                bn1 = self.batch_norm(fc1)
                rl1 = tf.nn.relu(bn1)

            with tf.variable_scope("layer2"):  # layer2 fc nx1024 -> nx6272
                fc2 = self.fully_connect(rl1, 1024, 7 * 7 * self.BASE_CHANNEL * 2, self.SEED)
                bn2 = self.batch_norm(fc2)
                rl2 = tf.nn.relu(bn2)

            with tf.variable_scope("layer3"):  # layer3 deconv nx6272 -> nx7x7x128 -> nx14x14x64
                shape = tf.shape(rl2)
                reshape3 = tf.reshape(rl2, [shape[0], 7, 7, 128])
                deconv3 = self.conv2d_transpose(reshape3, self.BASE_CHANNEL * 2, self.BASE_CHANNEL, 4, 2, self.SEED)
                bn3 = self.batch_norm(deconv3)
                rl3 = tf.nn.relu(bn3)

            with tf.variable_scope("layer4"):  # layer3 deconv nx14x14x64 -> nx28x28x1
                deconv4 = self.conv2d_transpose(rl3, self.BASE_CHANNEL, self.IMG_CHANNEL, 4, 2, self.SEED)
                tanh4 = tf.nn.tanh(deconv4)
                print('tanh4.get_shape(), ', tanh4.get_shape())

        return tanh4


    def discriminator(self, x, reuse=False, is_training=True):  # z[n, 200], x[n, 28, 28, 1]
        with tf.variable_scope('discriminator', reuse=reuse):
            with tf.variable_scope("layer1"):  # layer x1 conv [n, 28, 28, 1] -> [n, 14, 14, 64]
                conv1 = self.conv2d(x, self.IMG_CHANNEL, self.BASE_CHANNEL, 4, 2, self.SEED)
                lr1 = self.leaky_relu(conv1, alpha=0.1)
                # drop1 = tf.layers.dropout(lr1, rate=1.0 - self.KEEP_PROB, name='dropout', training=is_training)

            with tf.variable_scope("layer2"):  # layer x2 conv [n, 14, 14, 64] -> [n, 7, 7, 128] -> [n, 3136]
                conv2 = self.conv2d(lr1, self.BASE_CHANNEL, self.BASE_CHANNEL * 2, 4, 2, self.SEED)
                bn2 = self.batch_norm(conv2)
                lr2 = self.leaky_relu(bn2, alpha=0.1)
                # drop2 = tf.layers.dropout(lr2, rate=1.0 - self.KEEP_PROB, name='dropout', training=is_training)
                shape2 = tf.shape(lr2)
                reshape2 = tf.reshape(lr2, [shape2[0], shape2[1] * shape2[2] * shape2[3]])

            with tf.variable_scope("layer3"):  # layer1 fc [n, 6272], [n, 1024]
                fc3 = self.fully_connect(reshape2, 6272, 1024, self.SEED)
                bn3 = self.batch_norm(fc3)
                self.lr3 = self.leaky_relu(bn3, alpha=0.1)
                # self.drop3 = tf.layers.dropout(lr3, rate=1.0 - self.KEEP_PROB, name='dropout', training=is_training)

            with tf.variable_scope("output_d"):
                fc_d = self.fully_connect(self.lr3, 1024, 2, self.SEED)
                self.output_d = tf.nn.softmax(fc_d)
            print('self.output_d.get_shape(), ', self.output_d.get_shape())
            return self.output_d, self.lr3


    def q_network(self, hidden, reuse=False):
        with tf.variable_scope('q_model', reuse=reuse):
            with tf.variable_scope("output_q1"):
                fcq = self.fully_connect(hidden, 1024, 128, self.SEED)
                bnq = self.batch_norm(fcq)
                lrq = self.leaky_relu(bnq, alpha=0.1)

            with tf.variable_scope("output_q2"):
                z_all = self.fully_connect(lrq, 128, self.CONTINUOUS_NUM * 2 + self.CATEGORICAL_NUM, self.SEED)
                self.conti_mean, self.conti_var, categ = tf.split(z_all, [self.CONTINUOUS_NUM, self.CONTINUOUS_NUM, 
                                                                          self.CATEGORICAL_NUM], axis=1)
                self.categ = tf.nn.softmax(categ)
        print('self.conti_mean.get_shape(), ', self.conti_mean.get_shape())
        print('self.conti_var.get_shape(), ', self.conti_var.get_shape())
        print('self.categ.get_shape(), ', self.categ.get_shape())
        return self.conti_mean, self.conti_var, self.categ

    
    # def gaussian_negative_log_likelihood(self, x, mean, var):
    #     s_diag = tf.linalg.diag(tf.exp(var))
    #     s_inv = tf.linalg.inv(s_diag)
    #     x_shape = tf.shape(x)
    #     x_mean_trans = tf.reshape(x-mean, [x_shape[0], tf.constant(1, dtype=tf.int32), x_shape[1]])
    #     x_mean = tf.reshape(x-mean, [x_shape[0], x_shape[1], 1])
    #     pi = tf.constant(np.pi, dtype=tf.float32)
    #     # nll = tf.reduce_mean(tf.log(tf.sqrt(tf.pow(2*pi, dim))) + 0.5 * tf.reduce_sum(tf.square(x - mean), axis=1))
    #     nll = tf.reduce_mean(tf.log(tf.pow((2*pi),2) * tf.linalg.det(s_diag) + 1e-6) +
    #                          0.5 * tf.reduce_sum(tf.matmul(tf.matmul(x_mean_trans, s_inv), x_mean), axis=(1,2)))
    #     print('nll.get_shape(), ', nll.get_shape())
    #     return nll

    def gaussian_negative_log_likelihood(self, x, mean, var):
        pi = tf.constant(np.pi, dtype=tf.float32)
        nll = tf.reduce_mean(tf.reduce_sum(0.5*tf.log(2*pi * tf.exp(var) + 1e-6) +
                             tf.div(tf.square(x - mean), (2 * tf.exp(var)+ 1e-6)), axis=1))
        print('nll.get_shape(), ', nll.get_shape())
        return nll



    def cross_entropy_loss_accuracy(self, prob, tar):
        crossEntropy_loss = - tf.reduce_mean(tf.multiply(tar, tf.log(tf.clip_by_value(prob, 1e-10, 1.0))),
                                        name='cross_entropy_loss')
        correct_prediction = tf.equal(tf.argmax(prob, 1), tf.argmax(tar, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print('crossEntropy_loss.get_shape(), ', crossEntropy_loss.get_shape())
        return crossEntropy_loss, accuracy
