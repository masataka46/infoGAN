import numpy as np
import os
import tensorflow as tf
import utility as Utility
import argparse
from model import InfoGAN as Model
from make_datasets_MNIST import Make_datasets_MNIST as Make_datasets

def parser():
    parser = argparse.ArgumentParser(description='train LSGAN')
    parser.add_argument('--batch_size', '-b', type=int, default=100, help='Number of images in each mini-batch')
    parser.add_argument('--log_file_name', '-lf', type=str, default='log180926', help='log file name')
    parser.add_argument('--epoch', '-e', type=int, default=100, help='epoch')
    parser.add_argument('--file_name', '-fn', type=str, default='./mnist.npz', help='file name of data')
    parser.add_argument('--valid_span', '-vs', type=int, default=10, help='validation span')

    return parser.parse_args()

args = parser()


#global variants
BATCH_SIZE = args.batch_size
LOGFILE_NAME = args.log_file_name
EPOCH = args.epoch
FILE_NAME = args.file_name
IMG_WIDTH = 28
IMG_HEIGHT = 28
IMG_CHANNEL = 1
BASE_CHANNEL = 64
NOISE_ONLY_NUM = 62
CATEGORICAL_NUM = 10
CONTINUOUS_NUM = 2
NOISE_MEAN = 0.0
NOISE_STDDEV = 1.0
TEST_DATA_SAMPLE = 5 * 5
L2_NORM = 0.001
KEEP_PROB_RATE = 0.5
SEED = 1234
SCORE_ALPHA = 0.9 # using for cost function
VALID_SPAN = args.valid_span
np.random.seed(seed=SEED)
BOARD_DIR_NAME = './tensorboard/' + LOGFILE_NAME
OUT_IMG_DIR = './out_images_infoGAN' #output image file
out_model_dir = './out_models_infoGAN' #output model file
CYCLE_LAMBDA = 1.0

try:
    os.mkdir('log')
except:
    pass
try:
    os.mkdir('out_graph')
except:
    pass
try:
    os.mkdir(OUT_IMG_DIR)
except:
    pass
try:
    os.mkdir(out_model_dir)
except:
    pass
try:
    os.mkdir('./out_images_Debug') #for debug
except:
    pass

make_datasets = Make_datasets(FILE_NAME, IMG_WIDTH, IMG_HEIGHT, SEED)
model = Model(NOISE_ONLY_NUM+CATEGORICAL_NUM+CONTINUOUS_NUM, IMG_CHANNEL, SEED, BASE_CHANNEL, KEEP_PROB_RATE)

z_ = tf.placeholder(tf.float32, [None, NOISE_ONLY_NUM], name='z_') #noise to generator
c_categ_ = tf.placeholder(tf.float32, [None, CATEGORICAL_NUM], name='c_categ_') #categorical c
c_conti_ = tf.placeholder(tf.float32, [None, CONTINUOUS_NUM], name='c_conti_') #categorical c
x_ = tf.placeholder(tf.float32, [None, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL], name='x_') #image to classifier
# d_dis_f_ = tf.placeholder(tf.float32, [None, 2], name='d_dis_g_') #target of discriminator related to generator
# d_dis_r_ = tf.placeholder(tf.float32, [None, 2], name='d_dis_r_') #target of discriminator related to real image
d_dis_f_ = tf.placeholder(tf.float32, [None, 2], name='d_dis_g_') #target of discriminator related to generator
d_dis_r_ = tf.placeholder(tf.float32, [None, 2], name='d_dis_r_') #target of discriminator related to real image
is_training_ = tf.placeholder(tf.bool, name = 'is_training')

with tf.variable_scope('generator_model'):
    z_all = tf.concat([z_, c_categ_, c_conti_], axis=1)
    x_gen = model.generator(z_all, reuse=False, is_training=is_training_)

with tf.variable_scope('discriminator_model'):
    #stream around discriminator
    output_d_real, _ = model.discriminator(x_, reuse=False, is_training=is_training_) #real pair
    output_d_fake, hidden_D = model.discriminator(x_gen, reuse=True, is_training=is_training_) #real pair

with tf.variable_scope('q_network_model'):
    conti_fake_mean, conti_fake_var, categ_fake = model.q_network(hidden_D, reuse=False) #real pair


with tf.variable_scope("loss"):
    #adversarial loss
    loss_dis_real, accu_real = model.cross_entropy_loss_accuracy(output_d_real, d_dis_r_)  # loss related to real
    loss_dis_fake, accu_fake = model.cross_entropy_loss_accuracy(output_d_fake, d_dis_f_)  #loss related to fake
    #categorical loss
    loss_categ, accu_categ = model.cross_entropy_loss_accuracy(categ_fake, c_categ_)
    #continuous loss
    loss_conti = model.gaussian_negative_log_likelihood(c_conti_, conti_fake_mean, conti_fake_var)
    #total loss
    loss_dis_total = loss_dis_fake + loss_dis_real
    loss_gen_total = loss_dis_fake + loss_categ + loss_conti*0.1
    # loss_gen_total = loss_dis_fake


tf.summary.scalar('loss_dis_total', loss_dis_total)
# tf.summary.scalar('loss_gen_total', loss_gen_total)
merged = tf.summary.merge_all()

# t_vars = tf.trainable_variables()
gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator")
print("gen_vars, ", gen_vars)
dis_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator")
print("dis_vars, ", dis_vars)
q_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="q_network_model")
print("q_vars, ", q_vars)
gen_q_vars = gen_vars + q_vars
print("gen_q_vars, ", gen_q_vars)

with tf.name_scope("train"):
    train_dis = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(loss_dis_total, var_list=dis_vars
                                                                                , name='Adam_dis')
    train_gen = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5).minimize(loss_gen_total, var_list=gen_q_vars
                                                                                , name='Adam_gen')

sess = tf.Session()
sess.run(tf.global_variables_initializer())

summary_writer = tf.summary.FileWriter(BOARD_DIR_NAME, sess.graph)

log_list = []
log_list.append(['epoch', 'AUC'])
#training loop
for epoch in range(0, EPOCH):
    sum_loss_dis_f = np.float32(0)
    sum_loss_dis_r = np.float32(0)
    sum_loss_dis_f_gen = np.float32(0)
    sum_loss_dis_total = np.float32(0)
    sum_loss_gen_total = np.float32(0)
    sum_loss_categ = np.float32(0)
    sum_loss_conti = np.float32(0)
    sum_accu_real = np.float32(0)
    sum_accu_fake_d = np.float32(0)
    sum_accu_fake_g = np.float32(0)
    sum_accu_categ = np.float32(0)

    len_data = make_datasets.make_data_for_1_epoch()

    for i in range(0, len_data, BATCH_SIZE):
        img_batch, _ = make_datasets.get_data_for_1_batch(i, BATCH_SIZE)
        z = make_datasets.make_uniform_z(len(img_batch), NOISE_ONLY_NUM)
        # c_conti = make_datasets.make_random_s_with_norm(NOISE_MEAN, NOISE_STDDEV, len(img_batch), CONTINUOUS_NUM)
        c_conti = make_datasets.make_uniform_z(len(img_batch), CONTINUOUS_NUM)

        c_categ = make_datasets.make_categorical_s(len(img_batch), CATEGORICAL_NUM)
        tar_g_10 = make_datasets.make_target_1_0('1_0', len(img_batch)) #10 -> real
        tar_g_01 = make_datasets.make_target_1_0('0_1', len(img_batch)) #01 -> fake

        #train discriminator
        sess.run(train_dis, feed_dict={z_:z, c_categ_:c_categ, c_conti_:c_conti, x_: img_batch, d_dis_f_: tar_g_01, 
                                       d_dis_r_: tar_g_10, is_training_:True})

        #train generator
        sess.run(train_gen, feed_dict={z_:z, c_categ_:c_categ, c_conti_:c_conti, d_dis_f_: tar_g_10, 
                                       is_training_:True})

        # loss for discriminator
        loss_dis_total_, loss_dis_r_, loss_dis_f_d, accu_real_, accu_fake_d = sess.run([loss_dis_total, loss_dis_fake,
                                                                                loss_dis_real, accu_real, accu_fake],
                                feed_dict={z_:z, c_categ_:c_categ, c_conti_:c_conti, x_: img_batch, d_dis_f_: tar_g_01, 
                                       d_dis_r_: tar_g_10, is_training_:False})

        #loss for generator
        loss_gen_total_, loss_dis_f_g, loss_categ_, loss_conti_, accu_fake_g, accu_categ_ = sess.run([loss_gen_total, 
                                                loss_dis_fake, loss_categ, loss_conti, accu_fake, accu_categ], 
                        feed_dict={z_:z, c_categ_:c_categ, c_conti_:c_conti, d_dis_f_: tar_g_10, is_training_:False})

        #for tensorboard
        merged_ = sess.run(merged, feed_dict={z_:z, c_categ_:c_categ, c_conti_:c_conti, x_: img_batch, d_dis_f_: tar_g_01, 
                                       d_dis_r_: tar_g_10, is_training_:False})

        summary_writer.add_summary(merged_, epoch)

        sum_loss_dis_f += loss_dis_f_d * len(img_batch)
        sum_loss_dis_r += loss_dis_r_ * len(img_batch)
        sum_loss_dis_f_gen += loss_dis_f_g * len(img_batch)
        sum_loss_dis_total += loss_dis_total_ * len(img_batch)
        sum_loss_gen_total += loss_gen_total_ * len(img_batch)
        sum_loss_categ += loss_categ_ * len(img_batch)
        sum_loss_conti += loss_conti_ * len(img_batch)
        sum_accu_real += accu_real_ * len(img_batch)
        sum_accu_fake_d += accu_fake_d * len(img_batch)
        sum_accu_fake_g += accu_fake_g * len(img_batch)
        sum_accu_categ += accu_categ_ * len(img_batch)

    print("----------------------------------------------------------------------")
    print("epoch = {:}, Generator Total Loss = {:.4f}, Discriminator Total Loss = {:.4f}".format(
        epoch, sum_loss_gen_total / len_data, sum_loss_dis_total / len_data))
    print("Discriminator Real Loss = {:.4f}, Discriminator Fake Loss = {:.4f}".format(
        sum_loss_dis_r / len_data, sum_loss_dis_f / len_data))
    print("Generator adv Loss = {:.4f}, Categorical Loss = {:.4f}, Continuous Loss = {:.4f}".format(
        sum_loss_dis_f_gen / len_data, sum_loss_categ / len_data, sum_loss_conti / len_data))
    print("Discriminator Real Accuracy = {:.4f}, Discriminator Fake Accuracy = {:.4f}, Generator Fake Accuracy = {:.4f}".format(
        sum_accu_real / len_data, sum_accu_fake_d / len_data, sum_accu_fake_g / len_data))
    print("Category Accuracy = {:.4f}".format(sum_accu_categ / len_data))
    
    if epoch % VALID_SPAN == 0:
        len_valid_data = 10
        z = make_datasets.make_uniform_z(len_valid_data, NOISE_ONLY_NUM)
        c_conti = make_datasets.make_perm_s(len_valid_data, CONTINUOUS_NUM)
        # c_categ = make_datasets.make_categorical_s(len_valid_data, CATEGORICAL_NUM)
        c_categ = make_datasets.make_categorical_s_arange(len_valid_data, CATEGORICAL_NUM)
        x_gen_list=[]
        for num, c_conti_1 in enumerate(c_conti):
            c_conti_1_0 = c_conti_1[0].reshape(1, 1)
            c_conti_1_1 = c_conti[5, 1].reshape(1, 1)
            c_conti_1_same = np.concatenate((c_conti_1_0, c_conti_1_1), axis=1)
            c_conti_1_same = np.tile(c_conti_1_same, (len_valid_data, 1))
            x_gen_ = sess.run(x_gen, feed_dict={z_: z, c_categ_: c_categ, c_conti_: c_conti_1_same, is_training_: False})
            x_gen_list.append(x_gen_)
        for num, c_conti_1 in enumerate(c_conti):
            c_conti_1_0 = c_conti[5, 0].reshape(1, 1)
            c_conti_1_1 = c_conti_1[1].reshape(1, 1)
            c_conti_1_same = np.concatenate((c_conti_1_0, c_conti_1_1), axis=1)
            c_conti_1_same = np.tile(c_conti_1_same, (len_valid_data, 1))
            x_gen_ = sess.run(x_gen, feed_dict={z_: z, c_categ_: c_categ, c_conti_: c_conti_1_same, is_training_: False})
            x_gen_list.append(x_gen_)

        Utility.make_output_img(x_gen_list, epoch, LOGFILE_NAME, OUT_IMG_DIR)
