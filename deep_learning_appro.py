import math
import copy
from itertools import groupby
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
import tensorflow as tf
import numpy as np


class dp_appro():
    """
    create deep learning model
    """
    def __init__(self, read_d):
        self.read_d = read_d
        self.train_data = read_d.train_set
        self.test_data = read_d.test_set
        self.length_train = len(self.train_data)
        self.length_test = len(self.test_data)
        self.time_sequence = self.read_d.time_sequence
        self.batch_size = 512
        self.vital_length = 9
        self.lab_length = 25
        self.latent_dim = 50
        self.static_length = 19
        self.epoch = 6
        self.gamma = 2
        self.tau = 1.5
        self.positive_sample_time = 4

    def deep_layers(self):
        """
        create deep learning architecture
        """
        self.input_y_logit = tf.keras.backend.placeholder(
            [None,1])
        self.input_x = tf.keras.backend.placeholder(
            [None, self.time_sequence, self.vital_length+self.lab_length])
        self.input_x_static = tf.keras.backend.placeholder(
            [None, self.static_length])
        self.embed_static = tf.compat.v1.layers.dense(inputs=self.input_x_static,
                                                   units=self.latent_dim,
                                                   kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                                                   activation=tf.nn.sigmoid)
        self.lstm_layer1 = tf.keras.layers.LSTM(self.latent_dim,return_sequences=True,return_state=True)
        self.sequence1,self.last_h1,self.last_c1 = self.lstm_layer1(self.input_x)
        self.embedding = tf.concat([self.last_h1,self.input_x_static],axis=1)
        self.logit_sig = tf.compat.v1.layers.dense(inputs=self.embedding,
                                                   units=1,
                                                   kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                                                   activation=tf.nn.sigmoid)

        self.Dense_patient_time = self.sequence1

    def self_contrast_t(self):
        """
        implement self time step contrastive learning
        """
        """
        positive inner product
        """
        self.positive_broad_time = tf.broadcast_to(self.x_origin_time,
                                                   [self.batch_size, self.time_sequence, self.positive_sample_size,
                                                    self.input_size])
        self.negative_broad_time = tf.broadcast_to(self.x_origin_time,
                                                   [self.batch_size, self.time_sequence, self.negative_sample_size,
                                                    self.input_size])

        self.positive_broad_norm_time = tf.math.l2_normalize(self.positive_broad_time, axis=3)
        self.positive_sample_norm_time = tf.math.l2_normalize(self.x_skip_contrast_time, axis=3)

        self.positive_dot_prod_time = tf.multiply(self.positive_broad_norm_time, self.positive_sample_norm_time)
        # self.positive_dot_prod_sum = tf.reduce_sum(tf.math.exp(tf.reduce_sum(self.positive_dot_prod, 2)),1)
        self.positive_dot_prod_sum_time = tf.math.exp(tf.reduce_sum(self.positive_dot_prod_time, 3) / self.tau)

        """
        negative inner product
        """
        self.negative_broad_norm_time = tf.math.l2_normalize(self.negative_broad_time, axis=3)
        self.negative_sample_norm_time = tf.math.l2_normalize(self.x_negative_contrast_time, axis=3)

        self.negative_dot_prod_time = tf.multiply(self.negative_broad_norm_time, self.negative_sample_norm_time)
        self.negative_dot_prod_sum_time = tf.reduce_sum(
            tf.reduce_sum(tf.math.exp(tf.reduce_sum(self.negative_dot_prod_time, 3) / self.tau), 2), 1)
        self.negative_dot_prod_sum_time = tf.expand_dims(self.negative_dot_prod_sum_time, 1)
        self.negative_dot_prod_sum_time = tf.expand_dims(self.negative_dot_prod_sum_time, 1)

        """
        Compute normalized probability and take log form
        """
        self.denominator_normalizer_time = tf.math.add(self.positive_dot_prod_sum_time, self.negative_dot_prod_sum_time)
        self.normalized_prob_log_time = tf.math.log(
            tf.math.divide(self.positive_dot_prod_sum_time, self.denominator_normalizer_time))
        self.normalized_prob_log_k_time = tf.reduce_sum(tf.reduce_sum(self.normalized_prob_log_time, 2), 1)
        self.log_normalized_prob_time = tf.math.negative(tf.reduce_mean(self.normalized_prob_log_k_time, 0))



    def config_model(self):
        self.deep_layers()
        bce = tf.keras.losses.BinaryCrossentropy()
        self.cross_entropy = bce(self.logit_sig, self.input_y_logit)
        self.train_step_ce = tf.compat.v1.train.AdamOptimizer(1e-3).minimize(self.cross_entropy)
        """
        focal loss
        """
        alpha = 0.25
        alpha_t = self.input_y_logit * alpha + (tf.ones_like(self.input_y_logit) - self.input_y_logit) * (1 - alpha)

        p_t = self.input_y_logit * self.logit_sig + (tf.ones_like(self.input_y_logit) - self.input_y_logit) * (
                tf.ones_like(self.input_y_logit) - self.logit_sig) + tf.keras.backend.epsilon()

        self.focal_loss_ = - alpha_t * tf.math.pow((tf.ones_like(self.input_y_logit) - p_t), self.gamma) * tf.math.log(
            p_t)
        self.focal_loss = tf.reduce_mean(self.focal_loss_)
        self.train_step_fl = tf.compat.v1.train.AdamOptimizer(1e-3).minimize(self.focal_loss)
        self.sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()


    def aquire_batch_data(self, starting_index, data_set,length):
        self.one_batch_data = np.zeros((length,self.time_sequence,self.vital_length+self.lab_length))
        self.one_batch_data_static = np.zeros((length,self.static_length))
        self.one_batch_logit = np.zeros((length,1))
        for i in range(length):
            name = data_set[starting_index+i]
            self.check_name = name
            self.read_d.return_data_dynamic(name)
            one_data = self.read_d.one_data_tensor
            #one_data[one_data==0]=np.nan
            #one_data = np.nan_to_num(np.nanmean(one_data,1))
            self.one_batch_data[i,:,:] = one_data
            self.one_batch_logit[i,0] = self.read_d.logit_label
            self.one_batch_data_static[i, :] = self.read_d.one_data_tensor_static

    def train(self):
        self.iteration = np.int(np.floor(np.float(self.length_train) / self.batch_size))
        for i in range(self.epoch):
            for j in range(self.iteration):
                print(j)
                self.aquire_batch_data(j*self.batch_size, self.train_data, self.batch_size)
                self.err_ = self.sess.run([self.focal_loss, self.train_step_fl],
                                          feed_dict={self.input_x: self.one_batch_data,
                                                     self.input_x_static:self.one_batch_data_static,
                                                     self.input_y_logit: self.one_batch_logit})
                print(self.err_[0])
            self.test()

    def test(self):
        self.aquire_batch_data(0, self.test_data, self.length_test)
        # print(self.lr.score(self.one_batch_data,self.one_batch_logit))
        self.out_logit = self.sess.run(self.logit_sig, feed_dict={self.input_x: self.one_batch_data,
                                                                  self.input_x_static: self.one_batch_data_static})
        print(roc_auc_score(self.one_batch_logit, self.out_logit))







