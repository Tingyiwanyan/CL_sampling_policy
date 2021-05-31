import math
import copy
from itertools import groupby
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import tensorflow as tf
import numpy as np

class tradition_b():
    """
    Create tradition model
    """

    def __init__(self, read_d):
        self.read_d = read_d
        #self.train_data = read_d.train_set
        #self.test_data = read_d.test_set
        #self.length_train = len(self.train_data)
        #self.length_test = len(self.test_data)
        self.train_data_cohort = read_d.file_names_cohort[0:500]
        self.train_data_control = read_d.file_names_control[0:2000]
        self.test_data_cohort = read_d.file_names_cohort[500:700]
        self.test_data_control = read_d.file_names_control[2000:3000]
        self.batch_size = 32
        self.vital_length = 8
        self.lab_length = 19
        self.blood_length = 27
        self.epoch = 2
        self.gamma = 2
        self.tau = 1
        self.lr = LogisticRegression(random_state=0)
        self.rf = RandomForestClassifier(max_depth=100,random_state=0)

    def aquire_batch_data_cohort(self, starting_index, data_set,length):
        self.one_batch_data_cohort = np.zeros((length,self.vital_length+self.lab_length+self.blood_length))#+self.static_length))
        self.one_batch_logit_cohort = list(np.ones(length))
        self.one_batch_logit_dp = np.zeros((length,1))
        for i in range(length):
            name = data_set[starting_index+i]
            self.read_d.return_data_dynamic_cohort(name)
            one_data = self.read_d.one_data_tensor
            #one_data[one_data==0]=np.nan
            #one_data = np.nan_to_num(np.nanmean(one_data,0))
            one_data = np.mean(one_data,0)
            self.one_batch_data_cohort[i,:] = one_data
            #self.one_batch_data[i,self.vital_length+self.lab_length:] = self.read_d.one_data_tensor_static
            #self.one_batch_logit[i] = self.read_d.logit_label
            #self.one_batch_logit_dp[i,0] = self.read_d.logit_label

    def aquire_batch_data_control(self, starting_index, data_set,length):
        self.one_batch_data_control = np.zeros((length,self.vital_length+self.lab_length+self.blood_length))#+self.static_length))
        self.one_batch_logit_control = list(np.zeros(length))
        self.one_batch_logit_dp = np.zeros((length,1))
        for i in range(length):
            name = data_set[starting_index+i]
            self.read_d.return_data_dynamic_control(name)
            one_data = self.read_d.one_data_tensor
            #one_data[one_data==0]=np.nan
            #one_data = np.nan_to_num(np.nanmean(one_data,0))
            one_data = np.mean(one_data,0)
            self.one_batch_data_control[i,:] = one_data
            #self.one_batch_data[i,self.vital_length+self.lab_length:] = self.read_d.one_data_tensor_static
            #self.one_batch_logit[i] = self.read_d.logit_label
            #self.one_batch_logit_dp[i,0] = self.read_d.logit_label

    def aquire_batch_data_whole(self):
        self.one_batch_data_whole = np.concatenate((self.one_batch_data_cohort,self.one_batch_data_control),axis=0)
        self.one_batch_logit_whole = self.one_batch_logit_cohort + self.one_batch_logit_control

    def MLP_config(self):
        self.input_y_logit = tf.keras.backend.placeholder(
            [None, 1])
        self.input_x = tf.keras.backend.placeholder(
            [None, self.vital_length + self.lab_length])
        self.embedding = tf.compat.v1.layers.dense(inputs=self.input_x,
                                                   units=50,
                                                   kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                                                   activation=tf.nn.relu)
        self.logit_sig = tf.compat.v1.layers.dense(inputs=self.embedding,
                                                   units=1,
                                                   kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                                                   activation=tf.nn.sigmoid)

        bce = tf.keras.losses.BinaryCrossentropy()
        self.cross_entropy = bce(self.logit_sig, self.input_y_logit)
        self.train_step_ce = tf.compat.v1.train.AdamOptimizer(1e-3).minimize(self.cross_entropy)

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

    def MLP_train(self):
        #init_hidden_state = np.zeros(
            #(self.batch_size, 1 + self.positive_sample_size + self.negative_sample_size, self.latent_dim))
        self.iteration = np.int(np.floor(np.float(self.length_train) / self.batch_size))
        for i in range(self.epoch):
            for j in range(self.iteration):
                print(j)
                self.aquire_batch_data(j*self.batch_size, self.train_data, self.batch_size)
                self.err_ = self.sess.run([self.focal_loss, self.train_step_fl],
                                          feed_dict={self.input_x: self.one_batch_data,
                                                     #self.input_x_static:self.one_batch_data_static,
                                                     self.input_y_logit: self.one_batch_logit_dp})
                                                     #self.init_hiddenstate: init_hidden_state})
                print(self.err_[0])
            self.MLP_test()

    def MLP_test(self):
        #init_hidden_state = np.zeros(
            #(self.length_test, 1 + self.positive_sample_size + self.negative_sample_size, self.latent_dim))
        self.aquire_batch_data(0, self.test_data, self.length_test)
        # print(self.lr.score(self.one_batch_data,self.one_batch_logit))
        self.out_logit = self.sess.run(self.logit_sig, feed_dict={self.input_x: self.one_batch_data})
                                                                  #self.init_hiddenstate: init_hidden_state})
                                                                  #self.input_x_static: self.one_batch_data_static})
        print(roc_auc_score(self.one_batch_logit, self.out_logit))



    def logistic_regression(self):
        #self.iteration = np.int(np.floor(np.float(self.length_train) / self.batch_size))
        #for i in range(self.epoch):
            #for j in range(self.iteration):
                #print(j)
        self.aquire_batch_data_cohort(0,self.train_data_cohort,len(self.train_data_cohort))
        self.aquire_batch_data_control(0,self.train_data_control,len(self.train_data_control))
        self.aquire_batch_data_whole()
        self.lr.fit(self.one_batch_data_whole,self.one_batch_logit_whole)
                #print(self.lr.score(self.one_batch_data,self.one_batch_logit))
                #print(roc_auc_score(self.one_batch_logit,self.lr.predict_proba(self.one_batch_data)[:,1]))

        self.test_logistic_regression()

    def test_logistic_regression(self):
        #self.aquire_batch_data(0,self.test_data,self.length_test)
        self.aquire_batch_data_cohort(0, self.test_data_cohort, len(self.test_data_cohort))
        self.aquire_batch_data_control(0, self.test_data_control, len(self.test_data_control))
        self.aquire_batch_data_whole()
        #print(self.lr.score(self.one_batch_data,self.one_batch_logit))
        print(roc_auc_score(self.one_batch_logit_whole, self.lr.predict_proba(self.one_batch_data_whole)[:,1]))


    def random_forest(self):
        self.aquire_batch_data(0, self.train_data, 3000)
        self.rf.fit(self.one_batch_data, self.one_batch_logit)
                # print(self.lr.score(self.one_batch_data,self.one_batch_logit))
                # print(roc_auc_score(self.one_batch_logit,self.lr.predict_proba(self.one_batch_data)[:,1]))

        self.test_random_forest()

    def test_random_forest(self):
        self.aquire_batch_data(0,self.test_data,self.length_test)
        #print(self.lr.score(self.one_batch_data,self.one_batch_logit))
        print(roc_auc_score(self.one_batch_logit, self.rf.predict(self.one_batch_data)))


