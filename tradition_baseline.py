import math
import copy
from itertools import groupby
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import tensorflow as tf
import numpy as np
from sklearn.utils import resample
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats
from sklearn import svm
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
import random

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
        self.train_data_control = read_d.file_names_control[0:3000]
        self.test_data_cohort = read_d.file_names_cohort[500:700]
        self.test_data_control = read_d.file_names_control[3000:4000]
        self.val_data_cohort = read_d.file_names_cohort[700:750]
        self.val_data_control = read_d.file_names_control[4000:4600]
        self.train_length_cohort = len(self.train_data_cohort)
        self.train_length_control = len(self.train_data_control)
        self.length_train = self.train_length_control + self.train_length_cohort
        self.train_data_all = self.train_data_cohort + self.train_data_control
        self.logit = np.zeros(self.train_length_cohort + self.train_length_control)
        self.logit[0:self.train_length_cohort] = 1
        self.batch_size = 64
        self.vital_length = 8
        self.lab_length = 19
        self.blood_length = 27
        self.boost_iteration = 10
        self.epoch = 3
        self.gamma = 2
        self.tau = 1
        self.lr = LogisticRegression(random_state=0)
        self.rf = RandomForestClassifier(max_depth=500,random_state=0)
        self.svm = svm.SVC(probability=True)
        self.xg_model = XGBClassifier()

    def aquire_batch_data(self, starting_index, data_set,length,logit_input):
        self.one_batch_data = np.zeros((length,self.vital_length+self.lab_length+self.blood_length))
        self.one_batch_logit = np.array(list(logit_input[starting_index:starting_index+length]))
        self.one_batch_logit_dp = np.zeros((length,1))
        self.one_batch_logit_dp[:,0] = self.one_batch_logit
        for i in range(length):
            name = data_set[starting_index+i]
            if self.one_batch_logit[i] == 1:
                self.read_d.return_data_dynamic_cohort(name)
            else:
                self.read_d.return_data_dynamic_control(name)
            one_data = self.read_d.one_data_tensor
            one_data = np.mean(one_data, 0)
            self.one_batch_data[i,:] = one_data

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

    def shuffle_train_data(self):
        self.shuffle_num = np.array(range(self.train_length_cohort+self.train_length_control))
        np.random.shuffle(self.shuffle_num)
        self.shuffle_train = np.array(self.train_data_all)[self.shuffle_num]
        self.shuffle_logit = self.logit[self.shuffle_num]

    def MLP_config(self):
        self.shuffle_train_data()
        self.input_y_logit = tf.keras.backend.placeholder(
            [None, 1])
        self.input_x = tf.keras.backend.placeholder(
            [None, self.vital_length + self.lab_length+self.blood_length])
        self.embedding = tf.compat.v1.layers.dense(inputs=self.input_x,
                                                   units=80,
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
        self.step = []
        self.acc = []
        self.iteration = np.int(np.floor(np.float(self.length_train) / self.batch_size))
        for i in range(self.epoch):
            for j in range(self.iteration):
                #print(j)
                self.aquire_batch_data(j*self.batch_size, self.shuffle_train, self.batch_size,self.shuffle_logit)
                self.err_ = self.sess.run([self.focal_loss, self.train_step_fl],
                                          feed_dict={self.input_x: self.one_batch_data,
                                                     #self.input_x_static:self.one_batch_data_static,
                                                     self.input_y_logit: self.one_batch_logit_dp})
                                                     #self.init_hiddenstate: init_hidden_state})
                #print(self.err_[0])
                if j%10 == 0:
                    print(j)
                    self.MLP_val()
                    self.acc.append(self.temp_auc)
            print("epoch")
            print(i)

        self.MLP_test()

    def MLP_val(self):
        self.aquire_batch_data_cohort(0, self.val_data_cohort, len(self.val_data_cohort))
        self.aquire_batch_data_control(0, self.val_data_control, len(self.val_data_control))
        self.aquire_batch_data_whole()
        self.out_logit = self.sess.run(self.logit_sig, feed_dict={self.input_x: self.one_batch_data_whole})

        print(roc_auc_score(self.one_batch_logit_whole, self.out_logit))
        self.temp_auc = roc_auc_score(self.one_batch_logit_whole, self.out_logit)

    def MLP_test(self):
        #init_hidden_state = np.zeros(
            #(self.length_test, 1 + self.positive_sample_size + self.negative_sample_size, self.latent_dim))
        #self.aquire_batch_data(0, self.test_data, self.length_test)
        # print(self.lr.score(self.one_batch_data,self.one_batch_logit))
        #self.out_logit = self.sess.run(self.logit_sig, feed_dict={self.input_x: self.one_batch_data_whole})
                                                                  #self.init_hiddenstate: init_hidden_state})
                                                                  #self.input_x_static: self.one_batch_data_static})
        #print(roc_auc_score(self.one_batch_logit, self.out_logit))
        self.acc_mlp = []
        sample_size_cohort = np.int(np.floor(len(self.test_data_cohort) * 4 / 5))
        sample_size_control = np.int(np.floor(len(self.test_data_control) * 4 / 5))
        auc = []
        auprc = []
        for i in range(self.boost_iteration):
            print(i)
            test_cohort = resample(self.test_data_cohort, n_samples=sample_size_cohort)
            test_control = resample(self.test_data_control, n_samples=sample_size_control)
            self.aquire_batch_data_cohort(0, test_cohort, len(test_cohort))
            self.aquire_batch_data_control(0, test_control, len(test_control))
            self.aquire_batch_data_whole()
            # print(self.lr.score(self.one_batch_data,self.one_batch_logit))
            self.out_logit = self.sess.run(self.logit_sig, feed_dict={self.input_x: self.one_batch_data_whole})
            auc.append(
                roc_auc_score(self.one_batch_logit_whole, self.out_logit))
            auprc.append(average_precision_score(self.one_batch_logit_whole, self.out_logit))
            self.acc_mlp.append(roc_auc_score(self.one_batch_logit_whole, self.out_logit))

        print("auc")
        print(bs.bootstrap(np.array(auc), stat_func=bs_stats.mean))
        print("auprc")
        print(bs.bootstrap(np.array(auprc), stat_func=bs_stats.mean))

    def MLP_test_whole(self):
        self.aquire_batch_data_cohort(0, self.test_data_cohort, len(self.test_data_cohort))
        self.aquire_batch_data_control(0, self.test_data_control, len(self.test_data_control))
        self.aquire_batch_data_whole()
        # print(self.lr.score(self.one_batch_data,self.one_batch_logit))
        self.out_logit = self.sess.run(self.logit_sig, feed_dict={self.input_x: self.one_batch_data_whole})
        print("auc")
        print(roc_auc_score(self.one_batch_logit_whole, self.out_logit))
        print("auprc")
        print(
            average_precision_score(self.one_batch_logit_whole, self.out_logit))
        np.savetxt('MLP_prob.out', self.out_logit)


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
        sample_size_cohort = np.int(np.floor(len(self.test_data_cohort) * 4 / 5))
        sample_size_control = np.int(np.floor(len(self.test_data_control) * 4 / 5))
        auc = []
        auprc = []
        for i in range(self.boost_iteration):
            test_cohort = resample(self.test_data_cohort, n_samples=sample_size_cohort)
            test_control = resample(self.test_data_control, n_samples=sample_size_control)
            self.aquire_batch_data_cohort(0,test_cohort, len(test_cohort))
            self.aquire_batch_data_control(0, test_control, len(test_control))
            self.aquire_batch_data_whole()
            # print(self.lr.score(self.one_batch_data,self.one_batch_logit))
            auc.append(roc_auc_score(self.one_batch_logit_whole, self.lr.predict_proba(self.one_batch_data_whole)[:,1]))
            auprc.append(average_precision_score(self.one_batch_logit_whole,
                                                 self.lr.predict_proba(self.one_batch_data_whole)[:,1]))

        print("auc")
        print(bs.bootstrap(np.array(auc), stat_func=bs_stats.mean))
        print("auprc")
        print(bs.bootstrap(np.array(auprc), stat_func=bs_stats.mean))

    def test_logistic_regression_whole(self):
        #self.aquire_batch_data(0,self.test_data,self.length_test)
        self.aquire_batch_data_cohort(0, self.test_data_cohort, len(self.test_data_cohort))
        self.aquire_batch_data_control(0, self.test_data_control, len(self.test_data_control))
        self.aquire_batch_data_whole()
        #print(self.lr.score(self.one_batch_data,self.one_batch_logit))
        print("auc")
        print(roc_auc_score(self.one_batch_logit_whole, self.lr.predict_proba(self.one_batch_data_whole)[:, 1]))
        print("auprc")
        print(average_precision_score(self.one_batch_logit_whole, self.lr.predict_proba(self.one_batch_data_whole)[:, 1]))
        np.savetxt('logistic_prob.out',self.lr.predict_proba(self.one_batch_data_whole)[:, 1])



    def random_forest(self):
        self.aquire_batch_data_cohort(0, self.train_data_cohort, len(self.train_data_cohort))
        self.aquire_batch_data_control(0, self.train_data_control, len(self.train_data_control))
        self.aquire_batch_data_whole()
        self.rf.fit(self.one_batch_data_whole, self.one_batch_logit_whole)
                # print(self.lr.score(self.one_batch_data,self.one_batch_logit))
                # print(roc_auc_score(self.one_batch_logit,self.lr.predict_proba(self.one_batch_data)[:,1]))

        self.test_random_forest()

    def test_random_forest(self):
        sample_size_cohort = np.int(np.floor(len(self.test_data_cohort) * 4 / 5))
        sample_size_control = np.int(np.floor(len(self.test_data_control) * 4 / 5))
        auc = []
        auprc = []
        #print(self.lr.score(self.one_batch_data,self.one_batch_logit))
        #print(roc_auc_score(self.one_batch_logit, self.rf.predict(self.one_batch_data)))
        for i in range(self.boost_iteration):
            test_cohort = resample(self.test_data_cohort, n_samples=sample_size_cohort)
            test_control = resample(self.test_data_control, n_samples=sample_size_control)
            self.aquire_batch_data_cohort(0,test_cohort, len(test_cohort))
            self.aquire_batch_data_control(0, test_control, len(test_control))
            self.aquire_batch_data_whole()
            # print(self.lr.score(self.one_batch_data,self.one_batch_logit))
            auc.append(roc_auc_score(self.one_batch_logit_whole, self.rf.predict_proba(self.one_batch_data_whole)[:,1]))
            auprc.append(average_precision_score(self.one_batch_logit_whole,
                                                 self.rf.predict_proba(self.one_batch_data_whole)[:,1]))

        print("auc")
        print(bs.bootstrap(np.array(auc), stat_func=bs_stats.mean))
        print("auprc")
        print(bs.bootstrap(np.array(auprc), stat_func=bs_stats.mean))

    def test_random_forest_whole(self):
        self.aquire_batch_data_cohort(0, self.test_data_cohort, len(self.test_data_cohort))
        self.aquire_batch_data_control(0, self.test_data_control, len(self.test_data_control))
        self.aquire_batch_data_whole()
        # print(self.lr.score(self.one_batch_data,self.one_batch_logit))
        print("auc")
        print(
            roc_auc_score(self.one_batch_logit_whole, self.rf.predict_proba(self.one_batch_data_whole)[:, 1]))
        print("auprc")
        print(average_precision_score(self.one_batch_logit_whole,
                                      self.rf.predict_proba(self.one_batch_data_whole)[:, 1]))
        np.savetxt('rf_prob.out', self.rf.predict_proba(self.one_batch_data_whole)[:, 1])


    def train_svm(self):
        self.aquire_batch_data_cohort(0, self.train_data_cohort, len(self.train_data_cohort))
        self.aquire_batch_data_control(0, self.train_data_control, len(self.train_data_control))
        self.aquire_batch_data_whole()
        self.svm.fit(self.one_batch_data_whole, self.one_batch_logit_whole)

        self.test_svm()

    def test_svm(self):
        sample_size_cohort = np.int(np.floor(len(self.test_data_cohort) * 4 / 5))
        sample_size_control = np.int(np.floor(len(self.test_data_control) * 4 / 5))
        auc = []
        auprc = []
        for i in range(self.boost_iteration):
            test_cohort = resample(self.test_data_cohort, n_samples=sample_size_cohort)
            test_control = resample(self.test_data_control, n_samples=sample_size_control)
            self.aquire_batch_data_cohort(0, test_cohort, len(test_cohort))
            self.aquire_batch_data_control(0, test_control, len(test_control))
            self.aquire_batch_data_whole()
            # print(self.lr.score(self.one_batch_data,self.one_batch_logit))
            auc.append(
                roc_auc_score(self.one_batch_logit_whole, self.svm.predict_proba(self.one_batch_data_whole)[:, 1]))
            auprc.append(average_precision_score(self.one_batch_logit_whole,
                                                 self.svm.predict_proba(self.one_batch_data_whole)[:, 1]))

        print("auc")
        print(bs.bootstrap(np.array(auc), stat_func=bs_stats.mean))
        print("auprc")
        print(bs.bootstrap(np.array(auprc), stat_func=bs_stats.mean))

    def test_whole_svm(self):
        self.aquire_batch_data_cohort(0, self.test_data_cohort, len(self.test_data_cohort))
        self.aquire_batch_data_control(0, self.test_data_control, len(self.test_data_control))
        self.aquire_batch_data_whole()
        # print(self.lr.score(self.one_batch_data,self.one_batch_logit))
        print("auc")
        print(
            roc_auc_score(self.one_batch_logit_whole, self.svm.predict_proba(self.one_batch_data_whole)[:, 1]))
        print("auprc")
        print(average_precision_score(self.one_batch_logit_whole,
                                      self.svm.predict_proba(self.one_batch_data_whole)[:, 1]))

        np.savetxt('svm_prob.out', self.svm.predict_proba(self.one_batch_data_whole)[:, 1])

    def train_xgb(self):
        self.aquire_batch_data_cohort(0, self.train_data_cohort, len(self.train_data_cohort))
        self.aquire_batch_data_control(0, self.train_data_control, len(self.train_data_control))
        self.aquire_batch_data_whole()
        self.xg_model.fit(self.one_batch_data_whole, self.one_batch_logit_whole)

        self.test_xgb()


    def test_xgb(self):
        sample_size_cohort = np.int(np.floor(len(self.test_data_cohort) * 4 / 5))
        sample_size_control = np.int(np.floor(len(self.test_data_control) * 4 / 5))
        auc = []
        auprc = []
        for i in range(self.boost_iteration):
            test_cohort = resample(self.test_data_cohort, n_samples=sample_size_cohort)
            test_control = resample(self.test_data_control, n_samples=sample_size_control)
            self.aquire_batch_data_cohort(0, test_cohort, len(test_cohort))
            self.aquire_batch_data_control(0, test_control, len(test_control))
            self.aquire_batch_data_whole()
            # print(self.lr.score(self.one_batch_data,self.one_batch_logit))
            auc.append(
                roc_auc_score(self.one_batch_logit_whole, self.xg_model.predict_proba(self.one_batch_data_whole)[:, 1]))
            auprc.append(average_precision_score(self.one_batch_logit_whole,
                                                 self.xg_model.predict_proba(self.one_batch_data_whole)[:, 1]))

        print("auc")
        print(bs.bootstrap(np.array(auc), stat_func=bs_stats.mean))
        print("auprc")
        print(bs.bootstrap(np.array(auprc), stat_func=bs_stats.mean))

    def test_whole_xgb(self):
        self.aquire_batch_data_cohort(0, self.test_data_cohort, len(self.test_data_cohort))
        self.aquire_batch_data_control(0, self.test_data_control, len(self.test_data_control))
        self.aquire_batch_data_whole()
        # print(self.lr.score(self.one_batch_data,self.one_batch_logit))
        print("auc")
        print(
            roc_auc_score(self.one_batch_logit_whole, self.xg_model.predict_proba(self.one_batch_data_whole)[:, 1]))
        print("auprc")
        print(average_precision_score(self.one_batch_logit_whole,
                                      self.xg_model.predict_proba(self.one_batch_data_whole)[:, 1]))
        np.savetxt('xgb_prob.out', self.xg_model.predict_proba(self.one_batch_data_whole)[:, 1])



