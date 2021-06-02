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

class seq_cl():
    """
    create deep learning model
    """
    def __init__(self, read_d):
        self.read_d = read_d
        self.train_data_cohort = read_d.file_names_cohort[0:1000]
        self.train_data_control = read_d.file_names_control[0:7000]
        self.test_data_cohort = read_d.file_names_cohort[1000:1400]
        self.test_data_control = read_d.file_names_control[7000:14000]
        self.train_data_cohort_mem = read_d.file_names_cohort[0:400]
        self.train_data_control_mem = read_d.file_names_control[0:2000]
        self.train_length_cohort_mem = len(self.train_data_cohort_mem)
        self.train_length_control_mem = len(self.train_data_control_mem)
        self.train_length_cohort = len(self.train_data_cohort)
        self.train_length_control = len(self.train_data_control)
        self.test_length_cohort = len(self.test_data_cohort)
        self.test_length_control = len(self.test_data_control)
        self.length_train = self.train_length_cohort+self.train_length_control
        self.batch_size = 64
        self.vital_length = 8
        self.lab_length = 19
        self.blood_length = 27
        self.epoch = 2
        self.gamma = 2
        self.tau = 1
        self.latent_dim = 100
        self.time_sequence = self.read_d.time_sequence
        self.positive_sample_size = 4
        self.negative_sample_size = 10
        self.train_data_all = self.train_data_cohort + self.train_data_control
        self.logit = np.zeros(self.train_length_cohort+self.train_length_control)
        self.logit[0:self.train_length_cohort] = 1
        self.test_data_all = self.test_data_cohort + self.test_data_control
        self.logit_test = np.zeros(self.test_length_cohort+self.test_length_control)
        self.logit_test[0:self.test_length_cohort] = 1

    def create_memory_bank(self):
        self.memory_bank_cohort = np.zeros((self.train_length_cohort_mem,self.time_sequence,
                                         self.vital_length + self.lab_length+self.blood_length))
        self.memory_bank_control = np.zeros((self.train_length_control_mem,self.time_sequence,
                                         self.vital_length + self.lab_length+self.blood_length))

        for i in range(self.train_length_cohort_mem):
            name = self.train_data_cohort_mem[i]
            self.read_d.return_data_dynamic_cohort(name)
            one_data = self.read_d.one_data_tensor
            self.memory_bank_cohort[i, :, :] = one_data

        for i in range(self.train_length_control_mem):
            name = self.train_data_control_mem[i]
            self.read_d.return_data_dynamic_control(name)
            one_data = self.read_d.one_data_tensor
            self.memory_bank_control[i, :, :] = one_data

    def shuffle_train_data(self):
        self.shuffle_num = np.array(range(self.train_length_cohort+self.train_length_control))
        np.random.shuffle(self.shuffle_num)
        self.shuffle_train = np.array(self.train_data_all)[self.shuffle_num]
        self.shuffle_logit = self.logit[self.shuffle_num]


    def LSTM_layers(self):
        self.lstm = tf.keras.layers.LSTM(self.latent_dim,return_sequences=True,return_state=True)
        self.input_x = tf.keras.backend.placeholder(
            [None, self.time_sequence,self.vital_length + self.lab_length+self.blood_length])
        self.whole_seq_output,self.final_memory_state,self.final_carry_state = self.lstm(self.input_x)
        self.input_y_logit = tf.keras.backend.placeholder([None, 1])

        """
        positive sample
        """
        self.input_x_pos = tf.keras.backend.placeholder(
            [self.batch_size*self.positive_sample_size, self.time_sequence,
             self.vital_length + self.lab_length+self.blood_length])
        self.whole_seq_output_pos, self.final_memory_state_pos, self.final_carry_state_pos = self.lstm(self.input_x_pos)
        self.whole_seq_out_pos_reshape = tf.reshape(self.whole_seq_output_pos,[self.batch_size,
                                                                               self.positive_sample_size,
                                                                               self.time_sequence,
                                                                               self.latent_dim])
        """
        negative sample
        """
        self.input_x_neg = tf.keras.backend.placeholder(
            [self.batch_size * self.negative_sample_size, self.time_sequence,
             self.vital_length + self.lab_length + self.blood_length])
        self.whole_seq_output_neg, self.final_memory_state_neg, self.final_carry_state_neg = self.lstm(self.input_x_neg)
        self.whole_seq_out_neg_reshape = tf.reshape(self.whole_seq_output_neg, [self.batch_size,
                                                                                self.negative_sample_size,
                                                                                self.time_sequence,
                                                                                self.latent_dim])

    def config_model(self):
        self.shuffle_train_data()
        self.LSTM_layers()
        bce = tf.keras.losses.BinaryCrossentropy()
        self.x_origin = self.whole_seq_output[:,self.time_sequence-1,:]
        self.x_skip_contrast = self.whole_seq_out_pos_reshape[:,:,self.time_sequence-1,:]
        self.x_negative_contrast = self.whole_seq_out_neg_reshape[:,:,self.time_sequence-1,:]
        self.contrastive_learning()
        self.logit_sig = tf.compat.v1.layers.dense(inputs=self.x_origin,
                                                   units=1,
                                                   kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                                                   activation=tf.nn.sigmoid)
        self.cross_entropy = bce(self.logit_sig, self.input_y_logit)
        self.train_step_ce = tf.compat.v1.train.AdamOptimizer(1e-3).minimize(self.cross_entropy)
        self.train_step_cl = tf.compat.v1.train.AdamOptimizer(1e-3).minimize(self.log_normalized_prob)
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
        self.train_step_combine_fl = tf.compat.v1.train.AdamOptimizer(1e-3).minimize(
            self.focal_loss + 0.8 * self.log_normalized_prob)
        self.sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()

    def contrastive_learning(self):
        """
         positive inner product
         """
        self.x_origin_cl = tf.expand_dims(self.x_origin,axis=1)
        self.positive_broad = tf.broadcast_to(self.x_origin_cl,
                                              [self.batch_size, self.positive_sample_size, self.latent_dim])
        self.negative_broad = tf.broadcast_to(self.x_origin_cl,
                                              [self.batch_size, self.negative_sample_size, self.latent_dim])

        self.positive_broad_norm = tf.math.l2_normalize(self.positive_broad, axis=2)
        self.positive_sample_norm = tf.math.l2_normalize(self.x_skip_contrast, axis=2)

        self.positive_dot_prod = tf.multiply(self.positive_broad_norm, self.positive_sample_norm)
        self.positive_check_prod = tf.reduce_sum(self.positive_dot_prod, 2)
        # self.positive_dot_prod_sum = tf.reduce_sum(tf.math.exp(tf.reduce_sum(self.positive_dot_prod, 2)),1)
        self.positive_dot_prod_sum = tf.math.exp(tf.reduce_sum(self.positive_dot_prod, 2) / self.tau)

        """
        negative inner product
        """
        self.negative_broad_norm = tf.math.l2_normalize(self.negative_broad, axis=2)
        self.negative_sample_norm = tf.math.l2_normalize(self.x_negative_contrast, axis=2)

        self.negative_dot_prod = tf.multiply(self.negative_broad_norm, self.negative_sample_norm)
        self.negative_check_prod = tf.reduce_sum(self.negative_dot_prod, 2)
        self.negative_dot_prod_sum = tf.reduce_sum(tf.math.exp(tf.reduce_sum(self.negative_dot_prod, 2) / self.tau), 1)
        self.negative_dot_prod_sum = tf.expand_dims(self.negative_dot_prod_sum, 1)

        """
        Compute normalized probability and take log form
        """
        self.denominator_normalizer = tf.math.add(self.positive_dot_prod_sum, self.negative_dot_prod_sum)
        self.normalized_prob_log = tf.math.log(tf.math.divide(self.positive_dot_prod_sum, self.denominator_normalizer))
        self.normalized_prob_log_k = tf.reduce_sum(self.normalized_prob_log, 1)
        self.log_normalized_prob = tf.math.negative(tf.reduce_mean(self.normalized_prob_log_k, 0))


    def contrastive_learning_time(self):
        """
        supervised time level contrastive learning
        """


    def aquire_batch_data(self, starting_index, data_set,length,logit_input):
        self.one_batch_data = np.zeros((length,self.time_sequence,self.vital_length+self.lab_length+self.blood_length))
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
            self.one_batch_data[i,:,:] = one_data

    def aquire_batch_data_cl(self,starting_index, data_set,length,logit_input):
        self.one_batch_data = np.zeros((length,self.time_sequence,self.vital_length+self.lab_length+self.blood_length))
        self.one_batch_data_pos = np.zeros((length*self.positive_sample_size, self.time_sequence,
             self.vital_length + self.lab_length+self.blood_length))
        self.one_batch_data_neg = np.zeros((length*self.negative_sample_size, self.time_sequence,
             self.vital_length + self.lab_length+self.blood_length))
        self.one_batch_logit = np.array(list(logit_input[starting_index:starting_index+length]))
        self.one_batch_logit_dp = np.zeros((length,1))
        self.one_batch_logit_dp[:,0] = self.one_batch_logit
        for i in range(length):
            name = data_set[starting_index+i]
            label = self.one_batch_logit[i]
            if self.one_batch_logit[i] == 1:
                self.read_d.return_data_dynamic_cohort(name)
            else:
                self.read_d.return_data_dynamic_control(name)
            one_data = self.read_d.one_data_tensor
            self.one_batch_data[i,:,:] = one_data
            self.aquire_pos_data_random(label)
            self.aquire_neg_data_random(label)
            self.one_batch_data_pos[i*self.positive_sample_size:(i+1)*self.positive_sample_size,:,:] = \
                self.patient_pos_sample_tensor
            self.one_batch_data_neg[i*self.negative_sample_size:(i+1)*self.negative_sample_size,:,:] = \
                self.patient_neg_sample_tensor

    def aquire_pos_data_random(self,label):
        #print("im in pos")
        self.patient_pos_sample_tensor = \
            np.zeros((self.positive_sample_size, self.time_sequence,
             self.vital_length + self.lab_length+self.blood_length))
        if label == 1:
            index_neighbor = \
                np.floor(np.random.uniform(0, self.train_length_cohort_mem, self.positive_sample_size)).astype(int)
            self.patient_pos_sample_tensor = self.memory_bank_cohort[index_neighbor,:,:]
            """
            neighbor_name = np.array(self.train_data_cohort)[index_neighbor]
            index = 0
            for i in neighbor_name:
                self.read_d.return_data_dynamic_cohort(i)
                one_data = self.read_d.one_data_tensor
                self.patient_pos_sample_tensor[index, :, :] = one_data
                index = index + 1
            """
        else:
            index_neighbor = \
                np.floor(np.random.uniform(0, self.train_length_control_mem, self.positive_sample_size)).astype(int)
            self.patient_pos_sample_tensor = self.memory_bank_control[index_neighbor, :, :]
            """
            neighbor_name = np.array(self.train_data_control)[index_neighbor]
            index = 0
            for i in neighbor_name:
                self.read_d.return_data_dynamic_control(i)
                one_data = self.read_d.one_data_tensor
                self.patient_pos_sample_tensor[index, :, :] = one_data
                index = index + 1
            """


    def aquire_neg_data_random(self,label):
        #print("im in neg")
        self.patient_neg_sample_tensor = \
            np.zeros((self.negative_sample_size, self.time_sequence,
                      self.vital_length + self.lab_length + self.blood_length))
        if label == 1:
            index_neighbor = \
                np.floor(np.random.uniform(0, self.train_length_control_mem, self.negative_sample_size)).astype(int)
            self.patient_neg_sample_tensor = self.memory_bank_control[index_neighbor,:,:]
            """
            neighbor_name = np.array(self.train_data_control)[index_neighbor]
            index = 0
            for i in neighbor_name:
                self.read_d.return_data_dynamic_control(i)
                one_data = self.read_d.one_data_tensor
                self.patient_neg_sample_tensor[index, :, :] = one_data
                index = index + 1
            """
        else:
            index_neighbor = \
                np.floor(np.random.uniform(0, self.train_length_cohort_mem, self.negative_sample_size)).astype(int)
            self.patient_neg_sample_tensor = self.memory_bank_cohort[index_neighbor,:,:]
            """
            neighbor_name = np.array(self.train_data_cohort)[index_neighbor]
            index = 0
            for i in neighbor_name:
                self.read_d.return_data_dynamic_cohort(i)
                one_data = self.read_d.one_data_tensor
                self.patient_neg_sample_tensor[index, :, :] = one_data
                index = index + 1
            """

    def pre_train(self):
        self.iteration = np.int(np.floor(np.float(self.length_train) / self.batch_size))
        for i in range(self.epoch):
            for j in range(self.iteration):
                print(j)
                self.aquire_batch_data_cl(j*self.batch_size, self.shuffle_train, self.batch_size,self.shuffle_logit)
                self.err_ = self.sess.run([self.log_normalized_prob, self.train_step_cl,self.logit_sig],
                                          feed_dict={self.input_x: self.one_batch_data,
                                                     self.input_y_logit: self.one_batch_logit_dp,
                                                     self.input_x_pos:self.one_batch_data_pos,
                                                     self.input_x_neg:self.one_batch_data_neg})

                print(self.err_[0])
                print(roc_auc_score(self.one_batch_logit, self.err_[2]))

    def train(self):
        self.iteration = np.int(np.floor(np.float(self.length_train) / self.batch_size))
        for i in range(self.epoch):
            for j in range(self.iteration):
                print(j)
                self.aquire_batch_data(j*self.batch_size, self.shuffle_train, self.batch_size,self.shuffle_logit)
                self.err_ = self.sess.run([self.focal_loss, self.train_step_fl,self.logit_sig],
                                          feed_dict={self.input_x: self.one_batch_data,
                                                     self.input_y_logit: self.one_batch_logit_dp})
                                                     #self.input_x_pos:self.one_batch_data_pos,
                                                     #self.input_x_neg:self.one_batch_data_neg})

                print(self.err_[0])
                print(roc_auc_score(self.one_batch_logit, self.err_[2]))
            #self.test()

    def test(self):
        self.aquire_batch_data(0, self.test_data_all, self.test_length_cohort+self.test_length_control,self.logit_test)
        # print(self.lr.score(self.one_batch_data,self.one_batch_logit))
        self.out_logit = self.sess.run(self.logit_sig, feed_dict={self.input_x: self.one_batch_data})
                                                                  #self.init_hiddenstate: init_hidden_state})
                                                                  #self.input_x_static: self.one_batch_data_static})
        print(roc_auc_score(self.one_batch_logit, self.out_logit))
