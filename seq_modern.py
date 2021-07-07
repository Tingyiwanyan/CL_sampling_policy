import math
import copy
from itertools import groupby
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.neighbors import NearestNeighbors
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats
import tensorflow as tf
import numpy as np
from sklearn.utils import resample
import random


class seq_modern():
    """
    create deep learning model
    """
    def __init__(self, read_d):
        self.read_d = read_d
        self.train_data_cohort = read_d.file_names_cohort[0:500]
        self.train_data_control = read_d.file_names_control[0:3000]
        self.test_data_cohort = read_d.file_names_cohort[500:700]
        self.test_data_control = read_d.file_names_control[3000:4000]
        self.train_data_cohort_mem = read_d.file_names_cohort[0:500]
        self.train_data_control_mem = read_d.file_names_control[0:3000]
        self.val_data_cohort = read_d.file_names_cohort[700:750]
        self.val_data_control = read_d.file_names_control[4000:4600]
        #self.train_data_cohort = self.train_data_cohort_mem
        #self.train_data_control = self.train_data_control_mem
        self.train_length_cohort_mem = len(self.train_data_cohort_mem)
        self.train_length_control_mem = len(self.train_data_control_mem)
        self.train_length_cohort = len(self.train_data_cohort)
        self.train_length_control = len(self.train_data_control)
       # self.train_length_cohort = self.train_length_cohort_mem
       # self.train_length_control = self.train_length_control_mem
        self.test_length_cohort = len(self.test_data_cohort)
        self.test_length_control = len(self.test_data_control)
        self.val_length_cohort = len(self.val_data_cohort)
        self.val_length_control = len(self.val_data_control)
        self.length_train = self.train_length_cohort+self.train_length_control
        self.batch_size = 64
        self.vital_length = 8
        self.lab_length = 19
        self.blood_length = 27
        self.epoch = 3
        self.epoch_pre = 2
        self.gamma = 2
        self.tau = 1
        self.latent_dim = 100
        self.layer2_dim = 50
        self.layer3_dim = 32
        self.final_dim = self.layer2_dim
        self.boost_iteration = 10
        self.time_sequence = self.read_d.time_sequence
        self.positive_sample_size = 5
        self.positive_sample_size_self = 2
        self.negative_sample_size = 15
        self.train_data_all = self.train_data_cohort + self.train_data_control
        self.logit = np.zeros(self.train_length_cohort+self.train_length_control)
        self.logit[0:self.train_length_cohort] = 1
        self.test_data_all = self.test_data_cohort + self.test_data_control
        self.val_data_all = self.val_data_cohort + self.val_data_control
        self.logit_test = np.zeros(self.test_length_cohort+self.test_length_control)
        self.logit_test[0:self.test_length_cohort] = 1
        self.logit_val = np.zeros(self.val_length_cohort + self.val_length_control)
        self.logit_val[0:self.val_length_cohort] = 1

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

    def construct_knn_attribute_cohort(self):
        """
        construct knn graph at every epoch using attribute information
        """
        print("Im here in constructing knn graph")

        self.knn_sim_matrix = np.zeros((self.train_length_cohort_mem,
                                        self.vital_length + self.lab_length+self.blood_length))
        self.knn_neighbor = {}

        for i in range(self.train_length_cohort_mem):
            name = self.train_data_cohort_mem[i]
            self.read_d.return_data_dynamic_cohort(name)
            one_data = self.read_d.one_data_tensor
            one_data = np.mean(one_data, 0)
            self.knn_sim_matrix[i, :] = one_data

        # self.norm_knn = np.expand_dims(np.linalg.norm(self.knn_sim_matrix, axis=1), 1)
        # self.knn_sim_matrix = self.knn_sim_matrix / self.norm_knn
        # self.knn_sim_score_matrix = np.matmul(self.knn_sim_matrix[:,0:8], self.knn_sim_matrix[:,0:8].T)
        self.knn_nbrs = NearestNeighbors(n_neighbors=self.train_length_cohort_mem, algorithm='auto',metric='euclidean').fit(
            self.knn_sim_matrix[:, :])
        distance, indices = self.knn_nbrs.kneighbors(self.knn_sim_matrix[:, :])
        for i in range(self.train_length_cohort_mem):
            # print(i)
            # vec = np.argsort(self.knn_sim_score_matrix[i, :])
            # vec = vec[::-1]
            self.vec = indices
            #center_patient_id = self.train_data_cohort_mem[i]
            center_patient_id = i
            center_patient_id_name = self.train_data_cohort_mem[i]
            index = 0
            for j in range(self.train_length_cohort_mem):
                if index == self.positive_sample_size:
                    break
                #compare_patient_id = self.train_data_cohort_mem[self.vec[i, j]]
                compare_patient_id = self.vec[i,j]
                if compare_patient_id == center_patient_id:
                    continue
                if center_patient_id_name not in self.knn_neighbor.keys():
                    self.knn_neighbor[center_patient_id_name] = {}
                    self.knn_neighbor[center_patient_id_name].setdefault('knn_neighbor', []).append(compare_patient_id)
                else:
                    self.knn_neighbor[center_patient_id_name].setdefault('knn_neighbor', []).append(compare_patient_id)

                index = index + 1

            index=0
            for j in range(self.train_length_cohort_mem):
                if index == self.negative_sample_size:
                    break
                #compare_patient_id = self.train_data_cohort_mem[self.vec[i,-1-j]]
                compare_patient_id = self.vec[i, -1-j]
                center_patient_id_name = self.train_data_cohort_mem[i]
                if compare_patient_id == center_patient_id:
                    continue
                if center_patient_id_name not in self.knn_neighbor.keys():
                    self.knn_neighbor[center_patient_id_name] = {}
                    self.knn_neighbor[center_patient_id_name].setdefault('neg_knn_neighbor', []).append(compare_patient_id)
                else:
                    self.knn_neighbor[center_patient_id_name].setdefault('neg_knn_neighbor', []).append(compare_patient_id)

                index = index + 1

    def construct_knn_attribute_control(self):
        """
        construct knn graph at every epoch using attribute information
        """
        print("Im here in constructing knn graph")

        self.knn_sim_matrix = np.zeros((self.train_length_control_mem,
                                        self.vital_length + self.lab_length+self.blood_length))
        self.knn_neighbor_control = {}

        for i in range(self.train_length_control_mem):
            name = self.train_data_control_mem[i]
            self.read_d.return_data_dynamic_control(name)
            one_data = self.read_d.one_data_tensor
            one_data = np.mean(one_data, 0)
            self.knn_sim_matrix[i, :] = one_data

        # self.norm_knn = np.expand_dims(np.linalg.norm(self.knn_sim_matrix, axis=1), 1)
        # self.knn_sim_matrix = self.knn_sim_matrix / self.norm_knn
        # self.knn_sim_score_matrix = np.matmul(self.knn_sim_matrix[:,0:8], self.knn_sim_matrix[:,0:8].T)
        self.knn_nbrs = NearestNeighbors(n_neighbors=self.train_length_control_mem, algorithm='auto',metric='euclidean').fit(
            self.knn_sim_matrix[:, :])
        distance, indices = self.knn_nbrs.kneighbors(self.knn_sim_matrix[:, :])
        for i in range(self.train_length_control_mem):
            # print(i)
            # vec = np.argsort(self.knn_sim_score_matrix[i, :])
            # vec = vec[::-1]
            self.vec = indices
            #center_patient_id = self.train_data_cohort_mem[i]
            center_patient_id = i
            center_patient_id_name = self.train_data_control_mem[i]
            index = 0
            for j in range(self.train_length_cohort_mem):
                if index == self.positive_sample_size:
                    break
                #compare_patient_id = self.train_data_cohort_mem[self.vec[i, j]]
                compare_patient_id = self.vec[i,j]
                if compare_patient_id == center_patient_id:
                    continue
                if center_patient_id_name not in self.knn_neighbor_control.keys():
                    self.knn_neighbor_control[center_patient_id_name] = {}
                    self.knn_neighbor_control[center_patient_id_name].setdefault('knn_neighbor', []).append(compare_patient_id)
                else:
                    self.knn_neighbor_control[center_patient_id_name].setdefault('knn_neighbor', []).append(compare_patient_id)

                index = index + 1

            index=0
            for j in range(self.train_length_control_mem):
                if index == self.negative_sample_size:
                    break
                #compare_patient_id = self.train_data_cohort_mem[self.vec[i,-1-j]]
                compare_patient_id = self.vec[i, -1-j]
                center_patient_id_name = self.train_data_control_mem[i]
                if compare_patient_id == center_patient_id:
                    continue
                if center_patient_id_name not in self.knn_neighbor_control.keys():
                    self.knn_neighbor_control[center_patient_id_name] = {}
                    self.knn_neighbor_control[center_patient_id_name].setdefault('neg_knn_neighbor', []).append(compare_patient_id)
                else:
                    self.knn_neighbor_control[center_patient_id_name].setdefault('neg_knn_neighbor', []).append(compare_patient_id)

                index = index + 1


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



    def LSTM_layers_stack(self, whole_seq_input, output_dim):
        lstm = tf.keras.layers.LSTM(output_dim, return_sequences=True, return_state=True)

        #whole_seq_output,final_memory_state,final_carry_state = lstm(whole_seq_input_act)
        dense = tf.keras.layers.Dense(output_dim, activation=tf.nn.relu,
                                      kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                                      kernel_regularizer=tf.keras.regularizers.l1(0.01),
                                      activity_regularizer=tf.keras.regularizers.l2(0.01)
                                      )
        layer = tf.keras.layers.Dropout(.2, input_shape=(output_dim,))
        BN = tf.keras.layers.BatchNormalization()

        whole_seq_output_ = dense(whole_seq_input)
        whole_seq_output_bn = BN(whole_seq_output_)
        whole_seq_output_act = layer(whole_seq_output_bn)

        whole_seq_output, final_memory_state, final_carry_state = lstm(whole_seq_output_act)

        return whole_seq_output

    def reverse_lstm_layer(self):
        self.lstm = tf.keras.layers.LSTM(self.latent_dim, return_sequences=True, return_state=True)
        self.input_x_reverse = tf.keras.backend.placeholder(
            [None, self.time_sequence, self.vital_length + self.lab_length + self.blood_length])
        self.whole_seq_output, self.final_memory_state, self.final_carry_state = self.lstm(self.input_x)
        self.input_y_logit = tf.keras.backend.placeholder([None, 1])

    def config_model(self):
        #self.create_memory_bank()
        #self.construct_knn_attribute_cohort()
        #self.construct_knn_attribute_control()
        self.shuffle_train_data()
        self.LSTM_layers()
        self.reverse_lstm_layer()
        """
        LSTM stack layers
        """

        whole_seq_output = self.LSTM_layers_stack(self.whole_seq_output, self.layer2_dim)
        whole_seq_output_reverse = self.LSTM_layers_stack(self.whole_seq_output, self.layer2_dim)
        #whole_seq_output_reverse = tf.reverse(whole_seq_output_reverse_,1)
        whole_seq_output_concat = tf.concat([whole_seq_output,whole_seq_output_reverse],2)
        # whole_seq_output, whole_seq_output_pos, whole_seq_output_neg = \
        # self.LSTM_layers_stack(whole_seq_output1,
        # whole_seq_output_pos1, whole_seq_output_neg1, self.layer3_dim)

        bce = tf.keras.losses.BinaryCrossentropy()
        self.x_origin = whole_seq_output_concat[:,self.time_sequence-1,:]
        #self.x_origin = tf.reshape(whole_seq_output,[tf.shape(whole_seq_output)[0],self.time_sequence*self.layer2_dim])
        #self.x_origin = tf.reshape(whole_seq_output_concat,
                                  # [tf.shape(whole_seq_output_concat)[0], self.time_sequence * self.layer2_dim*2])
        self.whole_seq_output_final = whole_seq_output
        self.logit_sig = tf.compat.v1.layers.dense(inputs=self.x_origin,
                                                   units=1,
                                                   kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                                                   activation=tf.nn.sigmoid)
        self.cross_entropy = bce(self.logit_sig, self.input_y_logit)
        self.train_step_ce = tf.compat.v1.train.AdamOptimizer(1e-3).minimize(self.cross_entropy)
        #self.train_step_cl_attribute = tf.compat.v1.train.AdamOptimizer(1e-3).minimize(self.log_normalized_prob_self)
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

    def aquire_batch_data(self, starting_index, data_set,length,logit_input):
        self.one_batch_data = np.zeros((length,self.time_sequence,self.vital_length+self.lab_length+self.blood_length))
        self.one_batch_data_reverse = np.zeros((length,self.time_sequence,self.vital_length+self.lab_length+self.blood_length))
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

        for i in range(self.time_sequence):
            self.one_batch_data_reverse[:,i,:] = self.one_batch_data[:,self.time_sequence-1-i,:]


    def train(self):
        self.step = []
        self.acc = []
        self.iteration = np.int(np.floor(np.float(self.length_train) / self.batch_size))
        for i in range(self.epoch):
            print(i)
            for j in range(self.iteration):
                #print(j)
                self.aquire_batch_data(j*self.batch_size, self.shuffle_train, self.batch_size,self.shuffle_logit)
                self.err_ = self.sess.run([self.focal_loss, self.train_step_fl,self.logit_sig],
                                          feed_dict={self.input_x: self.one_batch_data,
                                                     self.input_x_reverse: self.one_batch_data_reverse,
                                                     self.input_y_logit: self.one_batch_logit_dp})
                                                     #self.input_x_pos:self.one_batch_data_pos,
                                                    # self.input_x_neg:self.one_batch_data_neg,
                                                     #self.input_x_neg_self:self.one_batch_data_neg_self})

                #print(self.err_[0])
                #auc = roc_auc_score(self.one_batch_logit, self.err_[2])
                    #if j % 5 == 0:
                        #auc = self.test()
                #print(auc)
                #self.step.append(j)
                #self.acc.append(auc)

                if j % 10 == 0:
                    print(j)
                    self.val()
                    self.acc.append(self.temp_auc)
            print("epoch")

        self.test()

    def test(self):
        sample_size_cohort = np.int(np.floor(len(self.test_data_cohort) * 4 / 5))
        sample_size_control = np.int(np.floor(len(self.test_data_control) * 4 / 5))
        auc = []
        auprc = []
        for i in range(self.boost_iteration):
            print(i)
            test_cohort = resample(self.test_data_cohort, n_samples=sample_size_cohort)
            test_control = resample(self.test_data_control, n_samples=sample_size_control)
            test_data = test_cohort + test_control
            logit_test = np.zeros(len(test_cohort) + len(test_control))
            logit_test[0:len(test_cohort)] = 1
            self.aquire_batch_data(0, test_data, len(test_data),logit_test)
        # print(self.lr.score(self.one_batch_data,self.one_batch_logit))
            self.out_logit = self.sess.run(self.logit_sig, feed_dict={self.input_x: self.one_batch_data,
                                                                      self.input_x_reverse: self.one_batch_data_reverse})
                                                                  #self.init_hiddenstate: init_hidden_state})
                                                                  #self.input_x_static: self.one_batch_data_static})
            auc.append(
                roc_auc_score(self.one_batch_logit, self.out_logit))
            auprc.append(average_precision_score(self.one_batch_logit, self.out_logit))
        print("auc")
        print(bs.bootstrap(np.array(auc), stat_func=bs_stats.mean))
        print("auprc")
        print(bs.bootstrap(np.array(auprc), stat_func=bs_stats.mean))
        #print(roc_auc_score(self.one_batch_logit, self.out_logit))
        #return roc_auc_score(self.one_batch_logit, self.out_logit)

    def test_whole(self):
        # print(self.lr.score(self.one_batch_data,self.one_batch_logit))
        self.aquire_batch_data(0, self.test_data_all, self.test_length_cohort + self.test_length_control, self.logit_test)
        self.out_logit = self.sess.run(self.logit_sig, feed_dict={self.input_x: self.one_batch_data,
                                                                  self.input_x_reverse: self.one_batch_data_reverse})
        print("auc")
        print(
            roc_auc_score(self.one_batch_logit, self.out_logit))
        print("auprc")
        print(average_precision_score(self.one_batch_logit,
                                      self.out_logit))
        #np.savetxt('xgb_prob.out', self.out_logit)


    def val(self):
        self.aquire_batch_data(0, self.val_data_all, self.val_length_cohort+self.val_length_control,self.logit_val)
        self.out_logit = self.sess.run(self.logit_sig, feed_dict={self.input_x: self.one_batch_data,
                                                                  self.input_x_reverse: self.one_batch_data_reverse})

        print(roc_auc_score(self.one_batch_logit, self.out_logit))
        self.temp_auc = roc_auc_score(self.one_batch_logit, self.out_logit)

