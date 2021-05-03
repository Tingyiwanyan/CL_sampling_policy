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
        self.epoch_pre = 2
        self.gamma = 2
        self.tau = 1
        self.positive_sample_size = 4
        self.negative_sample_size = 20
        self.item_size = self.vital_length + self.lab_length
        self.input_size_ = self.latent_dim

        """
        define LSTM variables
        """
        self.init_hiddenstate = tf.keras.backend.placeholder(
            [None, 1 + self.positive_sample_size + self.negative_sample_size, self.latent_dim])
        self.input_y_logit = tf.keras.backend.placeholder([None, 1])
        self.input_x = tf.keras.backend.placeholder(
            [None, self.time_sequence, 1 + self.positive_sample_size + self.negative_sample_size,
             self.vital_length+self.lab_length])

        # self.input_x_demo = tf.concat([self.input_x_demo_,self.input_x_com],2)
        self.init_forget_gate = tf.keras.initializers.he_normal(seed=None)
        self.init_info_gate = tf.keras.initializers.he_normal(seed=None)
        self.init_cell_state = tf.keras.initializers.he_normal(seed=None)
        self.init_output_gate = tf.keras.initializers.he_normal(seed=None)
        self.init_forget_gate_weight = tf.keras.initializers.he_normal(seed=None)
        self.init_info_gate_weight = tf.keras.initializers.he_normal(seed=None)
        self.init_cell_state_weight = tf.keras.initializers.he_normal(seed=None)
        self.weight_forget_gate = \
            tf.Variable(
                self.init_forget_gate(shape=(self.item_size + self.latent_dim, self.latent_dim)))
        self.weight_info_gate = \
            tf.Variable(self.init_info_gate(shape=(self.item_size + self.latent_dim, self.latent_dim)))
        self.weight_cell_state = \
            tf.Variable(self.init_cell_state(shape=(self.item_size + self.latent_dim, self.latent_dim)))
        self.weight_output_gate = \
            tf.Variable(
                self.init_output_gate(shape=(self.item_size + self.latent_dim, self.latent_dim)))
        self.bias_forget_gate = tf.Variable(self.init_forget_gate_weight(shape=(self.latent_dim,)))
        self.bias_info_gate = tf.Variable(self.init_info_gate_weight(shape=(self.latent_dim,)))
        self.bias_cell_state = tf.Variable(self.init_cell_state_weight(shape=(self.latent_dim,)))
        self.bias_output_gate = tf.Variable(self.init_output_gate(shape=(self.latent_dim,)))

    def lstm_cell(self):
        cell_state = []
        hidden_rep = []
        for i in range(self.time_sequence):
            x_input_cur = tf.gather(self.input_x, i, axis=1)
            if i == 0:
                concat_cur = tf.concat([self.init_hiddenstate, x_input_cur], 2)
            else:
                concat_cur = tf.concat([hidden_rep[i - 1], x_input_cur], 2)
            forget_cur = \
                tf.math.sigmoid(tf.math.add(tf.matmul(concat_cur, self.weight_forget_gate), self.bias_forget_gate))
            info_cur = \
                tf.math.sigmoid(tf.math.add(tf.matmul(concat_cur, self.weight_info_gate), self.bias_info_gate))
            cellstate_cur = \
                tf.math.tanh(tf.math.add(tf.matmul(concat_cur, self.weight_cell_state), self.bias_cell_state))
            info_cell_state = tf.multiply(info_cur, cellstate_cur)
            if not i == 0:
                forget_cell_state = tf.multiply(forget_cur, cell_state[i - 1])
                cellstate_cur = tf.math.add(forget_cell_state, info_cell_state)
            output_gate = \
                tf.nn.relu(tf.math.add(tf.matmul(concat_cur, self.weight_output_gate), self.bias_output_gate))
            hidden_current = tf.multiply(output_gate, cellstate_cur)
            cell_state.append(cellstate_cur)
            hidden_rep.append(hidden_current)

        self.hidden_last = hidden_rep[self.time_sequence - 1]
        for i in range(self.time_sequence):
            hidden_rep[i] = tf.expand_dims(hidden_rep[i], 1)
        self.hidden_rep = tf.concat(hidden_rep, 1)
        self.hidden_global_vital = tf.reduce_mean(self.hidden_rep, 1)
        self.check = concat_cur

    def build_dhgm_model(self):
        """
        Build dynamic HGM model
        """
        # self.Dense_patient = tf.expand_dims(self.hidden_last,1)
        #self.Dense_patient = self.hidden_last
        self.Dense_patient_time = self.hidden_rep
        self.Dense_patient_global_vital = self.hidden_global_vital
        #self.Dense_patient_global_whole = tf.concat([self.Dense_patient_global_vital,self.Dense_comor],axis=2)
        self.Dense_patient = self.hidden_last
        # self.Dense_patient = tf.math.l2_normalize(self.Dense_patient,axis=2)

    def get_latent_rep_hetero_time(self):
        idx_origin = tf.constant([0])
        self.x_origin_time = tf.gather(self.Dense_patient_time, idx_origin, axis=2)
        self.x_origin_ce_time = tf.squeeze(self.x_origin_time, [2])
        self.x_origin = tf.gather(self.Dense_patient, idx_origin, axis=1)
        self.x_origin_ce = tf.squeeze(self.x_origin, [1])
        # self.knn_sim_matrix = tf.matmul(self.x_origin_ce, self.x_origin_ce, transpose_b=True)
        # self.x_origin = self.hidden_last

        """
        item_idx_skip = tf.constant([i+1 for i in range(self.positive_lab_size)])
        self.x_skip_item = tf.gather(self.Dense_lab,item_idx_skip,axis=1)
        item_idx_negative = tf.constant([i+self.positive_lab_size+1 for i in range(self.negative_lab_size)])
        self.x_negative_item = tf.gather(self.Dense_lab,item_idx_negative,axis=1)
        self.x_skip = tf.concat([self.x_skip,self.x_skip_item],axis=1)
        self.x_negative = tf.concat([self.x_negative,self.x_negative_item],axis=1)
        """
        patient_idx_skip = tf.constant([i + 1 for i in range(self.positive_sample_size)])
        self.x_skip_patient_time = tf.gather(self.Dense_patient_time, patient_idx_skip, axis=2)
        patient_idx_negative = tf.constant([i + self.positive_sample_size + 1 for i in range(self.negative_sample_size)])
        self.x_negative_patient_time = tf.gather(self.Dense_patient_time, patient_idx_negative, axis=2)

        # self.process_patient_att()

        # self.x_skip = tf.concat([self.x_skip_mor, self.x_skip_patient], axis=1)
        # self.x_negative = tf.concat([self.x_negative_mor, self.x_negative_patient], axis=1)
        self.x_skip_contrast_time = self.x_skip_patient_time
        self.x_negative_contrast_time = self.x_negative_patient_time

    def get_latent_rep_hetero(self):
        """
        Prepare data for SGNS loss function
        """
        idx_origin = tf.constant([0])
        self.x_origin = tf.gather(self.Dense_patient, idx_origin, axis=1)
        self.x_origin_ce = tf.squeeze(self.x_origin, [1])
        # self.knn_sim_matrix = tf.matmul(self.x_origin_ce, self.x_origin_ce, transpose_b=True)
        # self.x_origin = self.hidden_last

        """
        item_idx_skip = tf.constant([i+1 for i in range(self.positive_lab_size)])
        self.x_skip_item = tf.gather(self.Dense_lab,item_idx_skip,axis=1)
        item_idx_negative = tf.constant([i+self.positive_lab_size+1 for i in range(self.negative_lab_size)])
        self.x_negative_item = tf.gather(self.Dense_lab,item_idx_negative,axis=1)
        self.x_skip = tf.concat([self.x_skip,self.x_skip_item],axis=1)
        self.x_negative = tf.concat([self.x_negative,self.x_negative_item],axis=1)
        """
        patient_idx_skip = tf.constant([i + 1 for i in range(self.positive_sample_size)])
        self.x_skip_patient = tf.gather(self.Dense_patient, patient_idx_skip, axis=1)
        patient_idx_negative = tf.constant([i + self.positive_sample_size + 1 for i in range(self.negative_sample_size)])
        self.x_negative_patient = tf.gather(self.Dense_patient, patient_idx_negative, axis=1)

        # self.process_patient_att()

        # self.x_skip = tf.concat([self.x_skip_mor, self.x_skip_patient], axis=1)
        # self.x_negative = tf.concat([self.x_negative_mor, self.x_negative_patient], axis=1)
        self.x_skip_contrast = self.x_skip_patient
        self.x_negative_contrast = self.x_negative_patient


    def contrastive_loss(self):
        """
        Implement Contrastive Loss
        """
        """
        positive inner product
        """
        self.positive_broad = tf.broadcast_to(self.x_origin,
                                              [self.batch_size, self.positive_sample_size, self.input_size_])
        self.negative_broad = tf.broadcast_to(self.x_origin,
                                              [self.batch_size, self.negative_sample_size, self.input_size_])

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
        implement time-wise contrastive learning
        """
        """
        positive inner product
        """
        self.positive_broad_time = tf.broadcast_to(self.x_origin_time,
                                                   [self.batch_size, self.time_sequence, self.positive_sample_size,
                                                    self.latent_dim])
        self.negative_broad_time = tf.broadcast_to(self.x_origin_time,
                                                   [self.batch_size, self.time_sequence, self.negative_sample_size,
                                                    self.latent_dim])

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
        self.negative_dot_prod_sum_time = tf.expand_dims(self.negative_dot_prod_sum_time,0)
        self.negative_dot_prod_sum_all = tf.reduce_sum(self.negative_dot_prod_sum_time,1)
        #self.negative_dot_prod_sum_all = tf.expand_dims(self.negative_dot_prod_sum_all,0)
        self.negative_dot_prod_sum_all = tf.expand_dims(self.negative_dot_prod_sum_all, 0)
        self.negative_dot_prod_sum_all = tf.expand_dims(self.negative_dot_prod_sum_all, 0)
        self.negative_dot_prod_sum_all = tf.broadcast_to(self.negative_dot_prod_sum_all,[self.batch_size,self.time_sequence,self.positive_sample_size])
        #self.negative_dot_prod_sum_time = tf.expand_dims(self.negative_dot_prod_sum_time, 1)
        #self.negative_dot_prod_sum_time = tf.expand_dims(self.negative_dot_prod_sum_time, 1)

        """
        Compute normalized probability and take log form
        """
        self.denominator_normalizer_time = tf.math.add(self.positive_dot_prod_sum_time, self.negative_dot_prod_sum_all)
        self.normalized_prob_log_time = tf.math.log(
            tf.math.divide(self.positive_dot_prod_sum_time, self.denominator_normalizer_time))
        self.normalized_prob_log_k_time = tf.reduce_sum(tf.reduce_sum(self.normalized_prob_log_time, 2), 1)
        self.log_normalized_prob_time = tf.math.negative(tf.reduce_mean(self.normalized_prob_log_k_time, 0))



    def config_model(self):
        self.lstm_cell()
        self.build_dhgm_model()
        #self.get_latent_rep_hetero_time()
        self.get_latent_rep_hetero()
        #self.contrastive_learning_time()
        self.contrastive_loss()
        bce = tf.keras.losses.BinaryCrossentropy()
        self.logit_sig = tf.compat.v1.layers.dense(inputs=self.x_origin_ce,
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
        self.sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()

    def get_pos_supervised_t(self,length):
        self.patient_pos_sample_tensor = \
            np.zeros((length, self.time_sequence, self.positive_sample_size, self.vital_length + self.lab_length))
        self.positive_label_index = np.where(self.one_batch_logit[:,0]==1)[0]
        self.negative_label_index = np.where(self.one_batch_logit[:,0]==0)[0]
        for i in range(length):
            if self.one_batch_logit[i,0] == 1:
                neighbor_length = self.positive_label_index.shape[0]
                index_neighbor = np.floor(np.random.uniform(0, neighbor_length, self.positive_sample_size)).astype(int)
                neighbor_index = self.positive_label_index[index_neighbor]
                self.patient_pos_sample_tensor[i,:,:,:] = np.swapaxes(self.one_batch_data[neighbor_index],0,1)
            if self.one_batch_logit[i,0] == 0:
                neighbor_length = self.negative_label_index.shape[0]
                index_neighbor = np.floor(np.random.uniform(0, neighbor_length, self.positive_sample_size)).astype(int)
                neighbor_index = self.negative_label_index[index_neighbor]
                self.patient_pos_sample_tensor[i,:,:,:] = np.swapaxes(self.one_batch_data[neighbor_index],0,1)


    def get_neg_supervised_t(self,length):
        self.patient_neg_sample_tensor = \
            np.zeros((length, self.time_sequence, self.negative_sample_size, self.vital_length + self.lab_length))
        self.positive_label_index = np.where(self.one_batch_logit[:, 0] == 1)[0]
        self.negative_label_index = np.where(self.one_batch_logit[:, 0] == 0)[0]
        for i in range(length):
            if self.one_batch_logit[i, 0] == 1:
                neighbor_length = self.negative_label_index.shape[0]
                index_neighbor = np.floor(np.random.uniform(0, neighbor_length, self.negative_sample_size)).astype(int)
                neighbor_index = self.negative_label_index[index_neighbor]
                self.patient_neg_sample_tensor[i, :, :, :] = np.swapaxes(self.one_batch_data[neighbor_index],0,1)
            if self.one_batch_logit[i, 0] == 0:
                neighbor_length = self.positive_label_index.shape[0]
                index_neighbor = np.floor(np.random.uniform(0, neighbor_length, self.negative_sample_size)).astype(int)
                neighbor_index = self.negative_label_index[index_neighbor]
                self.patient_neg_sample_tensor[i, :, :, :] = np.swapaxes(self.one_batch_data[neighbor_index],0,1)

    def get_pos_self_t(self,length):
        self.patient_pos_sample_tensor = \
            np.zeros((length,self.time_sequence, self.positive_sample_size, self.vital_length+self.lab_length))
        for i in range(self.time_sequence):
            if i < self.positive_sample_size/2:
                self.patient_pos_sample_tensor[:,i,:,:] = self.one_batch_data[:,i+1:i+1+self.positive_sample_size,:]
            if i > self.positive_sample_size/2 and i < self.time_sequence - (self.positive_sample_size/2):
                self.patient_pos_sample_tensor[:,i,:,:] = self.one_batch_data[:,np.r_[i-1-(self.positive_sample_size/2):i-1,i+1:i+1+(self.positive_sample_size/2)],:]
            if i > self.time_sequence-self.positive_sample_size/2:
                self.patient_pos_sample_tensor[:,i,:,:] = self.one_batch_data[:,i-1-self.positive_sample_size:i-1,:]

    def get_neg_self_t(self,length):
        self.patient_neg_sample_tensor = \
            np.zeros((length,self.time_sequence, self.negative_sample_size, self.vital_length+self.lab_length))
        neg_bank = \
            np.reshape(self.one_batch_data, [length * self.time_sequence, self.vital_length+self.lab_length])
        neg_bank = np.expand_dims(neg_bank,0)
        neg_bank = np.broadcast_to(neg_bank,[self.time_sequence,length*self.time_sequence,self.vital_length+self.lab_length])
        for i in range(length):
            if i<self.negative_sample_size:
                self.patient_neg_sample_tensor[i,:,:,:] = neg_bank[:,(i+1)*self.time_sequence:((i+1)*self.time_sequence+self.negative_sample_size),:]
            elif i>length-self.negative_sample_size:
                self.patient_neg_sample_tensor[i,:,:,:] = neg_bank[:,(i*self.time_sequence-self.negative_sample_size):i*self.time_sequence,:]
            else:
                self.patient_neg_sample_tensor[i,:,:,:] = neg_bank[:,
                                                          np.r_[(i*self.time_sequence-self.negative_sample_size/2):i*self.time_sequence,
                                                          (i+1)*self.time_sequence:((i+1)*self.time_sequence+self.negative_sample_size/2)],:]


    def aquire_batch_data(self, starting_index, data_set,length):
        self.one_batch_data = np.zeros((length,self.time_sequence,self.vital_length+self.lab_length))
        self.one_batch_data_static = np.zeros((length,self.static_length))
        #self.get_pos_self_t(length)
        #self.get_neg_self_t(length)
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
            #for j in range(self.time_sequence):
            self.one_batch_data_static[i, :] = self.read_d.one_data_tensor_static

        self.get_pos_supervised_t(length)
        self.get_neg_supervised_t(length)

        self.one_batch_data = np.expand_dims(self.one_batch_data,axis=2)

        self.train_one_data_pos_neg = np.concatenate((self.patient_pos_sample_tensor, self.patient_neg_sample_tensor),
                                              axis=2)
        self.one_batch_data = np.concatenate((self.one_batch_data,self.train_one_data_pos_neg),axis=2)


    def pre_train(self):
        init_hidden_state = np.zeros(
            (self.batch_size, 1 + self.positive_sample_size + self.negative_sample_size, self.latent_dim))
        self.iteration = np.int(np.floor(np.float(self.length_train) / self.batch_size))
        for i in range(self.epoch_pre):
            for j in range(self.iteration):
                print(j)
                self.aquire_batch_data(j*self.batch_size, self.train_data, self.batch_size)
                self.err_ = self.sess.run([self.log_normalized_prob, self.train_step_cl,self.positive_check_prod,self.negative_check_prod],
                                          feed_dict={self.input_x: self.one_batch_data,
                                                     #self.input_x_static:self.one_batch_data_static,
                                                     self.input_y_logit: self.one_batch_logit,
                                                     self.init_hiddenstate: init_hidden_state})
                print(self.err_[0])


    def train(self):
        init_hidden_state = np.zeros(
            (self.batch_size, 1 + self.positive_sample_size + self.negative_sample_size, self.latent_dim))
        self.iteration = np.int(np.floor(np.float(self.length_train) / self.batch_size))
        for i in range(self.epoch):
            for j in range(self.iteration):
                print(j)
                self.aquire_batch_data(j*self.batch_size, self.train_data, self.batch_size)
                self.err_ = self.sess.run([self.focal_loss, self.train_step_fl],
                                          feed_dict={self.input_x: self.one_batch_data,
                                                     #self.input_x_static:self.one_batch_data_static,
                                                     self.input_y_logit: self.one_batch_logit,
                                                     self.init_hiddenstate: init_hidden_state})
                print(self.err_[0])
            self.test()

    def test(self):
        init_hidden_state = np.zeros(
            (self.length_test, 1 + self.positive_sample_size + self.negative_sample_size, self.latent_dim))
        self.aquire_batch_data(0, self.test_data, self.length_test)
        # print(self.lr.score(self.one_batch_data,self.one_batch_logit))
        self.out_logit = self.sess.run(self.logit_sig, feed_dict={self.input_x: self.one_batch_data,
                                                                  self.init_hiddenstate: init_hidden_state})
                                                                  #self.input_x_static: self.one_batch_data_static})
        print(roc_auc_score(self.one_batch_logit, self.out_logit))







