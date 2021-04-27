import math
import copy
from itertools import groupby
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import tensorflow as tf
import numpy as np

class knn_cl():
    """
    Create dynamic HGM model
    """

    def __init__(self, read_d):
        # self.hetro_model = hetro_model
        self.read_d = read_d
        self.train_data = read_d.train_data
        self.test_data = read_d.test_data
        self.train_data_label = read_d.train_data_label
        self.test_data_label = read_d.test_data_label
        self.latent_dim = 50
        self.gamma = 2
        self.tau = 1.5
        self.softmax_weight_threshold = 0.1
        self.epoch = 2
        self.input_size = self.latent_dim
        self.item_size = 7
        self.time_sequence = 4
        self.batch_size = 64
        self.train_sepsis_group = read_d.train_sepsis_group
        self.train_non_sepsis_group = read_d.train_non_sepsis_group
        self.test_data = read_d.test_data
        self.size_for_pos_knn_construction = 100
        self.size_for_neg_knn_construction = 200
        self.positive_lab_size = 5
        self.negative_lab_size = 20
        self.positive_sample_size = self.positive_lab_size
        self.negative_sample_size = self.negative_lab_size
        self.knn_neighbor_numbers = self.positive_lab_size

        """
        define LSTM variables
        """
        self.init_hiddenstate = tf.keras.backend.placeholder(
            [None, 1 + self.positive_lab_size + self.negative_lab_size, self.latent_dim])
        self.input_y_logit = tf.keras.backend.placeholder([None, 1])
        self.input_x = tf.keras.backend.placeholder(
            [None, self.time_sequence, 1 + self.positive_lab_size + self.negative_lab_size, self.item_size])

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
        self.check = concat_cur


    def build_dhgm_model(self):
        """
        Build dynamic HGM model
        """
        # self.Dense_patient = tf.expand_dims(self.hidden_last,1)
        self.Dense_patient = self.hidden_last
        # self.Dense_patient = tf.math.l2_normalize(self.Dense_patient,axis=2)

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
        patient_idx_skip = tf.constant([i + 1 for i in range(self.positive_lab_size)])
        self.x_skip_patient = tf.gather(self.Dense_patient, patient_idx_skip, axis=1)
        patient_idx_negative = tf.constant([i + self.positive_lab_size + 1 for i in range(self.negative_lab_size)])
        self.x_negative_patient = tf.gather(self.Dense_patient, patient_idx_negative, axis=1)

        # self.process_patient_att()

        # self.x_skip = tf.concat([self.x_skip_mor, self.x_skip_patient], axis=1)
        # self.x_negative = tf.concat([self.x_negative_mor, self.x_negative_patient], axis=1)
        self.x_skip_contrast = self.x_skip_patient
        self.x_negative_contrast = self.x_negative_patient


    def assign_patient_value_physionet(self,center_node_index,label):
        one_sample = np.zeros((self.time_sequence,self.item_size))
        one_sample_single = np.zeros(self.item_size)
        name = self.read_d.file_path+center_node_index
        table = np.array(pd.read_table(name,sep="|"))
        if label == 1:
            time_sepsis = np.where(table[:,40]==1)[0][0]
        else:
            length_of_stay = table.shape[0]
            time_sepsis = np.int(np.floor(np.random.uniform(self.time_sequence-1, length_of_stay, 1)))
        for i in range(self.time_sequence):
            row_index = time_sepsis+i+1-self.time_sequence
            row_value = table[row_index,:]
            for j in range(self.item_size):
                mean = self.read_d.median_vital_signal[j]
                std = self.read_d.std_vital_signal[j]
                if np.isnan(row_value[j]):
                    row_value[j] = 0
                value = (row_value[j] - mean)/std
                one_sample_single[j] = value
            one_sample[i,:] = one_sample_single

        return one_sample


    def get_positive_patient(self, center_node_index, label, mode):
        self.patient_pos_sample_vital = np.zeros((self.time_sequence, self.positive_lab_size + 1, self.item_size))

        self.positive_patient_id_list = []
        if label == 0:
            neighbor_patient = self.read_d.train_non_sepsis_group
        else:
            neighbor_patient = self.read_d.train_sepsis_group
        self.patient_pos_sample_vital[:, 0, :] = self.assign_patient_value_physionet(center_node_index,label)

        for i in range(self.positive_lab_size):
            if mode == "random":
                index_neighbor = np.int(np.floor(np.random.uniform(0, len(neighbor_patient), 1)))
                patient_id = neighbor_patient[index_neighbor]
                self.positive_patient_id_list.append(patient_id)
                self.patient_pos_sample_vital[:, i + 1, :] = self.assign_patient_value_physionet(patient_id,label)
            if mode == "proximity":
                neighbor_patient = self.knn_neighbor[center_node_index]['knn_neighbor']
                patient_id = neighbor_patient[i]
                self.positive_patient_id_list.append(patient_id)
                self.patient_pos_sample_vital[:, i + 1, :] = self.assign_patient_value_physionet(patient_id, label)


    def get_negative_patient(self,label):
        self.patient_neg_sample_vital = np.zeros((self.time_sequence, self.negative_lab_size, self.item_size))
        if label == 0:
            flag = 1
            neighbor_patient = self.read_d.train_sepsis_group
        else:
            flag = 0
            neighbor_patient = self.read_d.train_non_sepsis_group

        for i in range(self.negative_lab_size):
            index_neighbor = np.int(np.floor(np.random.uniform(0, len(neighbor_patient), 1)))
            #patient_id = self.neg_patient_id[i]
            #if patient_id == center_node_index:
                #continue
            patient_id = neighbor_patient[index_neighbor]
            self.patient_neg_sample_vital[:, i, :] = self.assign_patient_value_physionet(patient_id,flag)


    def contrastive_loss(self):
        """
        Implement Contrastive Loss
        """
        """
        positive inner product
        """
        self.positive_broad = tf.broadcast_to(self.x_origin,
                                              [self.batch_size, self.positive_sample_size, self.input_size])
        self.negative_broad = tf.broadcast_to(self.x_origin,
                                              [self.batch_size, self.negative_sample_size, self.input_size])

        self.positive_broad_norm = tf.math.l2_normalize(self.positive_broad, axis=2)
        self.positive_sample_norm = tf.math.l2_normalize(self.x_skip_contrast, axis=2)

        self.positive_dot_prod = tf.multiply(self.positive_broad_norm, self.positive_sample_norm)
        # self.positive_dot_prod_sum = tf.reduce_sum(tf.math.exp(tf.reduce_sum(self.positive_dot_prod, 2)),1)
        self.positive_dot_prod_sum = tf.math.exp(tf.reduce_sum(self.positive_dot_prod, 2) / self.tau)

        """
        negative inner product
        """
        self.negative_broad_norm = tf.math.l2_normalize(self.negative_broad, axis=2)
        self.negative_sample_norm = tf.math.l2_normalize(self.x_negative_contrast, axis=2)

        self.negative_dot_prod = tf.multiply(self.negative_broad_norm, self.negative_sample_norm)
        self.negative_dot_prod_sum = tf.reduce_sum(tf.math.exp(tf.reduce_sum(self.negative_dot_prod, 2) / self.tau), 1)
        self.negative_dot_prod_sum = tf.expand_dims(self.negative_dot_prod_sum, 1)

        """
        Compute normalized probability and take log form
        """
        self.denominator_normalizer = tf.math.add(self.positive_dot_prod_sum, self.negative_dot_prod_sum)
        self.normalized_prob_log = tf.math.log(tf.math.divide(self.positive_dot_prod_sum, self.denominator_normalizer))
        self.normalized_prob_log_k = tf.reduce_sum(self.normalized_prob_log, 1)
        self.log_normalized_prob = tf.math.negative(tf.reduce_mean(self.normalized_prob_log_k, 0))

    def config_model(self):
        self.lstm_cell()
        self.build_dhgm_model()
        self.get_latent_rep_hetero()
        self.contrastive_loss()
        self.logit_sig = tf.compat.v1.layers.dense(inputs=self.x_origin_ce,
                                                   units=1,
                                                   kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                                                   activation=tf.nn.sigmoid)
        bce = tf.keras.losses.BinaryCrossentropy()
        self.cross_entropy = bce(self.logit_sig, self.input_y_logit)
        self.train_step_ce = tf.compat.v1.train.AdamOptimizer(1e-3).minimize(self.cross_entropy)
        self.train_step_combine_ce = tf.compat.v1.train.AdamOptimizer(1e-3).minimize(
            self.cross_entropy + 0.4 * self.log_normalized_prob)
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


    def train(self,data,data_label):
        """
        train the system
        """
        init_hidden_state = np.zeros(
            (self.batch_size, 1 + self.positive_lab_size + self.negative_lab_size, self.latent_dim))
        iteration = np.int(np.floor(np.float(self.length_train) / self.batch_size))

        # self.construct_knn_graph_attribute()
        for j in range(self.epoch):
            print('epoch')
            print(j)
            # self.construct_knn_graph()
            for i in range(iteration):
                self.train_one_batch_vital = self.get_batch_train(self.batch_size, i * self.batch_size,
                                                                  data,data_label,"random")

                self.err_ = self.sess.run([self.cross_entropy, self.train_step_fl],
                                          feed_dict={self.input_x: self.train_one_batch_vital,
                                                     self.input_y_logit: self.real_logit,
                                                     self.init_hiddenstate: init_hidden_state})
                print(self.err_[0])

    def pretrain_random(self,data,data_label):
        self.length_train = len(data)
        init_hidden_state = np.zeros(
            (self.batch_size, 1 + self.positive_lab_size + self.negative_lab_size, self.latent_dim))
        iteration = np.int(np.floor(np.float(self.length_train) / self.batch_size))

        for j in range(self.epoch):
            print('epoch')
            print(j)
            for i in range(iteration):
                self.train_one_batch_vital = self.get_batch_train(self.batch_size, i * self.batch_size,
                                                                  data,data_label,"random")

                self.err_ = self.sess.run([self.cross_entropy, self.train_step_cl],
                                          feed_dict={self.input_x: self.train_one_batch_vital,
                                                     self.input_y_logit: self.real_logit,
                                                     self.init_hiddenstate: init_hidden_state})
                print(self.err_[0])


    def pretrain_proximity(self,data,data_label):
        """
        train the system
        """
        self.length_train = len(data)
        init_hidden_state = np.zeros(
            (self.batch_size, 1 + self.positive_lab_size + self.negative_lab_size, self.latent_dim))
        iteration = np.int(np.floor(np.float(self.length_train) / self.batch_size))

        for j in range(self.epoch):
            print('epoch')
            print(j)
            if not j == 0:
                self.construct_knn_graph(data,data_label)
            for i in range(iteration):
                if j == 0:
                    self.train_one_batch_vital = self.get_batch_train(
                        self.batch_size, i * self.batch_size,data,data_label,"random")

                    self.err_ = self.sess.run([self.cross_entropy, self.train_step_cl],
                                              feed_dict={self.input_x: self.train_one_batch_vital,
                                                         self.input_y_logit: self.real_logit,
                                                         self.init_hiddenstate: init_hidden_state,})
                else:
                    self.train_one_batch_vital = self.get_batch_train(
                        self.batch_size, i * self.batch_size, data,data_label,"proximity")

                    self.err_ = self.sess.run([self.cross_entropy, self.train_step_cl],
                                              feed_dict={self.input_x: self.train_one_batch_vital,
                                                         self.input_y_logit: self.real_logit,
                                                         self.init_hiddenstate: init_hidden_state, })
                print(self.err_[0])


    def test(self, data, data_label):
        test_length = len(data)
        init_hidden_state = np.zeros(
            (test_length, 1 + self.positive_lab_size + self.negative_lab_size, self.latent_dim))
        self.test_data_batch_vital = self.get_batch_train(test_length, 0, data,data_label,"random")
        self.out_logit = self.sess.run(self.logit_sig, feed_dict={self.input_x: self.test_data_batch_vital,
                                                                  self.init_hiddenstate: init_hidden_state})


        self.tp_correct = 0
        self.tp_neg = 0
        for i in range(test_length):
            if self.real_logit[i, 0] == 1:
                self.tp_correct += 1
            if self.real_logit[i, 0] == 0:
                self.tp_neg += 1

        threshold = -1.01
        self.resolution = 0.01
        self.tp_total = []
        self.fp_total = []
        self.precision_total = []
        self.recall_total = []
        self.out_logit_integer = np.zeros(self.out_logit.shape[0])

        while (threshold < 1.01):
            tp_test = 0
            fp_test = 0
            fn_test = 0

            for i in range(test_length):
                if self.out_logit[i, 0] > threshold:
                    self.out_logit_integer[i] = 1

            for i in range(test_length):
                if self.real_logit[i, 0] == 1 and self.out_logit[i, 0] > threshold:
                    tp_test += 1
                if self.real_logit[i, 0] == 0 and self.out_logit[i, 0] > threshold:
                    fp_test += 1
                if self.out_logit[i, 0] < threshold and self.real_logit[i, 0] == 1:
                    fn_test += 1

            tp_rate = tp_test / self.tp_correct
            fp_rate = fp_test / self.tp_neg

            if (tp_test + fp_test) == 0:
                precision_test = 1.0
            else:
                precision_test = np.float(tp_test) / (tp_test + fp_test)
            recall_test = np.float(tp_test) / (tp_test + fn_test)

            # precision_test = precision_score(np.squeeze(self.real_logit), self.out_logit_integer, average='macro')
            # recall_test = recall_score(np.squeeze(self.real_logit), self.out_logit_integer, average='macro')
            self.tp_total.append(tp_rate)
            self.fp_total.append(fp_rate)
            self.precision_total.append(precision_test)
            self.recall_total.append(recall_test)
            threshold += self.resolution
            self.out_logit_integer = np.zeros(self.out_logit.shape[0])

    def cal_auc(self):
        self.area = 0
        self.tp_total.sort()
        self.fp_total.sort()
        for i in range(len(self.tp_total) - 1):
            x = self.fp_total[i + 1] - self.fp_total[i]
            y = (self.tp_total[i + 1] + self.tp_total[i]) / 2
            self.area += x * y

    def cal_auprc(self):
        self.area_auprc = 0
        # self.precision_total.sort()
        # self.recall_total.sort()
        for i in range(len(self.precision_total) - 1):
            x = self.recall_total[i + 1] - self.recall_total[i]
            y = (self.precision_total[i + 1] + self.precision_total[i]) / 2
            self.area_auprc += x * y