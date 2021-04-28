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

class hier_cl():
    """
    Create dynamic HGM model
    """

    def __init__(self, read_d):
        # self.hetro_model = hetro_model
        self.read_d = read_d
        self.sepsis_group = read_d.file_names_sepsis
        self.non_sepsis_group = read_d.file_names_non_sepsis
        self.train_sepsis_group = read_d.train_sepsis_group
        self.train_non_sepsis_group = read_d.train_non_sepsis_group
        self.test_sepsis_group = read_d.test_sepsis_group
        self.test_non_sepsis_group = read_d.test_non_sepsis_group
        self.train_data = read_d.train_data
        self.test_data = read_d.test_data
        self.train_data_label = read_d.train_data_label
        self.test_data_label = read_d.test_data_label
        self.length_comor = 16
        self.item_size = 14
        self.latent_dim = 50
        self.latent_dim_comor = 50
        self.gamma = 2
        self.tau = 1.5
        self.softmax_weight_threshold = 0.1
        self.epoch = 2
        self.input_size = self.latent_dim
        self.input_size_ = 2*self.latent_dim
        self.time_sequence = 6
        self.batch_size = 64
        self.size_for_pos_knn_construction = 100
        self.size_for_neg_knn_construction = 200
        self.positive_lab_size = 3
        self.negative_lab_size = 10
        self.positive_sample_size = self.positive_lab_size
        self.negative_sample_size = self.negative_lab_size
        self.knn_neighbor_numbers = self.positive_lab_size

        self.input_x_comor = tf.keras.backend.placeholder(
            [None, 1 + self.positive_lab_size + self.negative_lab_size, self.length_comor])

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
        self.hidden_global_vital = tf.reduce_mean(self.hidden_rep,1)
        self.check = concat_cur

    def comor_layer(self):
        self.Dense_comor = tf.compat.v1.layers.dense(inputs=self.input_x_comor,
                                                    units=self.latent_dim_comor,
                                                    kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                                                    activation=tf.nn.relu)

    def build_dhgm_model(self):
        """
        Build dynamic HGM model
        """
        # self.Dense_patient = tf.expand_dims(self.hidden_last,1)
        #self.Dense_patient = self.hidden_last
        self.Dense_patient_time = self.hidden_rep
        self.Dense_patient_global_vital = self.hidden_global_vital
        self.Dense_patient_global_whole = tf.concat([self.Dense_patient_global_vital,self.Dense_comor],axis=2)
        self.Dense_patient = self.Dense_patient_global_whole
        # self.Dense_patient = tf.math.l2_normalize(self.Dense_patient,axis=2)

    def get_latent_rep_hetero_time(self):
        idx_origin = tf.constant([0])
        self.x_origin_time = tf.gather(self.Dense_patient_time, idx_origin, axis=2)
        self.x_origin_ce_time = tf.squeeze(self.x_origin_time, [2])
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
        self.x_skip_patient_time = tf.gather(self.Dense_patient_time, patient_idx_skip, axis=2)
        patient_idx_negative = tf.constant([i + self.positive_lab_size + 1 for i in range(self.negative_lab_size)])
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
        patient_idx_skip = tf.constant([i + 1 for i in range(self.positive_lab_size)])
        self.x_skip_patient = tf.gather(self.Dense_patient, patient_idx_skip, axis=1)
        patient_idx_negative = tf.constant([i + self.positive_lab_size + 1 for i in range(self.negative_lab_size)])
        self.x_negative_patient = tf.gather(self.Dense_patient, patient_idx_negative, axis=1)

        # self.process_patient_att()

        # self.x_skip = tf.concat([self.x_skip_mor, self.x_skip_patient], axis=1)
        # self.x_negative = tf.concat([self.x_negative_mor, self.x_negative_patient], axis=1)
        self.x_skip_contrast = self.x_skip_patient
        self.x_negative_contrast = self.x_negative_patient

    def assign_patient_value_mimic(self,center_node_name,label):
        self.one_sample_vital = np.zeros((self.time_sequence,self.item_size))
        self.one_sample_comor = np.zeros(self.length_comor)
        if label == 1:
            table_vital,table_comor = self.read_d.read_table_sepsis(center_node_name)
        if label == 0:
            table_vital,table_comor = self.read_d.read_table_non_sepsis(center_node_name)

        self.hours = np.unique(table_vital[:,0])
        self.hours.sort()
        length = self.hours.shape[0]
        if length < self.time_sequence:
            self.selected_hours = self.hours
            index_time = self.time_sequence - length
        else:
            self.selected_hours = self.hours[length-self.time_sequence:length]
            index_time = 0
        for i in self.selected_hours:
            indexs = np.where(table_vital[:,0]==i)[0]
            table_selected = table_vital[indexs[0],1:]
            for k in range(table_selected.shape[0]):
                try:
                    table_selected[k] = (float(table_selected[k]) - self.mean_table_vital[k])/self.std_table_vital[k]
                except:
                    table_selected[k] = 0
            self.one_sample_vital[index_time,:] = table_selected[:]
            index_time += 1
        self.one_sample_comor = table_comor

        return self.one_sample_vital

    def assign_patient_value_mimic_(self,center_node_name,label):
        self.one_sample_vital = np.zeros((self.time_sequence,self.item_size))
        self.one_sample_comor = np.zeros((1,self.length_comor))
        if label == 1:
            table_vital,table_comor = self.read_d.read_table_sepsis(center_node_name)
        if label == 0:
            table_vital,table_comor = self.read_d.read_table_non_sepsis(center_node_name)

        self.hours = np.unique(table_vital[:,0])
        self.hours.sort()
        length = self.hours.shape[0]
        if length < self.time_sequence:
            self.selected_hours = self.hours
            index_time = self.time_sequence - length
        else:
            self.selected_hours = self.hours[length-self.time_sequence:length]
            index_time = 0
        for i in self.selected_hours:
            indexs = np.where(table_vital[:,0]==i)[0]
            table_selected = table_vital[indexs[0],:]
            for k in range(table_selected.shape[0]):
                try:
                    table_selected[k] = float(table_selected[k])
                except:
                    table_selected[k] = 0
            self.one_sample_vital[index_time,:] = table_selected[1:]
            index_time += 1
        self.one_sample_comor[0,:] = table_comor

        return self.one_sample_vital

    def compute_mean_std_vital(self):
        length = len(self.train_data)
        self.table_vital_ = np.zeros((length,self.item_size))
        for i in range(length):
            n = self.train_data[i]
            label = self.train_data_label[i]
            if label == 1:
                file_path = self.read_d.file_path_sepsis
            else:
                file_path = self.read_d.file_path_non_sepsis
            name = file_path + n
            one_sample_vital = self.assign_patient_value_mimic_(name,label)
            self.table_vital_[i,:] = np.mean(one_sample_vital,axis=0)

        self.mean_table_vital = np.mean(self.table_vital_,axis=0)
        self.std_table_vital = np.std(self.table_vital_,axis=0)




    def get_positive_patient(self, center_node_index, label, mode):
        self.patient_pos_sample_vital = np.zeros((self.time_sequence, self.positive_lab_size + 1, self.item_size))
        self.patient_pos_sample_comor = np.zeros((self.positive_lab_size+1,self.length_comor))

        self.positive_patient_id_list = []
        if label == 0:
            neighbor_patient = self.read_d.train_non_sepsis_group
            file_path = self.read_d.file_path_non_sepsis
        else:
            neighbor_patient = self.read_d.train_sepsis_group
            file_path = self.read_d.file_path_sepsis
        center_node_name = file_path + center_node_index
        self.patient_pos_sample_vital[:, 0, :] = self.assign_patient_value_mimic(center_node_name,label)
        self.patient_pos_sample_comor[0,:] = self.one_sample_comor

        for i in range(self.positive_lab_size):
            if mode == "random":
                index_neighbor = np.int(np.floor(np.random.uniform(0, len(neighbor_patient), 1)))
                patient_id = file_path+neighbor_patient[index_neighbor]
                self.positive_patient_id_list.append(patient_id)
                self.patient_pos_sample_vital[:, i + 1, :] = self.assign_patient_value_mimic(patient_id,label)
                self.patient_pos_sample_comor[i+1,:] = self.one_sample_comor
            if mode == "proximity":
                neighbor_patient = self.knn_neighbor[center_node_index]['knn_neighbor']
                patient_id = file_path+neighbor_patient[i]
                self.positive_patient_id_list.append(patient_id)
                self.patient_pos_sample_vital[:, i + 1, :] = self.assign_patient_value_mimic(patient_id, label)
                self.patient_pos_sample_comor[i + 1, :] = self.one_sample_comor


    def get_negative_patient(self,label):
        self.patient_neg_sample_vital = np.zeros((self.time_sequence, self.negative_lab_size, self.item_size))
        self.patient_neg_sample_comor = np.zeros((self.negative_lab_size, self.length_comor))
        if label == 0:
            flag = 1
            neighbor_patient = self.read_d.train_sepsis_group
            file_path = self.read_d.file_path_sepsis
        else:
            flag = 0
            neighbor_patient = self.read_d.train_non_sepsis_group
            file_path = self.read_d.file_path_non_sepsis

        for i in range(self.negative_lab_size):
            index_neighbor = np.int(np.floor(np.random.uniform(0, len(neighbor_patient), 1)))
            #patient_id = self.neg_patient_id[i]
            #if patient_id == center_node_index:
                #continue
            patient_id = file_path+neighbor_patient[index_neighbor]
            self.patient_neg_sample_vital[:, i, :] = self.assign_patient_value_mimic(patient_id,flag)
            self.patient_neg_sample_comor[i,:] = self.one_sample_comor

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

    def contrastive_loss_time(self):
        """
        Implement Contrastive Loss
        """
        """
        positive inner product
        """
        self.positive_broad_time = tf.broadcast_to(self.x_origin_time,
                                              [self.batch_size, self.time_sequence,self.positive_sample_size, self.input_size])
        self.negative_broad_time = tf.broadcast_to(self.x_origin_time,
                                              [self.batch_size, self.time_sequence,self.negative_sample_size, self.input_size])

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
        self.negative_dot_prod_sum_time = tf.reduce_sum(tf.reduce_sum(tf.math.exp(tf.reduce_sum(self.negative_dot_prod_time, 3) / self.tau), 2),1)
        self.negative_dot_prod_sum_time = tf.expand_dims(self.negative_dot_prod_sum_time, 1)
        self.negative_dot_prod_sum_time = tf.expand_dims(self.negative_dot_prod_sum_time, 1)

        """
        Compute normalized probability and take log form
        """
        self.denominator_normalizer_time = tf.math.add(self.positive_dot_prod_sum_time, self.negative_dot_prod_sum_time)
        self.normalized_prob_log_time = tf.math.log(tf.math.divide(self.positive_dot_prod_sum_time, self.denominator_normalizer_time))
        self.normalized_prob_log_k_time = tf.reduce_sum(tf.reduce_sum(self.normalized_prob_log_time, 2),1)
        self.log_normalized_prob_time = tf.math.negative(tf.reduce_mean(self.normalized_prob_log_k_time, 0))


    def config_model(self):
        self.lstm_cell()
        self.comor_layer()
        self.build_dhgm_model()
        self.get_latent_rep_hetero()
        self.contrastive_loss()
        self.get_latent_rep_hetero_time()
        self.contrastive_loss_time()
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
        self.train_step_combine_fl_time = tf.compat.v1.train.AdamOptimizer(1e-3).minimize(
            self.focal_loss + self.log_normalized_prob_time + self.log_normalized_prob)
        self.sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()

    def get_batch_train(self, data_length, start_index, data, data_label, mode):
        """
        get training batch data
        """
        train_one_batch_vital = np.zeros(
            (data_length, self.time_sequence, 1 + self.positive_lab_size + self.negative_lab_size, self.item_size))
        train_one_batch_comor = np.zeros(
            (data_length, 1 + self.positive_lab_size + self.negative_lab_size, self.length_comor))
        self.real_logit = np.zeros((data_length, 1))
        self.neg_patient_id = []
        for i in range(data_length):
            self.check_patient = i
            self.patient_id = data[start_index + i]
            self.neg_patient_id.append(self.patient_id)
            self.patient_label = data_label[start_index + i]
            self.real_logit[i, 0] = self.patient_label

            """
            perform different mode positive&negative sampling
            """
            self.get_positive_patient(self.patient_id,self.patient_label,mode)
            self.get_negative_patient(self.patient_label)
            train_one_data_vital = np.concatenate((self.patient_pos_sample_vital, self.patient_neg_sample_vital),
                                                  axis=1)
            train_one_data_comor = np.concatenate((self.patient_pos_sample_comor, self.patient_neg_sample_comor),
                                                  axis=0)
            train_one_batch_vital[i, :, :, :] = train_one_data_vital
            train_one_batch_comor[i,:,:] = train_one_data_comor

        return train_one_batch_vital,train_one_batch_comor

    def train(self,data,data_label):
        """
        train the system
        """
        self.length_train = len(data)
        init_hidden_state = np.zeros(
            (self.batch_size, 1 + self.positive_lab_size + self.negative_lab_size, self.latent_dim))
        iteration = np.int(np.floor(np.float(self.length_train) / self.batch_size))

        # self.construct_knn_graph_attribute()
        for j in range(self.epoch):
            print('epoch')
            print(j)
            # self.construct_knn_graph()
            for i in range(iteration):
                self.train_one_batch_vital,self.train_one_batch_comor = self.get_batch_train(self.batch_size, i * self.batch_size,
                                                                  data,data_label,"random")

                self.err_ = self.sess.run([self.cross_entropy, self.train_step_combine_fl_time],
                                          feed_dict={self.input_x: self.train_one_batch_vital,
                                                     self.input_x_comor:self.train_one_batch_comor,
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


    def test(self, data, data_label):
        test_length = len(data)
        init_hidden_state = np.zeros(
            (test_length, 1 + self.positive_lab_size + self.negative_lab_size, self.latent_dim))
        self.test_data_batch_vital,self.test_data_batch_comor = self.get_batch_train(test_length, 0, data,data_label,"random")
        self.out_logit = self.sess.run(self.logit_sig, feed_dict={self.input_x: self.test_data_batch_vital,
                                                                  self.input_x_comor:self.test_data_batch_comor,
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