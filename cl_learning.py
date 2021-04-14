import math
import copy
from itertools import groupby
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


class knn_cl():
    """
    Create dynamic HGM model
    """

    def __init__(self, read_d):
        # self.hetro_model = hetro_model
        self.read_d = read_d
        self.train_data = read_d.train_data
        self.test_data = read_d.test_data
        self.latent_dim = 50
        self.item_size = 7
        self.time_sequence = 4
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
                self.init_forget_gate(shape=(self.item_size + self.lab_size + self.latent_dim, self.latent_dim)))
        self.weight_info_gate = \
            tf.Variable(self.init_info_gate(shape=(self.item_size + self.lab_size + self.latent_dim, self.latent_dim)))
        self.weight_cell_state = \
            tf.Variable(self.init_cell_state(shape=(self.item_size + self.lab_size + self.latent_dim, self.latent_dim)))
        self.weight_output_gate = \
            tf.Variable(
                self.init_output_gate(shape=(self.item_size + self.lab_size + self.latent_dim, self.latent_dim)))
        self.bias_forget_gate = tf.Variable(self.init_forget_gate_weight(shape=(self.latent_dim,)))
        self.bias_info_gate = tf.Variable(self.init_info_gate_weight(shape=(self.latent_dim,)))
        self.bias_cell_state = tf.Variable(self.init_cell_state_weight(shape=(self.latent_dim,)))
        self.bias_output_gate = tf.Variable(self.init_output_gate(shape=(self.latent_dim,)))


    def lstm_cell(self):
        cell_state = []
        hidden_rep = []
        self.project_input = tf.math.add(tf.matmul(self.input_x, self.weight_projection_w), self.bias_projection_b)
        # self.project_input = tf.matmul(self.input_x, self.weight_projection_w)
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
        one_sample_single = np.zers(self.item_size)
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
            neighbor_patient = self.read_d.non_sepsis_group
        else:
            neighbor_patient = self.read_d.sepsis_group
        self.patient_pos_sample_vital[:, 0, :] = self.assign_patient_value_physionet(center_node_index,label)

        for i in range(self.positive_lab_size):
            if mode == "random":
                index_neighbor = np.int(np.floor(np.random.uniform(0, len(neighbor_patient), 1)))
                patient_id = neighbor_patient[index_neighbor]
                self.positive_patient_id_list.append(patient_id)
                self.patient_pos_sample_vital[:, i + 1, :] = self.assign_patient_value_physionet(patient_id,label)


    def get_negative_patient(self,center_node_index,label):
        self.patient_neg_sample_vital = np.zeros((self.time_sequence, self.negative_lab_size, self.item_size))
        if label == 0:
            flag = 1
            neighbor_patient = self.read_d.sepsis_group
        else:
            flag = 0
            neighbor_patient = self.read_d.non_sepsis_group

        for i in range(self.negative_lab_size):
            #index_neighbor = np.int(np.floor(np.random.uniform(0, len(neighbor_patient), 1)))
            patient_id = self.neg_patient_id[i]
            if patient_id == center_node_index:
                continue
            #patient_id = neighbor_patient[index_neighbor]
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
        self.demo_layer()
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


    def compute_time_seq_single(self, central_node_variable):
        """
        compute single node feature values
        """
        time_seq_variable = np.zeros((self.item_size + self.lab_size, self.time_sequence))
        if self.kg.dic_patient[central_node_variable]['death_flag'] == 0:
            flag = 0
            # neighbor_patient = self.kg.dic_death[0]
        else:
            flag = 1
            # neighbor_patient = self.kg.dic_death[1]
        time_seq = self.kg.dic_patient[central_node_variable]['prior_time_vital'].keys()
        time_seq_int = [np.int(k) for k in time_seq]
        time_seq_int.sort()
        # time_index = 0
        # for j in self.time_seq_int:
        for j in range(self.time_sequence):
            # if time_index == self.time_sequence:
            #    break
            if flag == 0:
                pick_death_hour = self.kg.dic_patient[central_node_variable][
                    'pick_time']  # self.kg.mean_death_time + np.int(np.floor(np.random.normal(0, 20, 1)))
                start_time = pick_death_hour - self.predict_window_prior + float(j) * self.time_step_length
                end_time = start_time + self.time_step_length
            else:
                start_time = self.kg.dic_patient[central_node_variable][
                                 'death_hour'] - self.predict_window_prior + float(
                    j) * self.time_step_length
                end_time = start_time + self.time_step_length
            one_data_vital = self.assign_value_patient(central_node_variable, start_time, end_time)
            one_data_lab = self.assign_value_lab(central_node_variable, start_time, end_time)
            # one_data_icu_label = self.assign_value_icu_intubation(center_node_index, start_time, end_time)
            # one_data_demo = self.assign_value_demo(center_node_index)
            # self.patient_pos_sample_vital[j, 0, :] = one_data_vital
            # self.patient_pos_sample_lab[j, 0, :] = one_data_lab
            one_data = np.concatenate([one_data_vital, one_data_lab])
            time_seq_variable[:, j] = one_data

        return time_seq_variable

    def compute_relation_indicator(self, central_node, context_node):
        softmax_weight = np.zeros((self.item_size + self.lab_size))
        # features = list(self.kg.dic_vital.keys())+list(self.kg.dic_lab.keys())
        center_data = np.mean(self.compute_time_seq_single(central_node), axis=1)
        context_data = np.mean(self.compute_time_seq_single(context_node), axis=1)
        # difference = np.abs(center_data-context_data)
        difference = np.abs(center_data - context_data)

        return np.linalg.norm(difference)

    def compute_average_patient(self, central_node):
        center_data = np.mean(self.compute_time_seq_single(central_node), axis=1)

        return center_data

    def get_batch_train(self, data_length, start_index, data, data_label, mode):
        """
        get training batch data
        """
        train_one_batch_vital = np.zeros(
            (data_length, self.time_sequence, 1 + self.positive_lab_size + self.negative_lab_size, self.item_size))
        self.real_logit = np.zeros((data_length, 1))
        self.neg_patient_id = []
        for i in range(data_length):
            self.patient_id = data[start_index + i]
            self.neg_patient_id.append(self.patient_id)
        for i in range(data_length):
            self.check_patient = i
            self.patient_id = data[start_index + i]
            self.patient_label = data_label[start_index + i]
            self.real_logit[i, 0] = self.patient_label

            """
            perform different mode positive&negative sampling
            """
            self.get_positive_patient(self.patient_id,self.patient_label,mode)
            self.get_negative_patient(self.patient_id,self.patient_label)
            train_one_data_vital = np.concatenate((self.patient_pos_sample_vital, self.patient_neg_sample_vital),
                                                  axis=1)
            train_one_batch_vital[i, :, :, :] = train_one_data_vital

        return train_one_batch_vital

    def construct_knn_graph(self):
        """
        construct knn graph at every epoch
        """
        self.length_train = len(self.train_data)
        iteration = np.int(np.floor(np.float(self.length_train) / self.batch_size))

        self.knn_sim_matrix = np.zeros((iteration * self.batch_size, self.latent_dim + self.latent_dim_demo))
        self.knn_neighbor = {}
        self.knn_neg_neighbor = {}

        init_hidden_state = np.zeros(
            (self.batch_size, 1 + self.positive_lab_size + self.negative_lab_size, self.latent_dim))
        for i in range(iteration):
            self.train_one_batch_vital, self.train_one_batch_lab, self.train_one_batch_demo, self.one_batch_logit, self.one_batch_mortality, self.one_batch_com, self.one_batch_icu_intubation = self.get_batch_train_origin(
                self.batch_size, i * self.batch_size, self.train_data)
            self.test_patient = self.sess.run(self.Dense_patient,
                                              feed_dict={self.input_x_vital: self.train_one_batch_vital,
                                                         self.init_hiddenstate: init_hidden_state,
                                                         self.input_icu_intubation: self.one_batch_icu_intubation})[
                                :,
                                0, :]
            self.knn_sim_matrix[i * self.batch_size:(i + 1) * self.batch_size, :] = self.test_patient

        for i in range(iteration * self.batch_size):
            center_patient_id = self.train_data[i]
            self.knn_neighbor[center_patient_id] = {}
            self.knn_neighbor[center_patient_id]['knn_neighbor'] = []
            self.knn_neighbor[center_patient_id]['index'] = 0
            self.knn_neg_neighbor[center_patient_id] = {}
            self.knn_neg_neighbor[center_patient_id]['knn_neighbor'] = []
            self.knn_neg_neighbor[center_patient_id]['index'] = 0

        self.norm_knn = np.expand_dims(np.linalg.norm(self.knn_sim_matrix, axis=1), 1)
        self.knn_sim_matrix = self.knn_sim_matrix / self.norm_knn
        self.knn_sim_score_matrix = np.matmul(self.knn_sim_matrix, self.knn_sim_matrix.T)
        vec_compare = np.argsort(self.knn_sim_score_matrix, axis=1)
        print("Im here in constructing knn graph")

        for i in range(self.batch_size * iteration):
            # print(i)
            # vec = np.argsort(self.knn_sim_score_matrix[i,:])
            vec = vec_compare[i, :][::-1]
            center_patient_id = self.train_data[i]
            center_flag = self.kg.dic_patient[center_patient_id]['death_flag']
            # index = self.knn_neighbor[center_patient_id]['index']
            # index_real = 0
            for j in range(iteration * self.batch_size):
                index = self.knn_neighbor[center_patient_id]['index']
                if index == self.knn_neighbor_numbers or index > self.knn_neighbor_numbers:
                    break
                # if index_real == self.check_num_threshold_pos:
                # break
                compare_patient_id = self.train_data[vec[j]]
                if compare_patient_id == center_patient_id:
                    continue
                flag = self.kg.dic_patient[compare_patient_id]['death_flag']
                if center_flag == flag:
                    # if i in vec_compare[vec[j],:][::-1][0:self.check_num_threshold_pos]:
                    if not compare_patient_id in self.knn_neighbor[center_patient_id]['knn_neighbor']:
                        self.knn_neighbor[center_patient_id].setdefault('knn_neighbor', []).append(compare_patient_id)
                        self.knn_neighbor[center_patient_id]['index'] = self.knn_neighbor[center_patient_id][
                                                                            'index'] + 1
                        """
                        index_compare = self.knn_neighbor[compare_patient_id]['index']
                        if index_compare < self.knn_neighbor_numbers:
                            if not center_patient_id in self.knn_neighbor[compare_patient_id]['knn_neighbor']:
                                self.knn_neighbor[compare_patient_id].setdefault('knn_neighbor', []).append(
                                    center_patient_id)
                                self.knn_neighbor[compare_patient_id]['index'] = self.knn_neighbor[compare_patient_id][
                                                                                    'index'] + 1
                        """
                    # index_real = index_real + 1
            """
            index_real_neg = 0
            for j in range(iteration * self.batch_size):
                index = self.knn_neg_neighbor[center_patient_id]['index']
                if index == self.negative_lab_size or index > self.negative_lab_size:
                    break
                if index_real_neg == self.check_num_threshold_neg:
                    break
                compare_patient_id = self.train_data[vec[j]]
                if compare_patient_id == center_patient_id:
                    continue
                flag = self.kg.dic_patient[compare_patient_id]['death_flag']
                if not center_flag == flag:
                    if i in vec_compare[vec[j], :][::-1][0:self.check_num_threshold_neg]:
                        if not compare_patient_id in self.knn_neg_neighbor[center_patient_id]['knn_neighbor']:
                            self.knn_neg_neighbor[center_patient_id].setdefault('knn_neighbor', []).append(
                                compare_patient_id)
                            self.knn_neg_neighbor[center_patient_id]['index'] = self.knn_neg_neighbor[center_patient_id][
                                                                                'index'] + 1

                        index_compare = self.knn_neg_neighbor[compare_patient_id]['index']
                        if index_compare < self.negative_lab_size:
                            if not center_patient_id in self.knn_neg_neighbor[compare_patient_id]['knn_neighbor']:
                                self.knn_neg_neighbor[compare_patient_id].setdefault('knn_neighbor', []).append(
                                    center_patient_id)
                                self.knn_neg_neighbor[compare_patient_id]['index'] = self.knn_neg_neighbor[compare_patient_id][
                                                                                     'index'] + 1
                    index_real_neg = index_real_neg + 1
            """
            """
            index_neg = 0
            index_real_neg = 0
            for j in range(iteration*self.batch_size):
                if index_neg == self.negative_lab_size:
                    break
                compare_patient_id = self.train_data[vec[j]]
                if compare_patient_id == center_patient_id:
                    continue
                flag = self.kg.dic_patient[compare_patient_id]['death_flag']
                if not center_flag == flag:
                    if center_patient_id not in self.knn_neg_neighbor.keys():
                        self.knn_neg_neighbor[center_patient_id] = {}
                        self.knn_neg_neighbor[center_patient_id].setdefault('knn_neighbor', []).append(compare_patient_id)
                    else:
                        self.knn_neg_neighbor[center_patient_id].setdefault('knn_neighbor', []).append(compare_patient_id)

                    index_neg = index_neg + 1
            """


    def construct_knn_graph_attribute(self):
        """
        construct knn graph at every epoch using attribute information
        """
        print("Im here in constructing knn graph")
        self.length_train = len(self.train_data)
        iteration = np.int(np.floor(np.float(self.length_train) / self.batch_size))

        self.knn_sim_matrix = np.zeros((self.length_train, self.item_size + self.lab_size))
        self.knn_neighbor = {}

        for i in range(self.length_train):
            central_node = self.train_data[i]
            patient_input = self.compute_average_patient(central_node)
            self.knn_sim_matrix[i, :] = patient_input

        # self.norm_knn = np.expand_dims(np.linalg.norm(self.knn_sim_matrix, axis=1), 1)
        # self.knn_sim_matrix = self.knn_sim_matrix / self.norm_knn
        # self.knn_sim_score_matrix = np.matmul(self.knn_sim_matrix[:,0:8], self.knn_sim_matrix[:,0:8].T)
        self.knn_nbrs = NearestNeighbors(n_neighbors=self.length_train, algorithm='ball_tree').fit(
            self.knn_sim_matrix[:, self.kg.list_index])
        distance, indices = self.knn_nbrs.kneighbors(self.knn_sim_matrix[:, self.kg.list_index])
        for i in range(self.length_train):
            # print(i)
            # vec = np.argsort(self.knn_sim_score_matrix[i, :])
            # vec = vec[::-1]
            self.vec = indices
            center_patient_id = self.train_data[i]
            center_flag = self.kg.dic_patient[center_patient_id]['death_flag']
            index = 0
            for j in range(self.length_train):
                if index == self.positive_lab_size:
                    break
                compare_patient_id = self.train_data[self.vec[i, j]]
                if compare_patient_id == center_patient_id:
                    continue
                flag = self.kg.dic_patient[compare_patient_id]['death_flag']
                if not center_flag == flag:
                    continue

                if center_patient_id not in self.knn_neighbor.keys():
                    self.knn_neighbor[center_patient_id] = {}
                    self.knn_neighbor[center_patient_id].setdefault('knn_neighbor', []).append(compare_patient_id)
                else:
                    self.knn_neighbor[center_patient_id].setdefault('knn_neighbor', []).append(compare_patient_id)

                index = index + 1


    def check_higest_value(self, neighbors, compare_graph):
        highest_neighbor = neighbors[0]
        value = compare_graph[neighbors[0]]['similarity']
        for i in neighbors:
            value_compare = compare_graph[i]['similarity']
            if value_compare > value:
                highest_neighbor = i

        return highest_neighbor

    def get_positive_patient_knn(self, center_node_index):
        self.patient_pos_sample_vital = np.zeros((self.time_sequence, self.positive_lab_size + 1, self.item_size))
        self.patient_pos_sample_lab = np.zeros((self.time_sequence, self.positive_lab_size + 1, self.lab_size))
        self.patient_pos_sample_icu_intubation_label = np.zeros((self.time_sequence, self.positive_lab_size + 1, 2))
        self.patient_pos_sample_demo = np.zeros((self.positive_lab_size + 1, self.demo_size))
        self.patient_pos_sample_com = np.zeros((self.positive_lab_size + 1, self.com_size))
        if self.kg.dic_patient[center_node_index]['death_flag'] == 0:
            flag = 0
            neighbor_patient_ = self.kg.dic_death[0]
        else:
            flag = 1
            neighbor_patient_ = self.kg.dic_death[1]
        neighbor_patient = self.knn_neighbor[center_node_index]['knn_neighbor']
        if len(neighbor_patient) == 0:
            neighbor_patient = [center_node_index]

        time_seq = self.kg.dic_patient[center_node_index]['prior_time_vital'].keys()
        time_seq_int = [np.int(k) for k in time_seq]
        time_seq_int.sort()
        # time_index = 0
        # for j in self.time_seq_int:
        for j in range(self.time_sequence):
            # if time_index == self.time_sequence:
            #    break
            if flag == 0:
                pick_death_hour = self.kg.dic_patient[center_node_index][
                    'pick_time']  # self.kg.mean_death_time + np.int(np.floor(np.random.normal(0, 20, 1)))
                start_time = pick_death_hour - self.predict_window_prior + float(j) * self.time_step_length
                end_time = start_time + self.time_step_length
            else:
                start_time = self.kg.dic_patient[center_node_index]['death_hour'] - self.predict_window_prior + float(
                    j) * self.time_step_length
                end_time = start_time + self.time_step_length
            one_data_vital = self.assign_value_patient(center_node_index, start_time, end_time)
            one_data_lab = self.assign_value_lab(center_node_index, start_time, end_time)
            # one_data_icu_label = self.assign_value_icu_intubation(center_node_index, start_time, end_time)
            # one_data_demo = self.assign_value_demo(center_node_index)
            self.patient_pos_sample_vital[j, 0, :] = one_data_vital
            self.patient_pos_sample_lab[j, 0, :] = one_data_lab
            # self.patient_pos_sample_icu_intubation_label[j,0,:] = one_data_icu_label
            # time_index += 1
        one_data_demo = self.assign_value_demo(center_node_index)
        # one_data_com = self.assign_value_com(center_node_index)
        self.patient_pos_sample_demo[0, :] = one_data_demo
        # self.patient_pos_sample_com[0,:] = one_data_com
        for i in range(self.positive_lab_size):
            index_neighbor = np.int(np.floor(np.random.uniform(0, len(neighbor_patient), 1)))
            patient_id = neighbor_patient[index_neighbor]
            time_seq = self.kg.dic_patient[patient_id]['prior_time_vital'].keys()
            time_seq_int = [np.int(k) for k in time_seq]
            time_seq_int.sort()
            one_data_demo = self.assign_value_demo(patient_id)
            # one_data_com = self.assign_value_com(patient_id)
            self.patient_pos_sample_demo[i + 1, :] = one_data_demo
            # self.patient_pos_sample_com[i+1,:] = one_data_com
            # time_index = 0
            # for j in time_seq_int:
            for j in range(self.time_sequence):
                # if time_index == self.time_sequence:
                #   break
                # self.time_index = np.int(j)
                # start_time = float(j)*self.time_step_length
                # end_time = start_time + self.time_step_length
                if flag == 0:
                    pick_death_hour = self.kg.dic_patient[center_node_index][
                        'pick_time']  # self.kg.mean_death_time + np.int(np.floor(np.random.normal(0, 20, 1)))
                    start_time = pick_death_hour - self.predict_window_prior + float(j) * self.time_step_length
                    end_time = start_time + self.time_step_length
                else:
                    start_time = self.kg.dic_patient[patient_id]['death_hour'] - self.predict_window_prior + float(
                        j) * self.time_step_length
                    end_time = start_time + self.time_step_length
                one_data_vital = self.assign_value_patient(patient_id, start_time, end_time)
                one_data_lab = self.assign_value_lab(patient_id, start_time, end_time)
                # one_data_icu_label = self.assign_value_icu_intubation(patient_id, start_time, end_time)
                self.patient_pos_sample_vital[j, i + 1, :] = one_data_vital
                self.patient_pos_sample_lab[j, i + 1, :] = one_data_lab
                # self.patient_pos_sample_icu_intubation_label[j,i+1,:] = one_data_icu_label
                # time_index += 1


    def train_fl(self):
        self.length_train = len(self.train_data)
        init_hidden_state = np.zeros(
            (self.batch_size, 1 + self.positive_lab_size + self.negative_lab_size, self.latent_dim))
        iteration = np.int(np.floor(np.float(self.length_train) / self.batch_size))

        for i in range(iteration):
            self.train_one_batch_vital, self.train_one_batch_lab, self.train_one_batch_demo, self.one_batch_logit, self.one_batch_mortality, self.one_batch_com, self.one_batch_icu_intubation = self.get_batch_train_origin(
                self.batch_size, i * self.batch_size, self.train_data)

            self.err_ = self.sess.run([self.cross_entropy, self.train_step_fl],
                                      feed_dict={self.input_x_vital: self.train_one_batch_vital,
                                                 self.input_x_lab: self.train_one_batch_lab,
                                                 self.input_x_demo: self.train_one_batch_demo,
                                                 # self.input_x_com: self.one_batch_com,
                                                 # self.lab_test: self.one_batch_item,
                                                 self.input_y_logit: self.real_logit,
                                                 self.mortality: self.one_batch_mortality,
                                                 self.init_hiddenstate: init_hidden_state,
                                                 self.input_icu_intubation: self.one_batch_icu_intubation})
            print(self.err_[0])

    def train_ce(self):
        self.length_train = len(self.train_data)
        init_hidden_state = np.zeros(
            (self.batch_size, 1 + self.positive_lab_size + self.negative_lab_size, self.latent_dim))
        iteration = np.int(np.floor(np.float(self.length_train) / self.batch_size))

        for i in range(iteration):
            self.train_one_batch_vital, self.train_one_batch_lab, self.train_one_batch_demo, self.one_batch_logit, self.one_batch_mortality, self.one_batch_com, self.one_batch_icu_intubation = self.get_batch_train_origin(
                self.batch_size, i * self.batch_size, self.train_data)

            self.err_ = self.sess.run([self.cross_entropy, self.train_step_ce],
                                      feed_dict={self.input_x_vital: self.train_one_batch_vital,
                                                 self.input_x_lab: self.train_one_batch_lab,
                                                 self.input_x_demo: self.train_one_batch_demo,
                                                 # self.input_x_com: self.one_batch_com,
                                                 # self.lab_test: self.one_batch_item,
                                                 self.input_y_logit: self.real_logit,
                                                 self.mortality: self.one_batch_mortality,
                                                 self.init_hiddenstate: init_hidden_state,
                                                 self.input_icu_intubation: self.one_batch_icu_intubation})
            print(self.err_[0])

    def train(self):
        """
        train the system
        """
        # self.area_total = []
        # self.auprc_total = []
        self.length_train = len(self.train_data)
        init_hidden_state = np.zeros(
            (self.batch_size, 1 + self.positive_lab_size + self.negative_lab_size, self.latent_dim))
        iteration = np.int(np.floor(np.float(self.length_train) / self.batch_size))

        # self.construct_knn_graph_attribute()
        for j in range(self.epoch):
            print('epoch')
            print(j)
            # self.construct_knn_graph()
            for i in range(iteration):
                self.train_one_batch_vital, self.train_one_batch_lab, self.train_one_batch_demo, self.one_batch_logit, self.one_batch_mortality, self.one_batch_com, self.one_batch_icu_intubation = self.get_batch_train_origin(
                    self.batch_size, i * self.batch_size, self.train_data)

                self.err_ = self.sess.run([self.cross_entropy, self.train_step_cl],
                                          feed_dict={self.input_x_vital: self.train_one_batch_vital,
                                                     self.input_x_lab: self.train_one_batch_lab,
                                                     self.input_x_demo: self.train_one_batch_demo,
                                                     # self.input_x_com: self.one_batch_com,
                                                     # self.lab_test: self.one_batch_item,
                                                     self.input_y_logit: self.real_logit,
                                                     self.mortality: self.one_batch_mortality,
                                                     self.init_hiddenstate: init_hidden_state,
                                                     self.input_icu_intubation: self.one_batch_icu_intubation})
                print(self.err_[0])

                """
                self.err_lstm = self.sess.run([self.cross_entropy, self.train_step_cross_entropy,self.init_hiddenstate,self.output_layer,self.logit_sig],
                                     feed_dict={self.input_x: self.train_one_batch,
                                                self.input_y_logit: self.one_batch_logit,
                                                self.init_hiddenstate:init_hidden_state})
                print(self.err_lstm[0])
                """
            # self.test(self.test_data)
            # self.cal_auc()
            # self.cal_auprc()
            # self.area_total.append(self.area)
            # self.auprc_total.append(self.area_auprc)

    def train_att(self):
        """
        train the system
        """
        # self.area_total = []
        # self.auprc_total = []
        self.length_train = len(self.train_data)
        init_hidden_state = np.zeros(
            (self.batch_size, 1 + self.positive_lab_size + self.negative_lab_size, self.latent_dim))
        iteration = np.int(np.floor(np.float(self.length_train) / self.batch_size))

        self.construct_knn_graph_attribute()
        for j in range(self.epoch):
            print('epoch')
            print(j)
            # self.construct_knn_graph()
            for i in range(iteration):
                self.train_one_batch_vital, self.train_one_batch_lab, self.train_one_batch_demo, self.one_batch_logit, self.one_batch_mortality, self.one_batch_com, self.one_batch_icu_intubation = self.get_batch_train(
                    self.batch_size, i * self.batch_size, self.train_data)

                self.err_ = self.sess.run([self.cross_entropy, self.train_step_cl],
                                          feed_dict={self.input_x_vital: self.train_one_batch_vital,
                                                     self.input_x_lab: self.train_one_batch_lab,
                                                     self.input_x_demo: self.train_one_batch_demo,
                                                     # self.input_x_com: self.one_batch_com,
                                                     # self.lab_test: self.one_batch_item,
                                                     self.input_y_logit: self.real_logit,
                                                     self.mortality: self.one_batch_mortality,
                                                     self.init_hiddenstate: init_hidden_state,
                                                     self.input_icu_intubation: self.one_batch_icu_intubation})
                print(self.err_[0])

    def train_combine(self):
        """
        train the system
        """
        # self.area_total = []
        # self.auprc_total = []
        self.length_train = len(self.train_data)
        init_hidden_state = np.zeros(
            (self.batch_size, 1 + self.positive_lab_size + self.negative_lab_size, self.latent_dim))
        iteration = np.int(np.floor(np.float(self.length_train) / self.batch_size))

        for j in range(self.epoch):
            print('epoch')
            print(j)
            if not j == 0:
                self.construct_knn_graph()
            for i in range(iteration):
                if j == 0:
                    self.train_one_batch_vital, self.train_one_batch_lab, self.train_one_batch_demo, self.one_batch_logit, self.one_batch_mortality, self.one_batch_com, self.one_batch_icu_intubation = self.get_batch_train_origin(
                        self.batch_size, i * self.batch_size, self.train_data)

                    self.err_ = self.sess.run([self.cross_entropy, self.train_step_cl],
                                              feed_dict={self.input_x_vital: self.train_one_batch_vital,
                                                         self.input_x_lab: self.train_one_batch_lab,
                                                         self.input_x_demo: self.train_one_batch_demo,
                                                         # self.input_x_com: self.one_batch_com,
                                                         # self.lab_test: self.one_batch_item,
                                                         self.input_y_logit: self.real_logit,
                                                         self.mortality: self.one_batch_mortality,
                                                         self.init_hiddenstate: init_hidden_state,
                                                         self.input_icu_intubation: self.one_batch_icu_intubation})
                else:
                    self.train_one_batch_vital, self.train_one_batch_lab, self.train_one_batch_demo, self.one_batch_logit, self.one_batch_mortality, self.one_batch_com, self.one_batch_icu_intubation = self.get_batch_train(
                        self.batch_size, i * self.batch_size, self.train_data)

                    self.err_ = self.sess.run([self.cross_entropy, self.train_step_cl],
                                              feed_dict={self.input_x_vital: self.train_one_batch_vital,
                                                         self.input_x_lab: self.train_one_batch_lab,
                                                         self.input_x_demo: self.train_one_batch_demo,
                                                         # self.input_x_com: self.one_batch_com,
                                                         # self.lab_test: self.one_batch_item,
                                                         self.input_y_logit: self.real_logit,
                                                         self.mortality: self.one_batch_mortality,
                                                         self.init_hiddenstate: init_hidden_state,
                                                         self.input_icu_intubation: self.one_batch_icu_intubation})
                print(self.err_[0])

            # self.test(self.test_data)
            # self.cal_auc()
            # self.cal_auprc()
            # self.area_total.append(self.area)
            # self.auprc_total.append(self.area_auprc)

    def test(self, data):
        Death = np.zeros([1, 2])
        Death[0][1] = 1
        test_length = len(data)
        init_hidden_state = np.zeros(
            (test_length, 1 + self.positive_lab_size + self.negative_lab_size, self.latent_dim))
        self.test_data_batch_vital, self.test_one_batch_lab, self.test_one_batch_demo, self.test_logit, self.test_mortality, self.test_com, self.one_batch_icu_intubation = self.get_batch_train_origin(
            test_length, 0, data)
        self.out_logit = self.sess.run(self.logit_sig, feed_dict={self.input_x_vital: self.test_data_batch_vital,
                                                                  self.input_x_lab: self.test_one_batch_lab,
                                                                  self.input_x_demo: self.test_one_batch_demo,
                                                                  # self.input_x_com: self.test_com,
                                                                  self.init_hiddenstate: init_hidden_state,
                                                                  self.input_icu_intubation: self.one_batch_icu_intubation})

        self.out_test_patient = self.sess.run(self.Dense_patient,
                                              feed_dict={self.input_x_vital: self.test_data_batch_vital,
                                                         self.input_x_lab: self.test_one_batch_lab,
                                                         self.input_x_demo: self.test_one_batch_demo,
                                                         # self.input_x_com: self.test_com,
                                                         self.init_hiddenstate: init_hidden_state,
                                                         self.input_icu_intubation: self.one_batch_icu_intubation})[:,
                                0, :]
        """
        self.test_att_score = self.sess.run([self.score_attention,self.input_importance,self.input_x],feed_dict={self.input_x_vital: self.test_data_batch_vital,
                                                                         self.input_x_lab: self.test_one_batch_lab,
                                                                         self.input_x_demo: self.test_one_batch_demo,
                                                                         self.init_hiddenstate: init_hidden_state,
                                                                         self.Death_input: Death,
                                                                         self.input_icu_intubation:self.one_batch_icu_intubation})
        """

        """
        self.correct_predict_death = np.array(self.correct_predict_death)

        feature_len = self.item_size + self.lab_size


        self.test_data_scores = self.test_att_score[1][self.correct_predict_death,:,0,:]
        self.ave_data_scores = np.zeros((self.time_sequence,feature_len))

        count = 0
        value = 0

        for j in range(self.time_sequence):
            for p in range(feature_len):
                for i in range(self.correct_predict_death.shape[0]):
                    if self.test_data_scores[i,j,p]!=0:
                        count += 1
                        value += self.test_data_scores[i,j,p]
                if count == 0:
                    continue
                self.ave_data_scores[j,p] = float(value/count)
                count = 0
                value = 0
        """

        self.tp_correct = 0
        self.tp_neg = 0
        for i in range(test_length):
            if self.test_logit[i, 1] == 1:
                self.tp_correct += 1
            if self.test_logit[i, 0] == 1:
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

    def bootstraping(self):
        self.config_model()
        self.sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        self.train_data = self.train_data_whole[i]
        self.test_data = self.test_data_whole[i]
        # self.construct_knn_graph_attribute()
        # print("im here in train representation")
        # self.train_representation()
        print("im here in train")
        self.train()
        self.test(self.test_data)

    def acc_epoch(self):
        self.f1_score_total = []
        self.acc_total = []
        self.area_total = []
        self.auprc_total = []
        self.test_logit_total = []
        self.tp_score_total = []
        self.fp_score_total = []
        self.precision_score_total = []
        self.precision_curve_total = []
        self.recall_score_total = []
        self.recall_curve_total = []
        self.test_patient_whole = []

        self.config_model()

    def cross_validation(self):
        self.f1_score_total = []
        self.acc_total = []
        self.area_total = []
        self.auprc_total = []
        self.test_logit_total = []
        self.tp_score_total = []
        self.fp_score_total = []
        self.precision_score_total = []
        self.precision_curve_total = []
        self.recall_score_total = []
        self.recall_curve_total = []
        self.test_patient_whole = []
        # feature_len = self.item_size + self.lab_size
        # self.ave_data_scores_total = np.zeros((self.time_sequence, feature_len))
        # self.generate_orthogonal_relatoin()

        self.config_model()
        for i in range(3):
            self.sess = tf.InteractiveSession()
            tf.global_variables_initializer().run()
            tf.local_variables_initializer().run()
            self.train_data = self.train_data_whole[i]
            self.test_data = self.test_data_whole[i]
            # self.construct_knn_graph_attribute()
            print("im here in train representation")
            self.train_representation()
            print("im here in train")
            self.train()
            self.test(self.test_data)
            # self.f1_score_total.append(self.f1_test)
            # self.acc_total.append(self.acc)
            self.tp_score_total.append(self.tp_total)
            self.fp_score_total.append(self.fp_total)
            self.cal_auc()
            self.cal_auprc()
            self.area_total.append(self.area)
            self.auprc_total.append(self.area_auprc)
            # self.precision_score_total.append(self.precision_test)
            # self.recall_score_total.append(self.recall_test)
            # self.precision_curve_total.append(self.precision_total)
            # self.recall_curve_total.append(self.recall_total)
            # self.test_patient_whole.append(self.test_patient)
            self.test_logit_total.append(self.test_logit)
            # self.ave_data_scores_total += self.ave_data_scores
            self.sess.close()

        # self.ave_data_scores_total = self.ave_data_scores_total/5
        # self.norm = np.linalg.norm(self.ave_data_scores_total)
        # self.ave_data_scores_total = self.ave_data_scores_total/self.norm
        self.tp_ave_score = np.sum(self.tp_score_total, 0) / 5
        self.fp_ave_score = np.sum(self.fp_score_total, 0) / 5
        self.precision_ave_score = np.sum(self.precision_curve_total, 0) / 5
        self.recall_ave_score = np.sum(self.recall_curve_total, 0) / 5
        # print("f1_ave_score")
        # print(np.mean(self.f1_score_total))
        # print("acc_ave_score")
        # print(np.mean(self.acc_total))
        print("area_ave_score")
        print(np.mean(self.area_total))
        # print("precision_ave_score")
        # print(np.mean(self.precision_total))
        # print("recall_ave_score")
        # print(np.mean(self.recall_total))
        print("auprc_ave_score")
        print(np.mean(self.auprc_total))

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