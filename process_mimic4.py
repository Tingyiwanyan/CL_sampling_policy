import numpy as np
import random
import math
import time
import pandas as pd
import json
from os import listdir
from hierarchy_cl_learning import hier_cl
from tradition_baseline import tradition_b


class read_data_mimic():
    """
    Loading data, mean and std are pre-computed
    """
    def __init__(self):
        self.file_path_vital = '/home/tingyi/MIMIC_4_DATA/vital_sepsis/'
        self.file_path_lab = '/home/tingyi/MIMIC_4_DATA/lab_sepsis/'
        self.file_path_static = '/home/tingyi/MIMIC_4_DATA/static_sepsis/'
        self.file_names_vital = listdir(self.file_path_vital)
        self.data_length = len(self.file_names_vital)
        self.train_percent = 0.8
        self.test_percent = 0.2
        self.lab_duration = 4
        self.cost_upper_lab = 97
        self.cost_lower_lab = 3
        self.time_sequence = 12
        self.predict_window = 4
        self.lab_length = 25
        self.vital_length = 9
        self.vital_column = ['heart_rate', 'sbp', 'dbp', 'mbp', 'resp_rate', 'temperature', 'spo2', 'glucose', 'gcs']
        self.lab_column = ['bicarbonate','bun','calcium','chloride','creatinine','sodium','potassium','po2',
                           'pco2','paofio2ratio','ph','baseexcess','lactate','hematocrit','hemoglobin','platelet','wbc',
                           'fibrinogen','inr','pt','ptt','bilirubin_total','bilirubin_direct','icp','crp']
        self.vital_index = np.array([4, 5, 6, 7, 8, 9, 10, 11, 13])
        #self.lab_index = np.array([np.where(np.array(self.lab_column) == i)[0][0] for i in self.lab_column])
        self.mean_vital = np.array([83.835,120.927,65.337,79.980,19.111,36.795,96.531,136.453,14.604])
        self.std_vital = np.array([14.584,16.512,11.172,11.189,3.538,0.369,1.896,40.488,0.898])

        self.lab_index = np.array([6,  7,  8,  9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
       24, 25, 26, 27, 28, 29, 30, 31])
        #self.mean_lab = np.array([24.788,23.022,8.493,102.662,1.115,138.196,4.132,105.312,44.381,217.575,7.368])
        #self.std_lab = np.array([4.43,15.174,0.663,5.801,0.737,4.379,0.503,71.439,13.496,121.267,0.100,])
        #self.lab_exclude = np.array([0,80.0,0,0,5.918,0,0,371.34,0,482.857,0,])
        #self.upper_lab = np.zeros(25)
        #self.lower_lab = np.zeros(25)
        #self.mean_lab = np.zeros(25)
        #self.std_lab = np.zeros(25)
        self.mean_lab = [2.47145025e+01, 2.37183867e+01, 8.48638057e+00, 1.02733419e+02,
         1.22818479e+00, 1.38293372e+02, 4.11240348e+00, 1.08984694e+02,
         4.32920875e+01, 2.31448588e+02, 7.36967005e+00, -1.96581197e-02,
         2.23681818e+00, 3.13471821e+01, 1.02661048e+01, 2.18659762e+02,
         1.00013165e+01, 3.43672321e+02, 1.39971625e+00, 1.49682837e+01,
         3.52915028e+01, 1.06577081e+00, 2.21535867e+00, 9.79636054e+00,
         7.01419221e+01]
        self.std_lab = [3.39657452e+00, 1.50802181e+01, 5.24928835e-01, 4.69421020e+00,
        9.36981452e-01, 3.39220857e+00, 3.99833740e-01, 7.09257284e+01,
        8.42329284e+00, 1.24071269e+02, 7.80668578e-02, 4.07627223e+00,
        1.57093888e+00, 4.94436957e+00, 1.71898869e+00, 8.76498507e+01,
        3.86478997e+00, 1.54477684e+02, 4.16272930e-01, 4.35450923e+00,
        1.21782725e+01, 1.20603519e+00, 2.52391963e+00, 4.71132448e+00,
        6.66320905e+01]
        self.upper_lab = [ 32.4,  67.0        ,   9.51088889, 111.47333333,
         4.6       , 144.86791667,   5.05      , 338.9       ,
        64.        , 482.85714286,   7.5245    ,  10.        ,
         7.695     ,  41.4       ,  13.8       , 435.65      ,
        19.67      , 689.1       ,   2.75      ,  29.33333333,
        73.845     ,   5.35      ,  10.263125  ,  21.25      ,
       221.075     ]
        self.lower_lab = [ 1.80000000e+01,  7.00000000e+00,  7.43800000e+00,  9.25000000e+01,
        4.73649123e-01,  1.31000000e+02,  3.45000000e+00,  2.95500000e+01,
        2.80000000e+01,  4.28571429e+01,  7.18550000e+00, -1.00000000e+01,
        8.00000000e-01,  2.33000000e+01,  7.50000000e+00,  7.48722222e+01,
        4.10000000e+00,  1.23911111e+02,  1.00000000e+00,  1.08000000e+01,
        2.37500000e+01,  2.00000000e-01,  1.00000000e-01,  2.75000000e+00,
        2.10000000e+00]


    def read_table(self,name):
        """
        extracting sepsis feature
        """
        self.vital_table = np.array(pd.read_csv(self.file_path_vital+name))
        sort_index_vital = np.argsort(self.vital_table[:,1])
        self.vital_table = self.vital_table[sort_index_vital]
        self.lab_table = np.array(pd.read_csv(self.file_path_lab+name))
        sort_index_lab = np.argsort(self.lab_table[:, 5])
        self.lab_table = self.lab_table[sort_index_lab]
        self.static_table = np.array(pd.read_csv(self.file_path_static + name))

    def return_data_dynamic(self,name):
        """
        return 3 tensor data
        """
        self.read_table(name)
        self.one_data_tensor = np.zeros((1,self.time_sequence,self.vital_length+self.lab_length))
        if 1 in self.vital_table[:,-1]:
            self.logit_label = 1
            index_onset = np.where(self.vital_table[:,-1]==1)[0][0]
            self.hr_onset = self.vital_table[index_onset][1]
            self.predict_window_start = self.hr_onset-self.predict_window
            self.observation_window_start = self.predict_window_start-self.time_sequence
            self.assign_value_vital_time()
            self.one_data_tensor[0,:,0:self.vital_length] = self.one_data_vital
            self.assign_value_lab_time()
            self.one_data_tensor[0,:,self.vital_length:self.vital_length+self.lab_length] = self.one_data_lab
        else:
            self.logit_label = 0
            length = self.vital_table.shape[0]
            if length > 6 + self.predict_window:
                self.hr_onset = np.int(np.floor(np.random.uniform(6+self.predict_window, length, 1)))
            else:
                self.hr_onset = length
            self.predict_window_start = self.hr_onset - self.predict_window
            self.observation_window_start = self.predict_window_start - self.time_sequence
            self.assign_value_vital_time()
            self.one_data_tensor[0, :, 0:self.vital_length] = self.one_data_vital
            self.assign_value_lab_time()
            self.one_data_tensor[0, :, self.vital_length:self.vital_length + self.lab_length] = self.one_data_lab


    def assign_value_vital_time(self):
        self.one_data_vital = np.zeros((1,self.time_sequence,self.vital_length))
        for i in range(self.time_sequence):
            self.hr_current = self.observation_window_start + i
            if self.hr_current in self.vital_table[:,1]:
                hr_index = np.where(self.vital_table[:,1]==self.hr_current)[0][0]
                self.one_data_vital[0,i,:] = self.assign_value_vital_single(hr_index)


    def assign_value_vital_single(self,hr_index):
        one_vital_sample = np.zeros(self.vital_length)
        for i in range(self.vital_length):
            index = self.vital_index[i]
            value = self.vital_table[hr_index,index]
            if np.isnan(value):
                z_score = 0
            else:
                z_score = (value-self.mean_vital[i])/self.std_vital[i]
            one_vital_sample[i] = z_score

        return one_vital_sample

    def assign_value_lab_time(self):
        self.one_data_lab = np.zeros((1, self.time_sequence, self.lab_length))
        for i in range(self.time_sequence):
            self.hr_current = self.observation_window_start + i
            self.lab_chartime = self.lab_table[:,5]
            try:
                hr_index = np.where((self.lab_chartime>self.hr_current-self.lab_duration)&(self.lab_chartime<self.hr_current+self.lab_duration))[0][0]
                self.one_data_lab[0, i, :] = self.assign_value_lab_single(hr_index)
            except:
                continue

    def assign_value_lab_single(self,hr_index):
        one_lab_sample = np.zeros(self.lab_length)
        for i in range(self.lab_length):
            index = self.lab_index[i]
            value = self.lab_table[hr_index,index]
            if np.isnan(value):
                z_score = 0
            else:
                z_score = (value-self.mean_lab[i])/self.std_lab[i]
            one_lab_sample[i] = z_score

        return one_lab_sample


    def construct_vital(self):
        """
        return mean&std value for vital signals
        """
        self.vital = {}
        for i in self.file_names_vital:
            try:
                self.read_table(i)
                print(i)
                for j in range(9):
                    index = self.vital_index[j]
                    vital_name = self.vital_column[j]
                    single_mean = np.mean([value for value in self.vital_table[:, index] if not np.isnan(value)])
                    if not np.isnan(single_mean):
                        self.vital.setdefault(vital_name, []).append(single_mean)
            except:
                continue

    def construct_lab(self):
        """
        retuen mean&std value for lab signals
        """
        self.lab = {}
        for i in self.file_names_vital:
            try:
                self.read_table(i)
                print(i)
                for j in range(25):
                    index = self.lab_index[j]
                    lab_name = self.lab_column[j]
                    single_mean = np.mean([value for value in self.lab_table[:, index] if not np.isnan(value)])
                    if not np.isnan(single_mean):
                        self.lab.setdefault(lab_name, []).append(single_mean)
            except:
                continue

    def compute_mean_std_lab(self):
        self.lab['mean'] = {}
        self.lab['std'] = {}
        index = 0
        for i in range(25):
            lab_name = self.lab_column[i]
            upper_per = np.percentile(self.lab[lab_name],self.cost_upper_lab)
            lower_per = np.percentile(self.lab[lab_name],self.cost_lower_lab)
            values = [value for value in self.lab[lab_name] if value < upper_per and value > lower_per]
            mean = np.mean(values)
            std = np.std(values)
            self.lab['mean'][lab_name]=mean
            self.lab['std'][lab_name]=std
            self.upper_lab[index] = upper_per
            self.lower_lab[index] = lower_per
            self.mean_lab[index] = mean
            self.std_lab[index] = std
            index += 1


    def split_train_test(self):
        self.train_num = np.int(np.floor(self.data_length * self.train_percent))
        self.train_set = self.file_names_vital[0:self.train_num]
        self.test_set = self.file_names_vital[self.train_num:]


    """
    def split_train_test(self):
        self.train_num_non_sepsis = 2000
        self.train_num_sepsis = 340
        self.train_non_sepsis_group = self.file_names_non_sepsis[0:2000]
        self.train_non_sepsis_label = list(np.zeros(2000))
        self.train_sepsis_group = self.file_names_sepsis[0:340]
        self.train_sepsis_label = list(np.ones(340))
        self.test_non_sepsis_group = self.file_names_non_sepsis[2000:3000]
        self.test_non_sepsis_label = list(np.zeros(1000))
        self.test_sepsis_group = self.file_names_sepsis[340:510]
        self.test_sepsis_label = list(np.ones(170))
        self.train_data_ = self.train_non_sepsis_group+self.train_sepsis_group
        self.train_data_label_ = self.train_non_sepsis_label+self.train_sepsis_label
        temp = list(zip(self.train_data_,self.train_data_label_))
        random.shuffle(temp)
        self.train_data,self.train_data_label = zip(*temp)

        self.test_data_ = self.test_non_sepsis_group+self.test_sepsis_group
        self.test_data_label_ = self.test_non_sepsis_label+self.test_sepsis_label
        temp = list(zip(self.test_data_,self.test_data_label_))
        random.shuffle(temp)
        self.test_data,self.test_data_label = zip(*temp)
    """


if __name__ == "__main__":
    read_d_mimic = read_data_mimic()
    read_d_mimic.split_train_test()
    tb = tradition_b(read_d_mimic)
    #read_d_mimic.split_train_test()
    #h_cl = hier_cl(read_d_mimic)
