import numpy as np
import random
import math
import time
import pandas as pd
import json
from os import listdir
from hierarchy_cl_learning import hier_cl
from tradition_baseline import tradition_b
from deep_learning_appro import dp_appro
from seq_cl import seq_cl


class read_data_mimic():
    """
    Loading data, mean and std are pre-computed
    """
    def __init__(self):
        self.file_path_vital_cohort = '/home/tingyi/m3_data/cohort_vital/'
        self.file_path_vital_control = '/home/tingyi/m3_data/control_vital/'
        self.file_path_lab_cohort = '/home/tingyi/m3_data/cohort_lab/'
        self.file_path_lab_control = '/home/tingyi/m3_data/control_lab/'
        self.file_path_blood_gas_cohort = '/home/tingyi/m3_data/cohort_blood_gas/'
        self.file_path_blood_gas_control = '/home/tingyi/m3_data/control_blood_gas/'
        self.file_path_sepsis_cohort = '/home/tingyi/m3_data/sepsis_cohort/'
        self.file_path_sepsis_control = '/home/tingyi/m3_data/sepsis_control/'
        self.file_path_sofa_6hourly_cohort = '/home/tingyi/m3_data/sofa_6hourly_cohort/'
        self.file_path_sofa_6hourly_control = '/home/tingyi/m3_data/sofa_6hourly_control/'
        file = open("sepsis_cohort.txt","r")
        self.cohort_names = file.read()
        self.file_names_cohort = self.cohort_names.split("\n") #listdir(self.file_path_vital)

        file_ = open("sepsis_control.txt", "r")
        self.control_names = file_.read()
        self.file_names_control = self.control_names.split("\n")  # listdir(self.file_path_vital)

        self.file_names_all = self.file_names_cohort+self.file_names_control

        self.time_sequence = 4
        self.vital_length = 8
        self.lab_length = 19
        self.blood_length = 27

        self.predict_window = 0

        self.mean_vital = [ 91.67865086, 120.80026402,  60.98199649,  78.39028852,
        19.64776703,  36.86108155,  96.71496552, 138.30870759]
        self.std_vital = [ 22.06970808,  15.60371556,  10.04527293,  10.28492616,
         3.68351108,   0.54126899,   2.35342106, 145.69260974]

        self.mean_lab = [ 13.52917098,   3.04025644,   6.3593078 ,  25.10450625,
         2.54660002,   1.30908416, 104.34706151, 128.61812616,
        32.38748733,  10.89649508,   2.02038763, 242.07981746,
         4.12945378,  37.50049151,   1.43635189,  15.50245902,
       138.69095615,  25.70903147,  11.44305704]
        self.std_lab = [  3.03052342,   0.62272815,   7.08962683,   3.96140344,
         4.40555066,   1.24528894,   4.71655414,  33.8800575 ,
         6.10499616,   2.10520567,   1.51516171, 119.79212037,
         0.46200667,  14.7245143 ,   0.63962939,   4.60692571,
         3.7211668 ,  18.65681214,   7.41795063]

        self.mean_blood = [481.65364062,   3.63645308,  24.3842838 ,  26.15601104,
         0.        , 105.65171196,   1.14570724, 143.38832569,
        32.73734709,  10.90898125,   0.        ,   2.12544714,
         2.30545267,  10.86782553,  55.49034225,  91.69372169,
        42.80988725,   6.94946502,   7.39624815, 150.19409285,
         4.21586548,  80.11906788, 136.99002226,  37.13594438,
       529.0333758 ,   0.        ,   0.        ]

        self.std_blood = [1.13392363e+02, 2.52987150e+00, 6.28620091e+00, 4.85620904e+00,
       0.00000000e+00, 5.88064883e+00, 1.92036194e-01, 4.70142389e+01,
       5.48629000e+00, 2.10731085e+00, 0.00000000e+00, 1.62243407e+00,
       5.58948891e+00, 1.39987057e+01, 2.11282943e+01, 9.00178541e+00,
       9.32691113e+00, 8.41019233e+00, 6.20876461e-02, 7.94072907e+01,
       6.34287493e-01, 1.54666235e+01, 4.81058879e+00, 8.07119217e-01,
       1.52688481e+02, 0.00000000e+00, 0.00000000e+00]


    def read_table_cohort(self,name):
        """
        extracting sepsis feature
        """
        self.vital_cohort = pd.read_csv(self.file_path_vital_cohort+name+'.csv')
        self.vital_cohort_ar = np.array(self.vital_cohort)
        self.lab_cohort = pd.read_csv(self.file_path_lab_cohort + name+'.csv')
        self.lab_cohort_ar = np.array(self.lab_cohort)
        self.blood_cohort = pd.read_csv(self.file_path_blood_gas_cohort + name+'.csv')
        self.blood_cohort_ar = np.array(self.blood_cohort)
        self.sepsis_cohort = pd.read_csv(self.file_path_sepsis_cohort + name+'.csv')
        self.sepsis_cohort_ar = np.array(self.sepsis_cohort)
        sort_index_vital = np.argsort(self.vital_cohort_ar[:,1])
        self.vital_cohort_ar = self.vital_cohort_ar[sort_index_vital]
        sort_index_lab = np.argsort(self.lab_cohort_ar[:,1])
        self.lab_cohort_ar = self.lab_cohort_ar[sort_index_lab]
        sort_index_blood = np.argsort(self.blood_cohort_ar[:,1])
        self.blood_cohort_ar = self.blood_cohort_ar[sort_index_blood]

    def read_table_control(self,name):
        self.vital_control = pd.read_csv(self.file_path_vital_control+name+'.csv')
        self.vital_control_ar = np.array(self.vital_control)
        self.lab_control = pd.read_csv(self.file_path_lab_control + name+'.csv')
        self.lab_control_ar = np.array(self.lab_control)
        self.blood_control = pd.read_csv(self.file_path_blood_gas_control + name+'.csv')
        self.blood_control_ar = np.array(self.blood_control)
        self.sepsis_control = pd.read_csv(self.file_path_sepsis_control + name+'.csv')
        self.sepsis_control_ar = np.array(self.sepsis_control)

        sort_index_vital = np.argsort(self.vital_control_ar[:, 1])
        self.vital_control_ar = self.vital_control_ar[sort_index_vital]
        sort_index_lab = np.argsort(self.lab_control_ar[:, 1])
        self.lab_control_ar = self.lab_control_ar[sort_index_lab]
        sort_index_blood = np.argsort(self.blood_control_ar[:, 1])
        self.blood_control_ar = self.blood_control_ar[sort_index_blood]

    def return_data_dynamic_cohort(self,name):
        """
        return 3 tensor data
        """
        self.read_table_cohort(name)
        self.one_data_tensor = np.zeros((self.time_sequence,self.vital_length+self.lab_length+self.blood_length))
        self.logit_label = 1
        self.hr_onset = self.sepsis_cohort_ar[0,1]
        self.predict_window_start = self.hr_onset-self.predict_window
        #self.observation_window_start = self.predict_window_start-(self.time_sequence*self.time_period)
        self.assign_value_vital_time(self.predict_window_start)
        self.one_data_tensor[:,0:self.vital_length] = self.one_data_vital
        self.assign_value_lab_time(self.predict_window_start)
        self.one_data_tensor[:,self.vital_length:self.vital_length+self.lab_length] = self.one_data_lab
        self.assign_value_blood_time(self.predict_window_start)
        self.one_data_tensor[:, self.vital_length+self.lab_length:self.vital_length + self.lab_length+self.blood_length] = self.one_data_blood

    def return_data_dynamic_control(self, name):
        """
        return 3 tensor data
        """
        self.read_table_control(name)
        self.one_data_tensor = np.zeros((self.time_sequence, self.vital_length + self.lab_length + self.blood_length))
        self.logit_label = 0
        self.hr_onset = self.sepsis_control_ar[0, 1]
        self.predict_window_start = self.hr_onset - self.predict_window
        # self.observation_window_start = self.predict_window_start-(self.time_sequence*self.time_period)
        self.assign_value_vital_time(self.predict_window_start)
        self.one_data_tensor[:, 0:self.vital_length] = self.one_data_vital
        self.assign_value_lab_time(self.predict_window_start)
        self.one_data_tensor[:, self.vital_length:self.vital_length + self.lab_length] = self.one_data_lab
        self.assign_value_blood_time(self.predict_window_start)
        self.one_data_tensor[:,
        self.vital_length + self.lab_length:self.vital_length + self.lab_length + self.blood_length] = self.one_data_blood


    def assign_value_vital_time(self,hr_back):
        self.one_data_vital = np.zeros((self.time_sequence,self.vital_length))
        for i in range(self.time_sequence):
            self.hr_current = hr_back-self.time_sequence + i
            self.one_data_vital[i,:] = self.assign_value_vital_single(self.hr_current)




    def assign_value_vital_single(self,hr_index):
        one_vital_sample = np.zeros(self.vital_length)
        for i in range(self.vital_length):
            index = i + 4
            try:
                if self.logit_label == 1:
                    index_hr = np.where(self.vital_cohort_ar[:,1]==hr_index)[0][0]
                    value = self.vital_cohort_ar[index_hr,index]
                else:
                    index_hr = np.where(self.vital_control_ar[:, 1] == hr_index)[0][0]
                    value = self.vital_control_ar[index_hr, index]
            except:
                value = 0
            if np.isnan(value):
                z_score = 0
            else:
                z_score = (value-self.mean_vital[i])/self.std_vital[i]

            one_vital_sample[i] = z_score

        return one_vital_sample


    def assign_value_lab_time(self,hr_back):
        self.one_data_lab = np.zeros((self.time_sequence, self.lab_length))
        for i in range(self.time_sequence):
            self.hr_current = hr_back-self.time_sequence + i
            self.one_data_lab[i, :] = self.assign_value_lab_single(self.hr_current)


    def assign_value_lab_single(self,hr_index):
        one_lab_sample = np.zeros(self.lab_length)
        for i in range(self.lab_length):
            index = i+6
            try:
                if self.logit_label == 1:
                    index_hr = np.where(self.lab_cohort_ar[:, 1] == hr_index)[0][0]
                    value = self.lab_cohort_ar[index_hr,index]
                else:
                    index_hr = np.where(self.lab_control_ar[:, 1] == hr_index)[0][0]
                    value = self.lab_control_ar[index_hr, index]
            except:
                value = 0
            if np.isnan(value):
                z_score = 0
            else:
                z_score = (value-self.mean_lab[i])/ self.std_lab[i]

            one_lab_sample[i] = z_score


        return one_lab_sample

    def assign_value_blood_time(self,hr_back):
        self.one_data_blood = np.zeros((self.time_sequence,self.blood_length))
        for i in range(self.time_sequence):
            self.hr_current = hr_back-self.time_sequence + i
            self.one_data_blood[i,:] = self.assign_value_blood_single(self.hr_current)




    def assign_value_blood_single(self,hr_index):
        one_blood_sample = np.zeros(self.blood_length)
        for i in range(self.blood_length):
            index = i + 7
            try:
                if self.logit_label == 1:
                    index_hr = np.where(self.blood_cohort_ar[:, 1] == hr_index)[0][0]
                    value = self.blood_cohort_ar[index_hr,index]
                else:
                    index_hr = np.where(self.blood_control_ar[:, 1] == hr_index)[0][0]
                    value = self.blood_control_ar[index_hr, index]
            except:
                value = 0
            if np.isnan(value) or self.std_blood[i]==0:
                z_score = 0
            else:
                z_score = (value-self.mean_blood[i])/self.std_blood[i]

            one_blood_sample[i] = z_score

        return one_blood_sample


    def construct_vital(self):
        """
        return mean&std value for vital signals
        """
        self.vital = {}
        for i in self.file_names_cohort:
            #i = i + '.csv'
            try:
                self.read_table_cohort(i)
                print(i)
                #if 1 in self.vital_table[:, -1]:
                    #print(i)
                for j in range(self.vital_length):
                    index = j+4
                    vital_name = self.vital_cohort.columns[index]
                    single_mean = np.mean([value for value in self.vital_cohort_ar[:, index] if not np.isnan(value)])
                    if not np.isnan(single_mean):
                        self.vital.setdefault(vital_name, []).append(single_mean)
            except:
                continue


        for i in self.file_names_control:
            # i = i + '.csv'
            try:
                self.read_table_control(i)
                print(i)
                # if 1 in self.vital_table[:, -1]:
                # print(i)
                for j in range(self.vital_length):
                    index = j + 4
                    vital_name = self.vital_control.columns[index]
                    single_mean = np.mean([value for value in self.vital_control_ar[:, index] if not np.isnan(value)])
                    if not np.isnan(single_mean):
                        self.vital.setdefault(vital_name, []).append(single_mean)
            except:
                continue



    def compute_mean_std_vital(self):
        self.vital['mean'] = {}
        self.vital['std'] = {}
        self.mean_vital = np.zeros(self.vital_length)
        self.std_vital = np.zeros(self.vital_length)
        #index = 0
        for i in range(self.vital_length):
            index = i+4
            self.vital_name = self.vital_cohort.columns[index]
            #upper_per = np.percentile(self.vital[vital_name], self.cost_upper_lab)
            #lower_per = np.percentile(self.vital[vital_name], self.cost_lower_lab)
            values = [value for value in self.vital[self.vital_name]]
            mean = np.mean(values)
            std = np.std(values)
            self.vital['mean'][self.vital_name]=mean
            self.vital['std'][self.vital_name]=std
            self.mean_vital[i] = mean
            self.std_vital[i] = std

    def construct_lab(self):
        """
        retuen mean&std value for lab signals
        """
        self.lab = {}
        self.blood = {}

        for i in self.file_names_cohort:

            #i = i + '.csv'
            try:
                self.read_table_cohort(i)
                print(i)
                #if not 1 in self.vital_table[:, -1]:
                for j in range(self.lab_length):
                    index = j + 6
                    lab_name = self.lab_cohort.columns[index]
                    single_mean = np.mean([value for value in self.lab_cohort_ar[:, index] if not np.isnan(value)])
                    if not np.isnan(single_mean):
                        self.lab.setdefault(lab_name, []).append(single_mean)

                for j in range(self.blood_length):
                    index = j + 7
                    blood_name = self.blood_cohort.columns[index]
                    single_mean = np.mean([value for value in self.blood_cohort_ar[:, index] if not np.isnan(value)])
                    if not np.isnan(single_mean):
                        self.blood.setdefault(blood_name, []).append(single_mean)

            except:
                continue

        for i in self.file_names_control:

            #i = i + '.csv'
            try:
                self.read_table_control(i)
                print(i)
                #if not 1 in self.vital_table[:, -1]:
                for j in range(self.lab_length):
                    index = j + 6
                    lab_name = self.lab_control.columns[index]
                    single_mean = np.mean([value for value in self.lab_control_ar[:, index] if not np.isnan(value)])
                    if not np.isnan(single_mean):
                        self.lab.setdefault(lab_name, []).append(single_mean)

                for j in range(self.blood_length):
                    index = j + 7
                    blood_name = self.blood_control.columns[index]
                    single_mean = np.mean([value for value in self.blood_control_ar[:, index] if not np.isnan(value)])
                    if not np.isnan(single_mean):
                        self.blood.setdefault(blood_name, []).append(single_mean)

            except:
                continue

    def compute_mean_std_lab(self):
        self.lab['mean'] = {}
        self.lab['std'] = {}
        self.mean_lab = np.zeros(self.lab_length)
        self.std_lab = np.zeros(self.lab_length)
        #index = 0
        for i in range(self.lab_length):
            index = i+6
            self.lab_name = self.lab_cohort.columns[index]
            #upper_per = np.percentile(self.vital[vital_name], self.cost_upper_lab)
            #lower_per = np.percentile(self.vital[vital_name], self.cost_lower_lab)
            values = [value for value in self.lab[self.lab_name]]
            mean = np.mean(values)
            std = np.std(values)
            self.lab['mean'][self.lab_name]=mean
            self.lab['std'][self.lab_name]=std
            self.mean_lab[i] = mean
            self.std_lab[i] = std

        self.blood['mean'] = {}
        self.blood['std'] = {}
        self.mean_blood = np.zeros(self.blood_length)
        self.std_blood = np.zeros(self.blood_length)
        # index = 0
        for i in range(self.blood_length):
            index = i + 7
            self.blood_name = self.blood_cohort.columns[index]
            # upper_per = np.percentile(self.vital[vital_name], self.cost_upper_lab)
            # lower_per = np.percentile(self.vital[vital_name], self.cost_lower_lab)
            if not self.blood_name in self.blood.keys():
                mean = 0
                std = 0
            else:
                values = [value for value in self.blood[self.blood_name]]
                mean = np.mean(values)
                std = np.std(values)
            self.blood['mean'][self.blood_name] = mean
            self.blood['std'][self.blood_name] = std
            self.mean_blood[i] = mean
            self.std_blood[i] = std


    def split_train_test(self):
        #self.train_num = np.int(np.floor(self.data_length * self.train_percent))
        self.train_num = 3000
        self.train_set_cohort = self.file_names_cohort[0:500]
        self.train_set_control = self.file_names_control[0:5000]
        self.test_set_cohort = self.file_names_cohort[500:700]
        self.test_set_control = self.file_names_control[5000:7000]


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
    #read_d_mimic.split_train_test()
    tb = tradition_b(read_d_mimic)
    #dp = dp_appro(read_d_mimic)
    seq = seq_cl(read_d_mimic)