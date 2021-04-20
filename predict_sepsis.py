import numpy as np
import random
import math
import time
import pandas as pd
import json
from os import listdir
from cl_learning import knn_cl

class read_data():
    """
    Loading data
    """
    def __init__(self):
        self.file_path = '/home/tingyi/adver_cl/training/'
        self.file_names = listdir(self.file_path)
        self.train_prop = 0.7
        self.test_prop = 0.3
        self.total_size = 3000
        self.sepsis_group = []
        self.non_sepsis_group = []
        self.total_data = []
        self.total_data_label = []
        self.total_gender_label = []
        self.test_data = []
        self.female_group = []
        self.male_group = []
        self.median_vital_signal = np.zeros(7)
        self.std_vital_signal = np.zeros(7)
        self.median_vital_signal_female = np.zeros(7)
        self.std_vital_signal_female = np.zeros(7)
        self.median_vital_signal_male = np.zeros(7)
        self.std_vital_signal_male = np.zeros(7)
        self.dic_item = {}
        self.dic_item_female = {}
        self.dic_item_male = {}
        self.dic_item_sepsis = {}
        self.dic_item_non_sepsis = {}



    def generate_lib(self):
        count = 0
        for i in self.file_names:
            if count > self.total_size:
                break
            name = self.file_path+i
            patient_table = np.array(pd.read_table(name, sep="|"))

            if 1 in patient_table[:,40]:
                sepsis_on_set_time = np.where(patient_table[:, 40] == 1)[0][0]
                if sepsis_on_set_time < 5:
                    continue
                else:
                    self.sepsis_group.append(i)
                    self.total_data.append(i)
                    self.total_data_label.append(1)
            else:
                self.non_sepsis_group.append(i)
                self.total_data.append(i)
                self.total_data_label.append(0)

            if patient_table[0,35] == 0:
                self.female_group.append(i)
                self.total_gender_label.append(0)
            else:
                self.male_group.append(i)
                self.total_gender_label.append(1)

            for j in range(7):
                entry_mean = np.mean([l for l in patient_table[:, j] if not np.isnan(l)])
                if np.isnan(entry_mean):
                    continue
                self.dic_item.setdefault(j,[]).append(entry_mean)
                if patient_table[0, 35] == 0:
                    self.dic_item_female.setdefault(j,[]).append(entry_mean)
                else:
                    self.dic_item_male.setdefault(j,[]).append(entry_mean)

                if 1 in patient_table[:, 40]:
                    self.dic_item_sepsis.setdefault(j,[]).append(entry_mean)
                else:
                    self.dic_item_non_sepsis.setdefault(j, []).append(entry_mean)


            count += 1


    def compute_ave_vital(self):
        for j in range(7):
            #median = np.median([i for i in self.mean_vital[:,j] if not np.isnan(i)])
            #std = np.std([i for i in self.mean_vital[:,j] if not np.isnan(i)])
            median = np.median(self.dic_item[j])
            std = np.std(self.dic_item[j])
            self.median_vital_signal[j] = median
            self.std_vital_signal[j] =std


    def divide_train_test(self):
        data_length = len(self.total_data)
        self.train_num = np.int(np.floor(data_length*self.train_prop))
        self.train_data = self.total_data[0:self.train_num]
        self.train_data_label = self.total_data_label[0:self.train_num]
        self.train_data_gender_label = self.total_gender_label[0:self.train_num]
        self.test_data = self.total_data[self.train_num:data_length]
        self.test_data_label = self.total_data_label[self.train_num:data_length]
        self.test_data_gender_label = self.total_gender_label[self.train_num:data_length]
        self.train_sepsis_group = list(np.array(self.train_data)[np.where(np.array(self.train_data_label)==1)[0]])
        self.train_non_sepsis_group = list(np.array(self.train_data)[np.where(np.array(self.train_data_label)==0)[0]])
        self.train_female_group = list(np.array(self.train_data)[np.where(np.array(self.train_data_gender_label)==0)[0]])
        self.train_male_group = list(np.array(self.train_data)[np.where(np.array(self.train_data_gender_label)==1)[0]])
        self.test_female_group = list(np.array(self.test_data)[np.where(np.array(self.test_data_gender_label)==0)[0]])
        self.test_male_group = list(np.array(self.test_data)[np.where(np.array(self.test_data_gender_label)==1)[0]])

        self.train_female_group_label = list(
            np.array(self.train_data_label)[np.where(np.array(self.train_data_gender_label)==0)[0]])
        self.train_male_group_label = list(
            np.array(self.train_data_label)[np.where(np.array(self.train_data_gender_label) == 1)[0]])
        self.test_female_group_label = list(
            np.array(self.test_data_label)[np.where(np.array(self.test_data_gender_label) == 0)[0]])
        self.test_male_group_label = list(
            np.array(self.test_data_label)[np.where(np.array(self.test_data_gender_label) == 1)[0]])






if __name__ == "__main__":
    read_d = read_data()
    read_d.generate_lib()
    read_d.compute_ave_vital()
    read_d.divide_train_test()
    cl_sample = knn_cl(read_d)

