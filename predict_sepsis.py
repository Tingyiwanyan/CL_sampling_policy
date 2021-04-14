import numpy as np
import random
import math
import time
import pandas as pd
from scipy.stats import iqr
import json
from os import listdir

class read_data():
    """
    Loading data
    """
    def __init__(self):
        self.file_path = '/home/tingyi/adver_cl/training/'
        self.file_names = listdir(self.file_path)
        self.train_data_size = 6000
        self.test_data_size = 500
        self.sepsis_group = []
        self.non_sepsis_group = []
        self.total_data = []
        self.total_data_label = []
        self.test_data = []
        self.female_group = []
        self.male_group = []
        self.mean_vital = np.zeros((self.train_data_size,7))
        self.mean_vital_single = np.zeros(7)
        self.std_vital_single = np.zeros(7)
        self.dic_item = {}



    def generate_lib(self):
        for i in self.file_names:
            name = self.file_path+i
            patient_table = np.array(pd.read_table(name, sep="|"))

            if 1 in patient_table[:,40]:
                sepsis_on_set_time = np.where(patient_table[:, 40] == 1)
                if sepsis_on_set_time < 3:
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
            else:
                self.male_group.append(i)

            for j in range(7):
                entry_mean = np.mean([l for l in patient_table[:, j] if not np.isnan(l)])
                if np.isnan(entry_mean):
                    continue
                self.dic_item.setdefault(j,[]).append(entry_mean)


    def compute_ave_vital(self):
        for j in range(7):
            #median = np.median([i for i in self.mean_vital[:,j] if not np.isnan(i)])
            #std = np.std([i for i in self.mean_vital[:,j] if not np.isnan(i)])
            median = np.median(self.dic_item[j])
            std = np.std(self.dic_item[j])
            self.median_vital_single[j] = median
            self.std_vital_single[j] =std



if __name__ == "__main__":
    read_d = read_data()

