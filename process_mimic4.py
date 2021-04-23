import numpy as np
import random
import math
import time
import pandas as pd
import json
from os import listdir
from hierarchy_cl_learning import hier_cl


class read_data_mimic():
    """
    Loading data
    """
    def __init__(self):
        self.file_path_sepsis = '/home/tingyi/MIMIC_4_DATA/sepsis_patients/'
        self.file_names_sepsis = listdir(self.file_path_sepsis)
        self.file_path_non_sepsis = '/home/tingyi/MIMIC_4_DATA/non_sepsis_patients/'
        self.file_names_non_sepsis = listdir(self.file_path_non_sepsis)

    def read_table_sepsis(self,name):
        """
        extracting sepsis feature
        """
        patient_table = np.array(pd.read_table(name, sep="|"))
        extract_feature_vital = patient_table[1:-1,np.r_[2,9:23]]
        extract_feature_comor = np.sum(patient_table[1:-1,23:39],axis=0)
        extract_feature_comor = extract_feature_comor.astype(bool).astype(int)

        return extract_feature_vital,extract_feature_comor

    def read_table_non_sepsis(self,name):
        """
        extracting non_sepsis feature
        """
        patient_table = np.array(pd.read_table(name, sep="|"))
        extract_feature_vital = patient_table[1:-1, np.r_[2, 8:22]]
        extract_feature_comor = np.sum(patient_table[1:-1,22:38],axis=0)
        extract_feature_comor = extract_feature_comor.astype(bool).astype(int)

        return extract_feature_vital,extract_feature_comor

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


if __name__ == "__main__":
    read_d_mimic = read_data_mimic()
    read_d_mimic.split_train_test()
    h_cl = hier_cl(read_d_mimic)
