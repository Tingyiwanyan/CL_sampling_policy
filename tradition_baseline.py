import math
import copy
from itertools import groupby
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import tensorflow as tf
import numpy as np

class tradition_b():
    """
    Create dynamic HGM model
    """

    def __init__(self, read_d):
        self.read_d = read_d
        self.train_data = read_d.train_set
        self.test_data = read_d.test_set
        self.length_train = len(self.train_data)
        self.length_test = len(self.test_data)
        self.batch_size = 512
        self.vital_length = 9
        self.lab_length = 25
        self.epoch = 6
        self.lr = SGDClassifier(loss="log")
        self.rf = RandomForestClassifier(max_depth=10,random_state=0)

    def aquire_batch_data(self, starting_index, data_set,length):
        self.one_batch_data = np.zeros((length,self.vital_length+self.lab_length))
        self.one_batch_logit = np.zeros(length)
        for i in range(length):
            name = data_set[starting_index+i]
            self.read_d.return_data_dynamic(name)
            one_data = self.read_d.one_data_tensor
            self.one_batch_data[i,:] = np.sum(one_data,1)
            self.one_batch_logit[i] = self.read_d.logit_label


    def logistic_regression(self):
        self.iteration = np.int(np.floor(np.float(self.length_train) / self.batch_size))
        for i in range(self.epoch):
            for j in range(self.iteration):
                print(j)
                self.aquire_batch_data(j*self.iteration,self.train_data,self.batch_size)
                self.lr.partial_fit(self.one_batch_data,self.one_batch_logit,classes=np.unique(self.one_batch_logit))
                #print(self.lr.score(self.one_batch_data,self.one_batch_logit))
                #print(roc_auc_score(self.one_batch_logit,self.lr.predict_proba(self.one_batch_data)[:,1]))

            self.test_logistic_regression()

    def test_logistic_regression(self):
        self.aquire_batch_data(0,self.test_data,self.length_test)
        #print(self.lr.score(self.one_batch_data,self.one_batch_logit))
        print(roc_auc_score(self.one_batch_logit, self.lr.predict_proba(self.one_batch_data)[:, 1]))


    def random_forest(self):
        self.aquire_batch_data(0, self.train_data, self.batch_size*20)
        self.rf.fit(self.one_batch_data, self.one_batch_logit)
                # print(self.lr.score(self.one_batch_data,self.one_batch_logit))
                # print(roc_auc_score(self.one_batch_logit,self.lr.predict_proba(self.one_batch_data)[:,1]))

        self.test_random_forest()

    def test_random_forest(self):
        self.aquire_batch_data(0,self.test_data,self.length_test)
        #print(self.lr.score(self.one_batch_data,self.one_batch_logit))
        print(roc_auc_score(self.one_batch_logit, self.rf.predict(self.one_batch_data)))


