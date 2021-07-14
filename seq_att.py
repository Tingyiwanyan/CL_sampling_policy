from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.neighbors import NearestNeighbors
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats
import tensorflow as tf
import numpy as np
from sklearn.utils import resample
from seq_cl import seq_cl
import random


class seq_cl_att(seq_cl):
    def __init__(self, read_d):
        seq_cl.__init__(self,read_d)