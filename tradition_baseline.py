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

class tradition_baseline():
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