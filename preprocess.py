# -*- coding: utf-8 -*-

# =============================================================================
# Created By    : Ahmadreza Farmahini Farahani
# Created Date  : 2023/4
# Project       : This project is developed for "Machin Learnin and Pattern Recognition" course
# Supervisor    : Prof. Sandro Cumani
# =============================================================================

import numpy as np
import scipy
import scipy.linalg
import scipy.stats
from utils import vcol
from utils import compute_sw
from utils import compute_sb

class normalization:
    def fit(self, x):
        self.mean = np.mean(x, axis=1, keepdims=True)
        self.std = np.std(x, axis=1, keepdims=True)
        
    def transform(self, x):
        normalized_data = (x-self.mean)/self.std
        return normalized_data
    
    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)



class gaussianization:
    
    def fit(self, x):
        self.x_train = x
    def transform(self, x):
        self.rank_test = np.zeros(x.shape)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                self.rank_test[i][j] = (self.x_train[i] < x[i][j]).sum() + 1
        self.rank_test /= (self.x_train.shape[1] + 2)
        return scipy.stats.norm.ppf(self.rank_test)
    
    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

class PCA:
    """
    This class takes data with shape (features,samples) and returns
    reduced data with shape (n_components, samples) also it returns
    variance ratio in percentage and eighenvectors.

    :param x: array(features,samples)
    :type x: numpy.array
    :return: dp(n_components, samples), variance ratio(features,) and
    eighenvectors(features, features)
    """
    def fit(self, x, n_components):
        
        mu = vcol(x.mean(1))
        dc = x - mu
        c = np.dot(dc, dc.T)/x.shape[1]
        evecs, evals, Vt = np.linalg.svd(c)
        self.p = evecs[:, 0:n_components]
        evals_ratio = (evals/evals.sum())*100
        return evals_ratio, evals, evecs    

    def transform(self, x):
        return np.dot(self.p.T, x)
    
    def fit_transform(self, x, n_components):
        self.fit(x, n_components)
        return self.transform(x)



class LDA():
    def fit(self, x, y, n_components):
        sb = compute_sb(x,y)
        sw = compute_sw(x,y)
        evals, evecs = scipy.linalg.eigh(sb,sw)
        evecs = evecs[:, ::-1]
        self.p = evecs[:, 0:n_components]
        evals = evals[::-1]
        return evecs, evals
    
    def transform(self, x):
        return np.dot(self.p.T, x)
    
    def fit_transform(self, x, y, n_components):
        self.fit(x, y, n_components)
        return self.transform(x)
    
    
