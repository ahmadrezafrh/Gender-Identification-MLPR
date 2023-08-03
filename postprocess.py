# -*- coding: utf-8 -*-

# =============================================================================
# Author        : Ahmadreza Farmahini Farahani
# Created Date  : 2023/4
# Project       : This project is developed for "Machine Learning and Pattern Recognition" course
# Supervisor    : Prof. Sandro Cumani
# =============================================================================

import numpy as np
from utils import vrow
from utils import single_fold
from classifiers import LR

    

class DCF:
    def __init__(self, cfn, cfp, scores, labels):
        self.scores = vrow(scores)[0]
        self.cfn = cfn
        self.cfp = cfp
        self.labels = labels
        
    def assign_labels(self, pi, th=None):
        if th is None:
            th = -np.log(pi*self.cfn)+np.log((1-pi)*self.cfp)
        p = self.scores > th
        return np.int32(p)
    
    def compute_conf_matrix_binary(self, pred):
        c = np.zeros((2, 2))
        c[0,0] = ((pred==0) * (self.labels[0]==0)).sum()
        c[0,1] = ((pred==0) * (self.labels[0]==1)).sum()
        c[1,0] = ((pred==1) * (self.labels[0]==0)).sum()
        c[1,1] = ((pred==1) * (self.labels[0]==1)).sum()
        return c
    
    def compute_emp_bayes_binary(self, cm, pi):
        fnr = cm[0,1] / (cm[0,1] + cm[1,1])
        fpr = cm[1,0] / (cm[0,0] + cm[1,0])
        return pi * self.cfn * fnr + (1-pi) * self.cfp * fpr
    
    def compute_normalized_emp_bayes(self, cm, pi):
        emp_bayes = self.compute_emp_bayes_binary(cm, pi)
        return emp_bayes / min(pi*self.cfn, (1-pi)*self.cfp)
    
    def compute_act_dcf(self, pi, th=None):
        pred = self.assign_labels(pi, th=th)
        cm = self.compute_conf_matrix_binary(pred)
        return self.compute_normalized_emp_bayes(cm, pi)
    
    def compute_min_dcf(self, pi):
        t = np.array(self.scores)
        t.sort()
        np.concatenate([np.array([-np.inf]), t, np.array([np.inf])])
        dcflist = []
        for _th in t:
            dcflist.append(self.compute_act_dcf(pi, th = _th))
        dcfarray = np.array(dcflist)
        return dcfarray.min()
    
    def compute_min_dcf_threshold(self, pi):
        t = np.array(self.scores)
        t.sort()
        np.concatenate([np.array([-np.inf]), t, np.array([np.inf])])
        dcflist = []
        for _th in t:
            dcflist.append(self.compute_act_dcf(pi, th = _th))
        dcfarray = np.array(dcflist)
        return t[np.where(dcfarray == dcfarray.min())]
    
    def bayes_error_plot(self, p_array, min_cost=False, scores=None):
        y = []
        if scores is not None:
            self.scores=scores
        for p in p_array:
            pi = 1 / (1 + np.exp(-p))
            if min_cost:
                y.append(self.compute_min_dcf(pi))
            else:
                y.append(self.compute_act_dcf(pi))
        return np.array(y)
    
    def calibrated_plot(self, p_array, th):
        y = []
        for p in p_array:
            pi = 1 / (1 + np.exp(-p))
            y.append(self.compute_act_dcf(pi, th=th))
        return np.array(y)
    
    def det_plot(self):
        fnrs = []
        fprs = []
        sorted_threasholds = self.scores.copy()
        sorted_threasholds.sort()
        for th in sorted_threasholds:
            pred = self.assign_labels(pi=0.5, th=th)
            cm = self.compute_conf_matrix_binary(pred)
            fnr = cm[0,1] / (cm[0,1] + cm[1,1]) * 100
            fpr = cm[1,0] / (cm[0,0] + cm[1,0]) * 100
            if fpr<5 and fnr<5:
                continue
            fnrs.append(fnr)
            fprs.append(fpr)
        return fnrs, fprs
    
        
    
class calibration:
    def __init__(self, pi=0.5, lambd=0.00001):
        self.pi=pi
        self.lambd=lambd
        self.model= LR()
    
    def fit(self, scores, labels):
        self.model.fit(scores, labels, self.lambd, pi=self.pi, balance=True)
    
    def predict(self, scores):
        return self.model.predict(scores)



class fusion:
    def __init__(self, pi=0.5, lambd=0.00001):
        self.pi=pi
        self.lambd=lambd
        self.model= LR()
    
    def fit(self, scores, labels):
        self.model.fit(scores, labels, self.lambd, pi=self.pi, balance=True)
    
    def predict(self, scores):
        return self.model.predict(scores)
 
    
def thresh_calibration(scores_train, labels_train, scores_test, labels_test, pi):
    train_dcf = DCF(1, 1, scores_train, labels_train)
    valid_dcf = DCF(1, 1, scores_test, labels_test)
    min_thresh = train_dcf.compute_min_dcf_threshold(pi)
    emprical_min = valid_dcf.compute_min_dcf(pi)
    emprical_act = valid_dcf.compute_act_dcf(pi, th=-np.log(pi/(1-pi)))
    threshold_act = valid_dcf.compute_act_dcf(pi, th=min_thresh)
    
    return emprical_min, emprical_act, threshold_act
