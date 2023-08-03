# -*- coding: utf-8 -*-

# =============================================================================
# Author        : Ahmadreza Farmahini Farahani
# Created Date  : 2023/4
# Project       : This project is developed for "Machine Learning and Pattern Recognition" course
# Supervisor    : Prof. Sandro Cumani
# =============================================================================
import pickle
import itertools
import json
import os
import numpy as np
import matplotlib.pyplot as plt


def read_data(file_name):
    
    """
    This function will receive a file_name and number of features
    available in .txt files and returns two numpy arrays with a shape of
    (number_of_lines, n_features) and (number_of_lines, 1), the first one
    is features and the second one is labels.

    :param file_name: The first number to add
    :param n_features: The second number to add
    :type file_name: n_features
    :type n_features: int
    :return: feature array, label array

    :Example:

    >>> x, y = read_data(file_name= "Test.txt", n_features= 12)
    >>> print(x.shape)
    (4000,12)
    >>> print(x.shape)
    (4000, 1) 
    
    .. seealso::
    .. warnings:: This maybe not useful for other .txt files
    .. note:: We could have used other methods for reading the data
    """
    
    with open(file_name) as data:
        
        lines = data.readlines()
        x = np.zeros([len(lines),len(lines[0].split(","))-1])
        y = np.zeros([len(lines),1])
        for c, line in enumerate(lines):
            split = line.split(",")
            x[c] = split[0:-1]
            y[c] = split[-1]
        return x.T, y.T


def vcol(x):
    """
    This function takes an array of shape (n,) and returns an
    array of shape (n,1)

    :param x: data array(n,)
    :type x: numpy.array
    :return: numpy.ndarray array with shape (n,1)
    """
    return x.reshape((x.size,1))


def vrow(x):
    """
    This function takes an array of shape (n,) and returns an
    array of shape (n,1)

    :param x: data array(n,)
    :type x: numpy.ndarray
    :return: numpy.ndarray with the shape of (n,1)
    """
    return x.reshape((1,x.size))

def plot_hist(x, y, n_features, save_path, show=False, n_rows_hist='auto', n_cols=4, max_features=20, fig_size=(30, 20)):
    
    
    """
    This function will show histograms of features in dataset.

    :param x: array of samples (n_features, samples)
    :param y: array of lables (1, samples)
    :param n_features: number of features
    :type x: numpy.ndarray
    :type y: numpy.ndarray
    :type n_features: int

    :Example:

    >>> plot_hist(x, y, n_features)
    """
    
    if n_rows_hist == 'auto':
        for i in range(int(max_features/n_cols)):
            if i*n_cols<n_features:
                continue
            else:
                n_rows = i
                break
            
    x0 = x[:, y[0]==0]
    x1 = x[:, y[0]==1]
    plt.figure(figsize=fig_size, dpi=100)
    for n in range(n_features):

        plt.xlabel(f'feature_{n}')
        plt.style.use('seaborn-whitegrid')
        plt.subplot(n_rows,4, n+1)
        plt.hist(x0[n, :], bins = 50, density = True, facecolor = 'darkkhaki', edgecolor='#169acf', label= f'class_{0}', alpha=0.6)
        plt.hist(x1[n, :], bins = 50, density = True, facecolor = 'darkcyan', edgecolor='#169acf', label= f'class_{1}', alpha=0.6)
        plt.legend()
        plt.tight_layout()
            
    plt.savefig(save_path)
    if show:
        plt.show()

def plot_scatter(x, y, n_features):
    
    """
    This function will show scatters of features in dataset.

    :param x: array of samples (n_features, samples)
    :param y: array of lables (1, samples)
    :param n_features: number of features
    :type x: numpy.ndarray
    :type y: numpy.ndarray
    :type n_features: int

    :Example:

    >>> plot_scatter(x, y, n_features)
    """
    
    x0 = x[:, y[0]==0]
    x1 = x[:, y[0]==1]
    print(x0.shape)
    print(np.zeros((1,x.shape[1])))
    
    if n_features == 1:
            plt.figure()
            plt.xlabel('single_feature')
            plt.ylabel('class')
            plt.style.use('seaborn-whitegrid')
            plt.scatter(x0[0], np.zeros((1,x0.shape[1])), label= 'class_0')
            plt.scatter(x1[0], np.ones((1,x1.shape[1])), label= 'class_1')
            
            plt.legend()
            plt.tight_layout()
            
    for n in range(n_features):
        for m in range(n_features):
            if m == n:
                continue
            if n_features==12:
                plt.figure()
                plt.xlabel(f'feature_{n}')
                plt.ylabel(f'feature_{m}')
                plt.scatter(x0[n, :], x0[m, :], facecolor='#2ab0ff', edgecolor='#169acf', label= f'class_{0}')
                plt.scatter(x1[n, :], x1[m, :], facecolor='#2ab0ff', edgecolor='#169acf',label= f'class_{1}')
                
                plt.legend()
                plt.tight_layout()
        
    plt.show()


def calculate_cov(x):
    """
    This function will calculate covariance from the samples.

    :param x: array of samples (n_features, samples)
    :type x: numpy.ndarray
    :return: covariance matrix of the samples with shape(n_features, n_features)
    """
    
    mu = calculate_mean(x)
    cov = np.dot((x-mu), (x-mu).T)/x.shape[1]
    return cov

def calculate_mean(x):
    """
    This function will calculate mean from the samples.

    :param x: array of samples (n_features, samples)
    :type x: numpy.ndarray
    :return: mean matrix of the features with shape(n_features, 1)
    """
    return vcol(x.mean(1))

def compute_sb(x,y):
    """
    This function will calculate SB from the data to be used in
    gausian classifiers

    :param x: array of samples (n_features, samples)
    :param y: array of lables (1, samples)
    :type x: numpy.ndarray
    :type y: numpy.ndarray
    :return: SB with the shape of (n_features, n_features)
    """
    
    
    SB = 0
    muG = calculate_mean(x)
    for i in set(list(y[0])):
        D = x[:, y[0]==i]
        mu = calculate_mean(D)
        SB += D.shape[1]*np.dot((mu - muG), (mu - muG).T)
    return SB/x.shape[1]

def compute_sw(x,y):
    """
    This function will calculate SW from the data to be used in
    gausian classifiers

    :param x: array of samples (n_features, samples)
    :param y: array of lables (1, samples)
    :type x: numpy.ndarray
    :type y: numpy.ndarray
    :return: SW with the shape of (n_features, n_features)
    """
    SW = 0
    for i in set(list(y[0])):
        SW += (y[0]==i).sum() * calculate_cov(x[:, y[0]==i])
    return SW/x.shape[1]


def logpdf_GAU_ND(x, mu, c):
    """
    This function liklihood for each sample

    :param x: array of samples (n_features, samples)
    :param mu: mean of the features (n_features, 1)
    :param c: covariance matrix of the features (n_features, n_features)
    :type x: numpy.ndarray
    :type mu: numpy.ndarray
    :type c: numpy.ndarray

    :return: an array of likelihood probabilities (samples,)
    """
    
    
    
    p = np.linalg.inv(c)
    res = -0.5 * x.shape[0] * np.log(2*np.pi)
    res += 0.5 * np.linalg.slogdet(p)[1]
    res += -0.5 * (np.dot(p, (x-mu)) * (x-mu)).sum(0)
    return res


    
def loglikelihood(x, mu, c):
    """
    This function will calculate sum of log likelihoods

    :param x: array of samples (n_features, samples)
    :param mu: mean of the features (n_features, 1)
    :param c: covariance matrix of the features (n_features, n_features)
    :type x: numpy.ndarray
    :type mu: numpy.ndarray
    :type c: numpy.ndarray

    :return: an float negative number that determines overall likelihood
    """
    return logpdf_GAU_ND(x, mu, c).sum()

def likelihood(x, mu, c):
    """
    This function will calculate sum of likelihoods

    :param x: array of samples (n_features, samples)
    :param mu: mean of the features (n_features, 1)
    :param c: covariance matrix of the features (n_features, n_features)
    :type x: numpy.ndarray
    :type mu: numpy.ndarray
    :type c: numpy.ndarray

    :return: a float negative number that determines overall likelihood
    """
    
    y = np.exp(logpdf_GAU_ND(x, mu, c).sum())
    return y


def change_preds(preds):
    """
    This function will change the predictions format to be comperable with
    tests. the predicted array is an array of lists. We change it to an array 
    of floats.

    :param preds: array of samples (1, samples)
    :type preds: numpy.ndarray

    :return: an array of floats whith the shape of (1, samples)
    """
    preds_list = []
    for i in range(preds.shape[0]):
        preds_list.append(int(list(preds)[i]))
    return np.array([preds_list])

def calculate_posterior(preds):
    preds_list = []
    for i in range(preds.shape[1]):
        preds_list.append(np.log(preds[1][i]/preds[0][i]))
    return np.array([preds_list])


def k_folds_balanced(num_folds, x, y, calsses=[0, 1]):
    folds_x = []
    folds_y = []
    folds = []
    classes = calsses
    num_folds = num_folds
    num_per_class = x.shape[1]/len(classes)
    num_samples = x.shape[1]/num_folds
    num_per_class_fold = num_samples/len(classes)
    indexes = np.arange(num_per_class, dtype='int64')
    np.random.seed(42)
    np.random.shuffle(indexes)
    indexes = np.reshape(indexes, (num_folds,int(num_per_class_fold)))

    for k in range(num_folds):
        x_fold = []
        y_fold = []
        for i in classes:
            class_indexes = np.where(y==i)[1][indexes[k]]
            x_fold.append(x[:,class_indexes])
            y_fold.append(y[:,class_indexes])
            
        folds_x.append(np.hstack(x_fold))
        folds_y.append(np.hstack(y_fold))
    
    for i in range(len(folds_x)):
        x = folds_x.copy()
        y = folds_y.copy()
        x_test = x[i]
        x.pop(i)
        x_train = np.hstack(x)
        
        y_test = y[i]
        y.pop(i)
        y_train = np.hstack(y)
        folds.append((x_train, y_train, x_test, y_test))
        
    return folds




def k_folds(num_folds, x, y):
    folds_x = []
    folds_y = []
    folds = []
    num_folds = num_folds
    num_samples = int(x.shape[1]/num_folds)
    indexes = np.random.permutation(x.shape[1])
    np.random.shuffle(indexes)


    for k in range(num_folds):
        folds_x.append(x[:,indexes[k*num_samples:(k+1)*num_samples]])
        folds_y.append(y[:,indexes[k*num_samples:(k+1)*num_samples]])
    
    for i in range(len(folds_x)):
        x = folds_x.copy()
        y = folds_y.copy()
        x_test = x[i]
        x.pop(i)
        x_train = np.hstack(x)
        
        y_test = y[i]
        y.pop(i)
        y_train = np.hstack(y)
        folds.append((x_train, y_train, x_test, y_test))
        
    return folds




def single_fold(split_ratio, x, y):
    num_samples = int(x.shape[1]*split_ratio)
    indexes = np.random.permutation(x.shape[1])
    x_train = x[:,indexes[0 : num_samples]]
    y_train = y[:,indexes[0 : num_samples]]
    x_test = x[:,indexes[num_samples : ]]
    y_test = y[:,indexes[num_samples : ]]
    return (x_train, y_train, x_test, y_test)


def min_results(results):
    min_dcf = min(results, key=lambda x:x['min_dcf'][0.5])
    min_models= [x for x in results if x['min_dcf'][0.5]==min_dcf['min_dcf'][0.5]]
    return min_dcf['min_dcf'][0.5], min_models


def load_results(path):
    with open(path, "rb") as fp:
        results = pickle.load(fp)
    return results



def create_params(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))



class load_configue:
    def __init__(self, path):
        with open(path) as model:
            self.configue = json.load(model)
        
    def drop(self, keys=['n_folds', 'split_ratio', 'train_path', 'save_path']):
        for key in keys: self.configue.pop(key)
        return self.configue

    def add(self, key, value):
        self.configue[key] = value
    
    def get_parameters(self):
        return self.configue



def check_path(path):
    if os.path.exists(path):
        os.remove(path)
