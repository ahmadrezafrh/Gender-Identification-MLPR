# -*- coding: utf-8 -*-

# =============================================================================
# Created By    : Ahmadreza Farmahini Farahani
# Created Date  : 2023/4
# Project       : This project is developed for "Machin Learnin and Pattern Recognition" course
# Supervisor    : Prof. Sandro Cumani
# =============================================================================


import numpy as np
import random
import scipy.linalg
import scipy.special
import scipy.optimize
from utils import calculate_mean
from utils import calculate_cov
from utils import logpdf_GAU_ND
from utils import vrow
from utils import vcol



class MVG:
    
    def __init__(self, model="full", tied=False):
        self. h = {}
        self.model = model
        self.tied = tied
        
    def fit(self, x_train, y_train):    
        
        if self.tied:
            c = 0
            for lab in [0,1]:
                c += calculate_cov(x_train[:, y_train[0]==lab])
            
            for lab in [0,1]:
                mu = calculate_mean(x_train[:, y_train[0]==lab])
                if self.model=="diagonal":
                    c = np.diag(np.diag(c))
                self.h[lab] = (mu, c/2) 
                
        else:
            for lab in [0,1]:
                mu = calculate_mean(x_train[:, y_train[0]==lab])
                c = calculate_cov(x_train[:, y_train[0]==lab])
                if self.model=="diagonal":
                    c = np.diag(np.diag(c))
                self.h[lab] = (mu, c) 
                
        return self.h
            
    def predict(self, x_test, pi=0.5):
        
        logSJoint = np.zeros((2, x_test.shape[1]))
        
        for lab in [0,1]:
            mu, c = self.h[lab]
            logSJoint[lab, :] = logpdf_GAU_ND(x_test, mu, c).ravel()

        logSMarginal = scipy.special.logsumexp(logSJoint, axis=0)
        logPost = logSJoint - vrow(logSMarginal)
        scores = logPost[1] - logPost[0] - np.log(pi/(1-pi))
        return scores
        
    def error(self, y_pred, y_test, th=None):
        if th is None:
            y_pred = y_pred>0
        else :
            y_pred = y_pred>th
        return (y_test.shape[1]-(y_pred == y_test).sum())/y_test.shape[1]
        
        
    def accuracy(self, y_pred, y_test, th=None):
        return 1-self.error(y_pred, y_test, th)









class LR:
    
    def __init__(self, model_type='linear'):
        self.model_type = model_type
        
    def fit(self, x_train, y_train, lamb, pi=0.5, balance=False):
        self.balance = balance
        self.pi = pi
        x_train = self._features_expansion(x_train) if self.model_type == 'quadratic' else x_train
        

        self.nt = x_train[:,y_train[0] == 1].shape[1]
        self.nf = x_train[:,y_train[0] == 0].shape[1]
        
        _logreg_obj = self._logreg_obj_wrap(x_train, y_train, lamb)
        self._v, self._J, self._d = scipy.optimize.fmin_l_bfgs_b(_logreg_obj,
                                                                 np.zeros(x_train.shape[0]+1),
                                                                 approx_grad=True)
        self._w = self._v[0:x_train.shape[0]]
        self._b = self._v[-1]
        return self._w, self._b
    
    def predict(self, x_test, pi=0.5):
        x_test = self._features_expansion(x_test) if self.model_type == 'quadratic' else x_test
        preds = (np.dot(self._w.T, x_test) + self._b)
        logodds = np.log(self.nt / self.nt)
        return preds - logodds
    
    def _logreg_obj_wrap(self, x, y, l):
        """
        This function is exclusively designed to be used as a cost function 
        for L-BFGS method in logistic regression
    
        :param x: array of samples (n_features, samples)
        :param y: array of lables (1, samples)
        :param l: lambda for regularization
        :type x: numpy.ndarray
        :type y: numpy.ndarray
        :type l: integer
    
        :return: the cost function
        """
        z = y * 2.0 - 1.0
        m = x.shape[0]
        
        def _logreg_obj(v):
            w = vcol(v[0:m])
            b = v[-1]
            cxe = 0
            s = np.dot(w.T, x) + b
            
            if self.balance:
                
                average_risk_1 = np.logaddexp(0, -s[y==1]*z[y==1]).sum()
                average_risk_0 = np.logaddexp(0, -s[y==0]*z[y==0]).sum()
                loss = (self.pi/self.nt) * average_risk_1 + ((1-self.pi)/self.nf) * average_risk_0 + 0.5*l*np.linalg.norm(w)**2
            else:
                cxe = np.logaddexp(0, -s*z).mean()
                loss = cxe + 0.5*l*np.linalg.norm(w)**2
            return loss
        
        
    
        return _logreg_obj
    
    def _features_expansion(self, x_train):
        expansion = []
        for i in range(x_train.shape[1]):
            vec = np.reshape(np.dot(vcol(x_train[:, i]), vcol(x_train[:, i]).T), (-1, 1), order='F')
            expansion.append(vec)
        return np.vstack((np.hstack(expansion), x_train))

    def error(self, y_pred, y_test):
        y_pred = y_pred > 0
        y_test = (y_test == 1)
        err = (y_test.shape[1]-(y_pred == y_test[0]).sum())/y_test.shape[1]
        return err
    
    def accuracy(self, y_pred, y_test):
        y_pred = y_pred > 0
        y_test = (y_test == 1)
        acc = (y_pred == y_test[0]).sum()/y_test.shape[1]
        return acc    
    

    
    
    
class GMM:
    
    def __init__(self, model="full", tied=False):
        
        self.tied = tied
        self.model = model #"diagonal" or "full"

        
    def _logpdf_GAU_ND(self, x, mu, c):
        """
        This function calculate liklihood for each sample
    
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
    
    
    def _GMM_per_sample(self, x, gmm):
        
        g = len(gmm)
        n = x.shape[1]
        s = np.zeros((g, n))
        for i in range(g):
            s[i,:] = self._logpdf_GAU_ND(x, gmm[i][1], gmm[i][2]) + np.log(gmm[i][0])
        return scipy.special.logsumexp(s, axis=0)
    

    def _GMM_EM(self, x, gmm, th=0.01):
        ll_new = None
        ll_old = None
        g = len(gmm)
        n = x.shape[1]
        while ll_old is None or (ll_new-ll_old) >1e-6:
            ll_old = ll_new
            sj = np.zeros((g, n))
            for i in range(g):
                sj[i,:] = self._logpdf_GAU_ND(x, gmm[i][1], gmm[i][2]) + np.log(gmm[i][0])

            sm = scipy.special.logsumexp(sj, axis=0)
            ll_new = sm.sum()/n
            p = np.exp(sj-sm)
            gmm_new = []
            sigma = 0
                
            for i in range(g):
                gamma = p[i, :]
                z = gamma.sum()
                f = (vrow(gamma)*x).sum(1)
                s = np.dot(x, (vrow(gamma)*x).T)
                w = z/n
                mu = vcol(f/z)
                sigma = s/z - np.dot(mu, mu.T)
                
                if self.model=="diagonal":
                    sigma = np.diag(np.diag(sigma))
                    
                u, s, _ = np.linalg.svd(sigma)
                s[s < th] = th
                sigma_new = np.dot(u, vcol(s) * u.T)
                gmm_new.append((w, mu, sigma_new))
                
            if self.tied:
                gmm_opt = []
                tied_cov = sum(np.array(cov[2]) for cov in gmm_new)/len(gmm_new)
                for i in range(g):
                    gmm_opt.append((gmm_new[i][0], gmm_new[i][1], tied_cov))
                gmm_new = gmm_opt
            gmm = gmm_new
        return gmm


    def fit(self, x, y, n_components, th = 0.01, w=1.0):
        mu = calculate_mean(x)
        cov = calculate_cov(x)
        gmm = [(w, mu, cov)]
        new_gmm = 0
        iteration = int(np.log2(n_components)) + 1
        self.gmm_class = {}
        for label in set(list(y[0])):
            gmm_init = gmm
            for i in range(iteration):
                new_gmm = self._GMM_EM(x[:, y[0]==label], gmm_init, th)
                if i<iteration-1:
                    gmm_init = self._gmmlbg(new_gmm, 0.1)
            self.gmm_class[label] = new_gmm
        return self.gmm_class
    
    def predict(self, x, pi=0.5):
        
        scores = np.zeros((2, x.shape[1]))
        scores[0] = self._GMM_per_sample(x, self.gmm_class[0])
        scores[1] = self._GMM_per_sample(x, self.gmm_class[1])
        logsMarginal = scipy.special.logsumexp(scores, axis=0)
        logPost = scores - vrow(logsMarginal)
        scores = logPost[1] - logPost[0] - np.log(pi/(1-pi))
        return scores
            
    def _GMM_generate_random(self, n_components, n_features):
        gmm = []
        for i in range(n_components):
            lst = []
            cov = []
            mean = []
            for n in range(n_features):
                row_cov = []
                for k in range(n_features):
                    if n==k:
                        row_cov.append(1)
                    else:
                        row_cov.append(0)
                cov.append(row_cov)
                
            for n in range(n_features):
                mean.append([random.randint(-3,3)])
            
            lst.append(1/n_features)
            lst.append(mean)
            lst.append(cov)
            gmm.append(lst)
        
        return gmm
    
    def _gmmlbg(self, gmm, alpha):
        g = len(gmm)
        new_gmm = []
        for g in range(g):
            (w, mu, cov) = gmm[g]
            u, s, _ = np.linalg.svd(cov)
            d = u[:, 0:1] * s[0]**0.5 * alpha
            new_gmm.append((w/2, mu - d, cov))
            new_gmm.append((w/2, mu + d, cov))
        return new_gmm




class SVM:
    def __init__(self,kernel="linear", fun=10000):
        self.kernel = kernel
        self.fun = fun
        
        
    def fit(self, x, y, gamma=1, c=1, iteration=10000, k=1, degree=2, bias=1, poly_bias=0, pi=0.5, balance=False):
        
        self.x = x
        self.y = y[0]
        self.gamma = gamma
        self.bias = bias
        self.degree = degree
        self.poly_bias = poly_bias
        self.balance = balance
        
        x_ext = np.vstack([self.x, np.ones((1, self.x.shape[1]))])
        z = np.zeros(self.y.shape)
        z[self.y == 1] = 1
        z[self.y == 0] = -1
        
        
        if self.balance:
            bounds = []
            ct = c*pi/(self.x[:,self.y == 1].shape[1]/self.x.shape[1])
            cf = c*(1-pi)/(self.x[:,self.y == 0].shape[1]/self.x.shape[1])
            bounds = []
            for i in range(self.x.shape[1]):
                if self.y[i] == 1:
                    bounds.append ((0,ct))
                elif self.y[i] == 0:
                    bounds.append ((0,cf))
                    
        else:
            bounds = [(0, c)] * self.x.shape[1]
            
        if self.kernel=="rbf":
            dist = vcol((x_ext**2).sum(0)) + vrow((x_ext**2).sum(0))- 2*np.dot(x_ext.T, x_ext)
            self.h = np.exp(-gamma*dist) + k
            self.h = vcol(z) * vrow(z) * self.h
        
        elif self.kernel=="linear":
            self.h = np.dot(x_ext.T, x_ext)
            self.h = vcol(z) * vrow(z) * self.h
        
        elif self.kernel=="polynomial":
            self.h = (np.dot(x_ext.T, x_ext) + self.poly_bias)**degree + self.bias
            self.h = vcol(z) * vrow(z) * self.h
        else:
            raise TypeError("The model entered not supported.")
                
        self.alpha_star, _x, _y = scipy.optimize.fmin_l_bfgs_b(
        self._Ldual,
        np.zeros(self.x.shape[1]),
        bounds = bounds,
        factr=0.0,
        maxiter=iteration,
        maxfun=self.fun)


        if self.kernel=="linear":
            self.w_star=np.dot(x_ext, vcol(self.alpha_star) * vcol(z))
            return self.w_star, self._Jdual(self.alpha_star)[0]
        else:
            return vrow(self.alpha_star), self._Jdual(self.alpha_star)[0]
    
        
    def _Jdual(self, alpha):
        ha = np.dot(self.h, vcol(alpha))
        aha = np.dot(vrow(alpha), ha)
        a1 = alpha.sum()
        return -0.5*aha.ravel() + a1, -ha.ravel() + np.ones(alpha.size)
    
    def _Ldual(self, alpha):
        loss, grad = self._Jdual(alpha)
        return -loss, -grad
    
    def _Jprimal(self, x_ext, z,  w):
        s = np.dot(vrow(w), x_ext)
        loss = np.maximum(np.zeros(s.shape), 1-z*s).sum()
        return 0.5*np.linalg.norm(w)**2 + loss
    
    def predict(self, x_test):
        x_test_ext = np.vstack([x_test, np.ones((1, x_test.shape[1]))])
        
        if self.kernel=="linear":
            scores = np.dot(vrow(self.w_star), x_test_ext)
            
        elif self.kernel=="rbf":
            x_ext = np.vstack([self.x, np.ones((1, self.x.shape[1]))])
            z = np.zeros(self.y.shape)
            z[self.y == 1] = 1
            z[self.y == 0] = -1
            dis = vcol((x_ext ** 2).sum(0)) + vrow((x_test_ext ** 2).sum(0)) - 2 * np.dot(x_ext.T, x_test_ext)
            k = np.exp(-self.gamma*dis) + self.bias
            scores = np.dot(vrow(self.alpha_star)*z, k)
            
        elif self.kernel=="polynomial":
            x_ext = np.vstack([self.x, np.ones((1, self.x.shape[1]))])
            z = np.zeros(self.y.shape)
            z[self.y == 1] = 1
            z[self.y == 0] = -1
            k = (np.dot(x_ext.T, x_test_ext) + self.poly_bias)**self.degree + self.bias
            scores = np.dot(vrow(self.alpha_star)*z, k)
        return scores
