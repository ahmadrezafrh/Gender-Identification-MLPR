# -*- coding: utf-8 -*-

# =============================================================================
# Created By    : Ahmadreza Farmahini Farahani
# Created Date  : 2023/4
# Project       : This project is developed for "Machin Learnin and Pattern Recognition" course
# Supervisor    : Prof. Sandro Cumani
# Universsity   : Politecnico di Torino
# =============================================================================

import argparse
import pickle
import numpy as np
import sys
import os
sys.path.insert(1, './../')

from utils import read_data
from utils import k_folds
from utils import single_fold
from utils import vrow
from utils import create_params
from utils import load_configue

from classifiers import SVM
from preprocess import PCA
from preprocess import normalization
from preprocess import gaussianization
from postprocess import DCF

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("conf")
    args = parser.parse_args()
    svm_path = './../configues/test'
    conf_name = args.conf
    
    configue_path = os.path.join(svm_path, conf_name)
    cfg = load_configue(configue_path)
    configue = cfg.get_parameters()
    save_path = configue['save_path']
    
    train_x, train_y = read_data(file_name=configue['train_path'])
    test_x, test_y = read_data(file_name=configue['test_path'])
    single_split = single_fold(configue['split_ratio'], train_x, train_y)

    if len(configue['c_values']) != 1:
        cfg.add(key="c_values",value=list(10**np.linspace(configue['c_values'][0], configue['c_values'][1], num=configue['c_values'][2])))
    else:
    	cfg.add(key="c_values",value=[10**float(configue['c_values'][0])])
    
    params = list(create_params(**cfg.drop(keys=['test_path', 'split_ratio', 'train_path', 'save_path'])))
    cfn = 1
    cfp = 1


    
    list_models = []
    model = SVM(kernel="polynomial")
    print("Start running polynomial models.")
    for param in params:
        
        print('model type: polynomial SVM')
        print(f'c: {param["c_values"]}')
        print(f'degree: {param["degrees"]}')
        print(f'{"Normalization applied" if param["normalize"] else "Whithout normalization"}')
        print(f'PCA Dimension: {param["pca_dims"]}')
        print(f'Split: {param["splits"]}\n')
        
        if param["splits"]=="all-data":
            if param["normalize"]=='z-score':
                norm = normalization()
                x_train = norm.fit_transform(train_x)
                x_test = norm.transform(test_x)
            
            elif param["normalize"]=='gaussianization':
                norm = gaussianization()
                x_train = norm.fit_transform(train_x)
                x_test = norm.transform(test_x)
            
            else:
                x_train = train_x
                x_test = test_x
                
            if param["pca_dims"]:
                pca = PCA()
                x_train = pca.fit_transform(x_train, n_components=param["pca_dims"])
                x_test = pca.transform(x_test)
                
            y_train = train_y
            y_test = test_y

                                        
            if param["balance_priors"][0] == 'balanced':        
                model.fit(x_train, y_train, degree=param["degrees"], c=param["c_values"], balance=True, pi=param["balance_priors"][1])
            else:
                model.fit(x_train, y_train, degree=param["degrees"], c=param["c_values"])

            scores = model.predict(x_test)
            dcf = DCF(cfn,cfp, scores, y_test)
        
        else:
    
            
            if param["normalize"]=='z-score':
                norm = normalization()
                x_train = norm.fit_transform(single_split[0])
                x_test = norm.transform(test_x)
            
            elif param["normalize"]=='gaussianization':
                norm = gaussianization()
                x_train = norm.fit_transform(single_split[0])
                x_test = norm.transform(test_x)
            
            
            else:
                x_train = single_split[0]
                x_test = test_x
                
            if param["pca_dims"]:
                pca = PCA()
                x_train = pca.fit_transform(x_train, n_components=param["pca_dims"])
                x_test = pca.transform(x_test)
                
            y_train = single_split[1]
            y_test = test_y
    
            if param["balance_priors"][0] == 'balanced':        
                model.fit(x_train, y_train, degree=param["degrees"], c=param["c_values"], balance=True, pi=param["balance_priors"][1])
            else:
                model.fit(x_train, y_train, degree=param["degrees"], c=param["c_values"])  
                
            scores = model.predict(x_test)
            dcf = DCF(cfn,cfp, scores, y_test)
    
    
        models={}
        models["c"] = f'{param["c_values"]}'
        models["degree"] = f'{param["degrees"]}'
        models["balance"] = f'{param["balance_priors"]}'
        models["normalization"] = param["normalize"]
        models["pca"] = param["pca_dims"]
        models["folds"] = param["splits"]
        models["act_dcf"] = {}
        models["min_dcf"] = {}
        models["min_thresh"] = {}
        
    
    
        for prior in param["effective_priors"]:
            act_dcf = dcf.compute_act_dcf(prior)
            min_dcf = dcf.compute_min_dcf(prior)
            min_thresh = dcf.compute_min_dcf_threshold(prior)
            models["act_dcf"][prior] = act_dcf
            models["min_dcf"][prior] = min_dcf
            models["min_thresh"][prior] = min_thresh
            print(f"actDCF for prior {prior}: {act_dcf}")
            print(f"minDCF for prior {prior}: {min_dcf}")
            print(f"minThreshold for prior {prior}: {min_thresh}\n")
        
        list_models.append(models)
                
        print("\n=======================================================================================\n")
    
    
    with open(save_path, "wb") as fp:
        pickle.dump(list_models, fp) 
        
    print("End running polynomial SVM models.\n")


if __name__ == "__main__":
    main()
