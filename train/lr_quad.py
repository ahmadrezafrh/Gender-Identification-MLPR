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

from classifiers import LR
from preprocess import PCA
from preprocess import normalization
from preprocess import gaussianization
from postprocess import DCF




def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("conf")
    args = parser.parse_args()
    lr_path = './../configues/train'
    conf_name = args.conf
    
    configue_path = os.path.join(lr_path, conf_name)
    cfg = load_configue(configue_path)
    configue = cfg.get_parameters()
    save_path = configue['save_path']
    
    x, y = read_data(file_name=configue['train_path'])
    folds = k_folds(configue['n_folds'], x, y)
    single_split = single_fold(configue['split_ratio'], x, y)
    
    if len(configue['lambdas']) != 1:
    	cfg.add(key="lambdas",value=list(10**np.linspace(configue['lambdas'][0], configue['lambdas'][1], num=configue['lambdas'][2])))
    
    else:
    	cfg.add(key="lambdas",value=[10**int(configue['lambdas'][0])])
    
    params = list(create_params(**cfg.drop()))
    cfn = 1
    cfp = 1
    
    list_models = []
    model = LR(model_type='quadratic')
    
    print("Start running Logistic Regression models.")
    for param in params:
        
        print(f'model type: logistic regression {param["balance_priors"][0]}')
        print(f'lambda:  {param["lambdas"]}')
        print(f'{"Normalization applied" if param["normalize"] else "Whithout normalization"}')
        print(f'PCA Dimension: {param["pca_dims"]}')
        print(f'Split: {param["splits"]}\n')
        
        if param["splits"]=="k-folds":
            all_scores = []
            all_labels = []
            for fold in folds:
                
                if param["normalize"]=='z-score':
                    norm = normalization()
                    x_train = norm.fit_transform(fold[0])
                    x_test = norm.transform(fold[2])
                
                elif param["normalize"]=='gaussianization':
                    norm = gaussianization()
                    x_train = norm.fit_transform(fold[0])
                    x_test = norm.transform(fold[2])
                
                else:
                    x_train = fold[0]
                    x_test = fold[2]
                    
                if param["pca_dims"]:
                    pca = PCA()
                    x_train = pca.fit_transform(x_train, n_components=param["pca_dims"])
                    x_test = pca.transform(x_test)
                    
                y_train = fold[1]
                y_test = fold[3]
    
    
                if param["balance_priors"][0] == 'balanced':
                    model.fit(x_train, y_train, param["lambdas"], balance=True, pi=param["balance_priors"][1])
                else:
                    model.fit(x_train, y_train, param["lambdas"])
                all_scores.append(model.predict(x_test))
                all_labels.append(y_test)
            
            scores = vrow(np.hstack(all_scores))
            y_test = vrow(np.hstack(all_labels))
            dcf = DCF(cfn,cfp, scores, y_test)
        
        else:
    
            
            if param["normalize"]=='z-score':
                norm = normalization()
                x_train = norm.fit_transform(single_split[0])
                x_test = norm.transform(single_split[2])
            
            elif param["normalize"]=='gaussianization':
                norm = gaussianization()
                x_train = norm.fit_transform(single_split[0])
                x_test = norm.transform(single_split[2])
            
            else:
                x_train = single_split[0]
                x_test = single_split[2]
                
            if param["pca_dims"]:
                pca = PCA()
                x_train = pca.fit_transform(x_train, n_components=param["pca_dims"])
                x_test = pca.transform(x_test)
                
            y_train = single_split[1]
            y_test = single_split[3]
    
    
            if param["balance_priors"][0] == 'balanced':
                model.fit(x_train, y_train, param["lambdas"], balance=True, pi=param["balance_priors"][1])
            else:
                model.fit(x_train, y_train, param["lambdas"])
                
            scores = model.predict(x_test)
            dcf = DCF(cfn,cfp, scores, y_test)
    
    
        models={}
        models["balance"] = f'{param["balance_priors"]}'
        models["lambda"] = f'{param["lambdas"]}'
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
    print("End running Logistic regression models.\n")
    
if __name__ == "__main__":
    main()