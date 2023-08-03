# -*- coding: utf-8 -*-

# =============================================================================
# Created By    : Ahmadreza Farmahini Farahani
# Created Date  : 2023/4
# Project       : This project is developed for "Machin Learnin and Pattern Recognition" course
# Supervisor    : Prof. Sandro Cumani
# Universsity   : Politecnico di Torino
# =============================================================================
import ast
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.ticker
import pandas as pd
import warnings
sys.path.insert(1, './../../')
pd.set_option('max_colwidth', 800)

from utils import check_path
from utils import load_results
from utils import read_data
from utils import vrow
from utils import k_folds
from postprocess import DCF, thresh_calibration
from postprocess import calibration, fusion
from classifiers import SVM, GMM

split_ratio = 0.8
n_folds = 4
n_features = 12
cfn = 1
cfp = 1
priors = [0.1, 0.9, 0.5]

train_path = "./../../data/Train.txt"
test_path = "./../../data/Test.txt"
results_dir = './../../results'

tables_dir = os.path.join(results_dir, "tables/test")
figs_dir = os.path.join(results_dir, "figs/test")
test_results_dir = os.path.join(results_dir, "test")
train_results_dir = os.path.join(results_dir, "train")


mvg1_1_path = os.path.join(test_results_dir, "part1_1_mvg.results")
mvg1_2_path = os.path.join(test_results_dir, "part1_2_mvg.results")
lr2_1_path = os.path.join(test_results_dir, "part2_1_lr_linear.results")
lr2_2_path = os.path.join(test_results_dir, "part2_2_lr_linear.results")
lr2_3_path = os.path.join(test_results_dir, "part2_3_lr_quad.results")
lr2_4_path = os.path.join(test_results_dir, "part2_4_lr_quad.results")
svm3_1_path = os.path.join(test_results_dir, "part3_1_svm_linear.results")
svm3_2_path = os.path.join(test_results_dir, "part3_2_svm_linear.results")
svm4_1_path = os.path.join(test_results_dir, "part4_1_rbf.results")
svm4_2_path = os.path.join(test_results_dir, "part4_2_rbf.results")
svm5_1_path = os.path.join(test_results_dir, "part5_1_poly.results")
svm5_2_path = os.path.join(test_results_dir, "part5_2_poly.results")
gmm6_1_path = os.path.join(test_results_dir, "part6_1_gmm.results")


mvg1_1_path_val = os.path.join(train_results_dir, "part1_1_mvg.results")
mvg1_2_path_val = os.path.join(train_results_dir, "part1_2_mvg.results")
lr2_1_path_val = os.path.join(train_results_dir, "part2_1_lr_linear.results")
lr2_2_path_val = os.path.join(train_results_dir, "part2_2_lr_linear.results")
lr2_3_path_val = os.path.join(train_results_dir, "part2_3_lr_quad.results")
lr2_4_path_val = os.path.join(train_results_dir, "part2_4_lr_quad.results")
svm3_1_path_val = os.path.join(train_results_dir, "part3_1_svm_linear.results")
svm3_2_path_val = os.path.join(train_results_dir, "part3_2_svm_linear.results")
svm4_1_path_val = os.path.join(train_results_dir, "part4_1_rbf.results")
svm4_2_path_val = os.path.join(train_results_dir, "part4_2_rbf.results")
svm5_1_path_val = os.path.join(train_results_dir, "part5_1_poly.results")
svm5_2_path_val = os.path.join(train_results_dir, "part5_2_poly.results")
gmm6_1_path_val = os.path.join(train_results_dir, "part6_1_gmm.results")

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
# =============================================================================
# In the below codes we analyse the results of MVG models.
# =============================================================================

path = mvg1_1_path
print(f'analyzing {path.split("/")[-1]}...')
save_name = path.split('/')[-1].split('.')[0] + '.txt'
mvg = load_results(path)
models = pd.DataFrame(mvg)
models['min_dcf'] = models['min_dcf'].map(lambda x: {0.5: round(x[0.5], 3), 0.9: round(x[0.9], 3), 0.1: round(x[0.1], 3)})
models.style.hide_index()
check_path(os.path.join(tables_dir,save_name))
for pca in models.pca.unique():
    single_zscore = models[(models['pca']==pca) & (models['folds']=='single-fold') & (models['normalization']=='z-score')]
    kfolds_zscore = models[(models['pca']==pca) & (models['folds']=='all-data') & (models['normalization']=='z-score')]
    single_gauss = models[(models['pca']==pca) & (models['folds']=='single-fold') & (models['normalization']=='gaussianization')]
    kfolds_gauss = models[(models['pca']==pca) & (models['folds']=='all-data') & (models['normalization']=='gaussianization')]
    with open(os.path.join(tables_dir,save_name), 'a') as f:
        f.write('\t\t\t\t\tz-score\n')
        f.write(f'\t\t\t\t\tpca: {pca}\n')
        f.write('\t\t\t\t\tsingle fold\n')
        f.write(single_zscore[['model', 'min_dcf']].to_string(index=False))
        f.write('\n')
        f.write('\n\t\t\t\t\tk-folds\n')
        f.write(kfolds_zscore[['model', 'min_dcf']].to_string(index=False))
        f.write('\n\n')
        f.write('\n\n')
        f.write('\t\t\t\t\tgaussianization\n')
        f.write(f'\t\t\t\t\tpca: {pca}\n')
        f.write('\t\t\t\t\tsingle fold\n')
        f.write(single_gauss[['model', 'min_dcf']].to_string(index=False))
        f.write('\n')
        f.write('\n\t\t\t\t\tk-folds\n')
        f.write(kfolds_gauss[['model', 'min_dcf']].to_string(index=False))
        f.write('\n\n')
        f.write('\n\n')
        f.write('------------------------------------------------------------------------------------')
        f.write('\n\n') 
    

path = mvg1_2_path
print(f'analyzing {path.split("/")[-1]}...')
save_name = mvg1_1_path.split('/')[-1].split('.')[0] + '.txt'
mvg = load_results(path)
models = pd.DataFrame(mvg)
models['min_dcf'] = models['min_dcf'].map(lambda x: {0.5: round(x[0.5], 3), 0.9: round(x[0.9], 3), 0.1: round(x[0.1], 3)})
models.style.hide_index()
for pca in models.pca.unique():
    single = models[(models['pca']==pca) & (models['folds']=='single-fold')]
    kfolds = models[(models['pca']==pca) & (models['folds']=='all-data')]
    with open(os.path.join(tables_dir,save_name), 'a') as f:
        f.write('\t\t\t\t\tNOT_NORMALIZED\n')
        f.write(f'\t\t\t\t\tpca: {pca}\n')
        f.write('\t\t\t\t\tsingle fold\n')
        f.write(single[['model', 'min_dcf']].to_string(index=False))
        f.write('\n')
        f.write('\n\t\t\t\t\tall-data\n')
        f.write(kfolds[['model', 'min_dcf']].to_string(index=False))
        f.write('\n\n')
        f.write('------------------------------------------------------------------------------------')
        f.write('\n\n') 
    
  
# =============================================================================
# In the below codes we analyse the results of Linear Regression models.
# =============================================================================

path = lr2_1_path
print(f'analyzing {path.split("/")[-1]}...')
lr = load_results(path)
models = pd.DataFrame(lr)
path_val = lr2_1_path_val
lr_val = load_results(path_val)
models_val = pd.DataFrame(lr_val)
models = models[models['pca']==False]
models['lambda'] = models['lambda'].map(lambda x: float(x))
models_val = models_val[models_val['pca']==False]
models_val['lambda'] = models_val['lambda'].map(lambda x: float(x))
models = models[models['folds']=='all-data']
models_val = models_val[models_val['folds']=='k-folds']
save_name = 'part2_linear_upZ_midG_downR_leftKFOLDS'
plt.figure(figsize=(18, 6), dpi=100)
m=0  
for norm in models.normalization.unique():
    model = models[models['normalization']==norm]
    model_val = models_val[models_val['normalization']==norm]
    plt.subplot(1,3,m+1)
    plt.plot(model['lambda'], model['min_dcf'].map(lambda x: x[0.5]), color='darkred', label='minDcf ($\pi$ = 0.5)  [eval]')
    plt.plot(model['lambda'], model['min_dcf'].map(lambda x: x[0.1]), color='darkblue', label='minDcf ($\pi$ = 0.1)  [eval]')
    plt.plot(model['lambda'], model['min_dcf'].map(lambda x: x[0.9]), color='darkgreen', label='minDcf ($\pi$ = 0.9)  [eval]')
    plt.plot(model_val['lambda'], model_val['min_dcf'].map(lambda x: x[0.5]), color='darkred', linestyle='dashed', label='minDcf ($\pi$ = 0.5) [val]')
    plt.plot(model_val['lambda'], model_val['min_dcf'].map(lambda x: x[0.1]), color='darkblue', linestyle='dashed', label='minDcf ($\pi$ = 0.1) [val]')
    plt.plot(model_val['lambda'], model_val['min_dcf'].map(lambda x: x[0.9]), color='darkgreen',linestyle='dashed', label='minDcf ($\pi$ = 0.9) [val]')
    plt.tight_layout()
    plt.xscale('log' )
    plt.xlabel('λ')
    plt.ylabel("DCF")
    plt.legend(loc ="upper left")
    plt.grid(linestyle='--')
    m+=1
plt.savefig(os.path.join(figs_dir, save_name))




path = lr2_2_path
print(f'analyzing {path.split("/")[-1]}...')
save_name = path.split('/')[-1].split('.')[0] + '.txt'
lr = load_results(path)
models = pd.DataFrame(lr)
models['min_dcf'] = models['min_dcf'].map(lambda x: {0.5: round(x[0.5], 3), 0.9: round(x[0.9], 3), 0.1: round(x[0.1], 3)})
models = models[models['pca']==False]
models['balance'] = models['balance'].map(lambda x: ast.literal_eval(x))
models.style.hide_index()
check_path(os.path.join(tables_dir,save_name))
for norm in models.normalization.unique():
    with open(os.path.join(tables_dir, save_name), 'a') as f:
        f.write(f'\t\t\t___________________normalization: {norm}________________________\n')
    for fold in models.folds.unique():
        mods = models[(models['folds']==fold) & (models['normalization']==norm)]
        
        with open(os.path.join(tables_dir, save_name), 'a') as f:
            f.write('\t\t\t\t\tNp pca\n')
            f.write(f'\t\t\t\t\traw features - fold: {fold}\n')
            f.write(mods[['balance', 'min_dcf']].to_string(index=False))
            f.write('\n\n')
            f.write('------------------------------------------------------------------------------------')
            f.write('\n\n') 
    
  
# =============================================================================
# In the below codes we analyse the results of Quadratic Regression models.
# =============================================================================

path = lr2_3_path
print(f'analyzing {path.split("/")[-1]}...')
lr = load_results(path)
models = pd.DataFrame(lr)
path_val = lr2_3_path_val
lr_val = load_results(path_val)
models_val = pd.DataFrame(lr_val)
models = models[models['pca']==False]
models['lambda'] = models['lambda'].map(lambda x: float(x))
models_val = models_val[models_val['pca']==False]
models_val['lambda'] = models_val['lambda'].map(lambda x: float(x))
models = models[models['folds']=='all-data']
models_val = models_val[models_val['folds']=='k-folds']
save_name = 'part2_quad_leftZ_midG_rightR_all'
plt.figure(figsize=(18, 6), dpi=100)
m=0   
for norm in models.normalization.unique():
    model = models[models['normalization']==norm]
    model_val = models_val[models_val['normalization']==norm]
    plt.subplot(1,3,m+1)
    plt.plot(model['lambda'], model['min_dcf'].map(lambda x: x[0.5]), color='darkred', label='minDcf ($\pi$ = 0.5)  [eval]')
    plt.plot(model['lambda'], model['min_dcf'].map(lambda x: x[0.1]), color='darkblue', label='minDcf ($\pi$ = 0.1)  [eval]')
    plt.plot(model['lambda'], model['min_dcf'].map(lambda x: x[0.9]), color='darkgreen', label='minDcf ($\pi$ = 0.9)  [eval]')
    plt.plot(model_val['lambda'], model_val['min_dcf'].map(lambda x: x[0.5]), color='darkred', linestyle='dashed', label='minDcf ($\pi$ = 0.5) [val]')
    plt.plot(model_val['lambda'], model_val['min_dcf'].map(lambda x: x[0.1]), color='darkblue', linestyle='dashed', label='minDcf ($\pi$ = 0.1) [val]')
    plt.plot(model_val['lambda'], model_val['min_dcf'].map(lambda x: x[0.9]), color='darkgreen',linestyle='dashed', label='minDcf ($\pi$ = 0.9) [val]')
    plt.xscale('log' )
    plt.xlabel('λ')
    plt.ylabel("DCF")
    plt.legend(loc ="upper left")
    plt.grid(linestyle='--')
    plt.tight_layout()
    m+=1
plt.savefig(os.path.join(figs_dir, save_name))



path = lr2_4_path
print(f'analyzing {path.split("/")[-1]}...')
save_name = path.split('/')[-1].split('.')[0] + '.txt'
lr = load_results(path)
models = pd.DataFrame(lr)
models['min_dcf'] = models['min_dcf'].map(lambda x: {0.5: round(x[0.5], 3), 0.9: round(x[0.9], 3), 0.1: round(x[0.1], 3)})
models = models[models['pca']==False]
models['balance'] = models['balance'].map(lambda x: ast.literal_eval(x))
models.style.hide_index()
check_path(os.path.join(tables_dir,save_name))
for norm in models.normalization.unique():
    with open(os.path.join(tables_dir, save_name), 'a') as f:
        f.write(f'\t\t\t___________________normalization: {norm}________________________\n')
    for fold in models.folds.unique():
        mods = models[(models['folds']==fold) & (models['normalization']==norm)]
        
        with open(os.path.join(tables_dir, save_name), 'a') as f:
            f.write('\t\t\t\t\tNo pca\n')
            f.write(f'\t\t\t\t\traw features - fold: {fold}\n')
            f.write(mods[['balance', 'min_dcf']].to_string(index=False))
            f.write('\n\n')
            f.write('------------------------------------------------------------------------------------')
            f.write('\n\n') 
            
            
# =============================================================================
# In the below codes we analyse the results of Linear SVM.
# =============================================================================

path = svm3_1_path
print(f'analyzing {path.split("/")[-1]}...')
svm = load_results(path)
models = pd.DataFrame(svm)
models['c'] = models['c'].map(lambda x: float(x))
path_val = svm3_1_path_val
svm_val = load_results(path_val)
models_val = pd.DataFrame(svm_val)
models_val['c'] = models_val['c'].map(lambda x: float(x))
mods_val = models_val[(models_val['normalization']==False) & (models_val['folds']=='k-folds')]
mods = models[(models['normalization']==False) & (models['folds']=='all-data')]
plt.figure(figsize=(12, 6), dpi=100)
save_name = 'part3_svm_linear_quad'
model = mods[(mods['pca']==False) & (mods['balance']=="['imbalanced', 0.5]")]
model_val = mods_val[(mods_val['pca']==False) & (mods_val['balance']=="['imbalanced', 0.5]")]
plt.subplot(1,2,1)
plt.plot(model['c'], model['min_dcf'].map(lambda x: x[0.5]), color='darkred', label='minDcf ($\pi$ = 0.5) [eval]')
plt.plot(model['c'], model['min_dcf'].map(lambda x: x[0.1]), color='darkblue', label='minDcf ($\pi$ = 0.1) [eval]')
plt.plot(model['c'], model['min_dcf'].map(lambda x: x[0.9]), color='darkgreen', label='minDcf ($\pi$ = 0.9) [eval]')
plt.plot(model_val['c'], model_val['min_dcf'].map(lambda x: x[0.5]), linestyle='dashed', color='darkred', label='minDcf ($\pi$ = 0.5) [val]')
plt.plot(model_val['c'], model_val['min_dcf'].map(lambda x: x[0.1]), linestyle='dashed', color='darkblue', label='minDcf ($\pi$ = 0.1) [val]')
plt.plot(model_val['c'], model_val['min_dcf'].map(lambda x: x[0.9]), linestyle='dashed', color='darkgreen', label='minDcf ($\pi$ = 0.9) [val]')
plt.xscale('log' )
plt.xlabel('C')
plt.ylabel("DCF")
plt.tight_layout()
plt.grid(linestyle='--')
plt.legend()

path = svm5_1_path
svm = load_results(path)
models = pd.DataFrame(svm)
models['c'] = models['c'].map(lambda x: float(x))
path_val = svm5_1_path_val
svm_val = load_results(path_val)
models_val = pd.DataFrame(svm_val)
models_val['c'] = models_val['c'].map(lambda x: float(x))
mods_val = models_val[models_val['folds']=='k-folds']
mods = models[models['folds']=='all-data']
mods_val = mods_val[mods_val['degree']=='2']
mods = mods[mods['degree']=='2']
mods_val = mods_val[mods_val['normalization']==False]
mods = mods[mods['normalization']==False]
model = mods[mods['pca']==False]
model_val = mods_val[mods_val['pca']==False]
plt.subplot(1,2,2)
plt.plot(model['c'], model['min_dcf'].map(lambda x: x[0.5]), color='darkred', label='minDcf ($\pi$ = 0.5) (degree = 2) [eval]')
plt.plot(model['c'], model['min_dcf'].map(lambda x: x[0.1]), color='darkblue', label='minDcf ($\pi$ = 0.1) (degree = 2) [eval]')
plt.plot(model['c'], model['min_dcf'].map(lambda x: x[0.9]), color='darkgreen', label='minDcf ($\pi$ = 0.9) (degree = 2) [eval]')
plt.plot(model_val['c'], model_val['min_dcf'].map(lambda x: x[0.5]), linestyle='dashed', color='darkred', label='minDcf ($\pi$ = 0.5) (degree = 2) [val]')
plt.plot(model_val['c'], model_val['min_dcf'].map(lambda x: x[0.1]), linestyle='dashed', color='darkblue', label='minDcf ($\pi$ = 0.1) (degree= 2) [val]')
plt.plot(model_val['c'], model_val['min_dcf'].map(lambda x: x[0.9]), linestyle='dashed', color='darkgreen', label='minDcf ($\pi$ = 0.9) (degree = 2) [val]')
plt.xscale('log' )
plt.xlabel('C')
plt.ylabel("DCF")
plt.grid(linestyle='--')
plt.legend()
plt.savefig(os.path.join(figs_dir, save_name))



path = svm3_2_path
print(f'analyzing {path.split("/")[-1]}...')
save_name = path.split('/')[-1].split('.')[0] + '.txt'
svm = load_results(path)
models = pd.DataFrame(svm)
models['min_dcf'] = models['min_dcf'].map(lambda x: {0.5: round(x[0.5], 3), 0.9: round(x[0.9], 3), 0.1: round(x[0.1], 3)})
models = models[models['folds']=='all-data']
models['balance'] = models['balance'].map(lambda x: ast.literal_eval(x))
models.style.hide_index()
check_path(os.path.join(tables_dir,save_name))
with open(os.path.join(tables_dir, save_name), 'a') as f:
    f.write('\t\t\t\t\t4-folds\n')
    f.write('\t\t\t\t\traw features - pca: No PCA\n')
    f.write(models[['c','balance', 'min_dcf']].to_string(index=False))
    f.write('\n\n')
    f.write('------------------------------------------------------------------------------------')
    f.write('\n\n') 



# =============================================================================
# In the below codes we analyse the results of RBF.
# =============================================================================



path = svm4_1_path
print(f'analyzing {path.split("/")[-1]}...')
svm = load_results(path)
models = pd.DataFrame(svm)
models = models[models['pca']==False]
models['c'] = models['c'].map(lambda x: float(x))
path_val = svm4_1_path_val
svm_val = load_results(path_val)
models_val = pd.DataFrame(svm_val)
models_val['c'] = models_val['c'].map(lambda x: float(x))
models_val = models_val[models_val['pca']==False]
models['min_dcf'] = models['min_dcf'].map(lambda x: x[0.5])
models_val['min_dcf'] = models_val['min_dcf'].map(lambda x: x[0.5])
mods_val = models_val
mods = models
mods = mods.drop(['balance'], axis=1)
save_name = 'part4_rbf_gamma_upALL_leftZ_midG_rightR_all'
plt.figure(figsize=(18, 12), dpi=100)
m=0
for fold in models.folds.unique():
    for norm in models.normalization.unique():
        fold_val = 'k-folds' if fold=='all-data' else fold
        model = mods[(mods['normalization']==norm) & (mods['folds']=='all-data')]
        model_val = mods_val[(mods_val['normalization']==norm) & (mods_val['folds']==fold_val)]
        plt.subplot(2,3,m+1)
        plt.plot(model[model['gamma']=='0.0001']['c'], model[model['gamma']=='0.0001']['min_dcf'], color='sienna', label='$\log\gamma$ = -4 [val]')
        plt.plot(model[model['gamma']=='0.001']['c'], model[model['gamma']=='0.001']['min_dcf'], color='tan', label='$\log\gamma$ = -3 [val]')
        plt.plot(model[model['gamma']=='0.01']['c'], model[model['gamma']=='0.01']['min_dcf'], color='teal', label='$\log\gamma$ = -2 [val]')
        plt.plot(model[model['gamma']=='0.1']['c'], model[model['gamma']=='0.1']['min_dcf'], color='gray', label='$\log\gamma$ = -1 [val]')
        plt.plot(model_val[model_val['gamma']=='0.0001']['c'], model_val[model_val['gamma']=='0.0001']['min_dcf'], linestyle='dashed', color='sienna', label='$\log\gamma$ = -4 [eval]')
        plt.plot(model_val[model_val['gamma']=='0.001']['c'], model_val[model_val['gamma']=='0.001']['min_dcf'], linestyle='dashed', color='tan', label='$\log\gamma$ = -3 [eval]')
        plt.plot(model_val[model_val['gamma']=='0.01']['c'], model_val[model_val['gamma']=='0.01']['min_dcf'], linestyle='dashed', color='teal', label='$\log\gamma$ = -2 [eval]' )
        plt.plot(model_val[model_val['gamma']=='0.1']['c'], model_val[model_val['gamma']=='0.1']['min_dcf'], linestyle='dashed', color='gray', label='$\log\gamma$ = -1 [eval]')
        plt.tight_layout()
        plt.xscale('log' )
        plt.xlabel('C')
        plt.ylabel("DCF")
        plt.legend()
        plt.grid(linestyle='--')
        m+=1
plt.savefig(os.path.join(figs_dir, save_name))




path = svm4_1_path
print(f'analyzing {path.split("/")[-1]}...')
svm = load_results(path)
models = pd.DataFrame(svm)
models['c'] = models['c'].map(lambda x: float(x))
path_val = svm4_1_path_val
svm_val = load_results(path_val)
models_val = pd.DataFrame(svm_val)
models_val['c'] = models_val['c'].map(lambda x: float(x))
mods_val = models_val[models_val['folds']=='k-folds']
mods = models[models['folds']=='all-data']
mods_val = mods_val[mods_val['gamma']=='0.001']
mods = mods[mods['gamma']=='0.001']
mods_val = mods_val[mods_val['pca']==False]
mods = mods[mods['pca']==False]
m=0
plt.figure(figsize=(18, 6), dpi=100)
save_name = 'part4_rbf_leftZ_midG_rightR_all-data_noPCA'
for norm in models.normalization.unique():
    model = mods[mods['normalization']==norm]
    model_val = mods_val[mods_val['normalization']==norm]
    plt.subplot(1,3,m+1)
    plt.plot(model['c'], model['min_dcf'].map(lambda x: x[0.5]), color='darkred', label='minDcf ($\pi$ = 0.5) ($\log\gamma$ = -2) [eval]')
    plt.plot(model['c'], model['min_dcf'].map(lambda x: x[0.1]), color='darkblue', label='minDcf ($\pi$ = 0.1) ($\log\gamma$ = -2) [eval]')
    plt.plot(model['c'], model['min_dcf'].map(lambda x: x[0.9]), color='darkgreen', label='minDcf ($\pi$ = 0.9) ($\log\gamma$ = -2) [eval]')
    plt.plot(model_val['c'], model_val['min_dcf'].map(lambda x: x[0.5]), linestyle='dashed', color='darkred', label='minDcf ($\pi$ = 0.5) ($\log\gamma$ = -2) [val]')
    plt.plot(model_val['c'], model_val['min_dcf'].map(lambda x: x[0.1]), linestyle='dashed', color='darkblue', label='minDcf ($\pi$ = 0.1) ($\log\gamma$ = -2) [val]')
    plt.plot(model_val['c'], model_val['min_dcf'].map(lambda x: x[0.9]), linestyle='dashed', color='darkgreen', label='minDcf ($\pi$ = 0.9) ($\log\gamma$ = -2) [val]')
    plt.xscale('log' )
    plt.xlabel('C')
    plt.ylabel("DCF")
    plt.grid(linestyle='--')
    plt.legend()
    plt.tight_layout()
    m+=1
plt.savefig(os.path.join(figs_dir, save_name))




path = svm4_2_path
print(f'analyzing {path.split("/")[-1]}...')
save_name = path.split('/')[-1].split('.')[0] + '.txt'
svm = load_results(path)
models = pd.DataFrame(svm)
models['min_dcf'] = models['min_dcf'].map(lambda x: {0.5: round(x[0.5], 3), 0.9: round(x[0.9], 3), 0.1: round(x[0.1], 3)})
models.style.hide_index()
models['gamma'] = models['gamma'].map('gamma: {}'.format)
models['c'] = models['c'].map('c: {}'.format)
models['balanced'] = models['balance'].map(lambda x: x[0])
models['prior'] = models['balance'].map(lambda x: x[1])
models['prior'] = models['prior'].map('prior: {}'.format)
check_path(os.path.join(tables_dir,save_name))
with open(os.path.join(tables_dir, save_name), 'a') as f:
    for norm in models.normalization.unique():
        f.write(f'\t\t\t___________________normalization: {norm}________________________\n')
        for fold in models.folds.unique():
            f.write(f'\t\t\t\t\t{fold}\n')
            for pca in models.pca.unique():
                model = models[(models['pca']==pca) & (models['folds']==fold) & (models['normalization']==norm)]
                f.write(f'\t\t\t\t\tpca: {pca}\n')
                f.write(model[['balance', 'gamma', 'c', 'prior', 'min_dcf']].to_string(index=False))
                f.write('\n')
                f.write('\n\n')
                
        f.write('------------------------------------------------------------------------------------')
        f.write('\n\n') 



# =============================================================================
# In the below codes we analyse the results of POLY.
# =============================================================================

path = svm5_1_path
print(f'analyzing {path.split("/")[-1]}...')
svm = load_results(path)
models = pd.DataFrame(svm)
models['c'] = models['c'].map(lambda x: float(x))
models['min_dcf'] = models['min_dcf'].map(lambda x: x[0.5])
path_val = svm5_1_path_val
svm_val = load_results(path_val)
models_val = pd.DataFrame(svm_val)
models_val['c'] = models_val['c'].map(lambda x: float(x))
models_val['min_dcf'] = models_val['min_dcf'].map(lambda x: x[0.5])
for norm in models.normalization.unique():
    m=0
    plt.figure(figsize=(12, 6), dpi=100)
    save_name = f'part5_leftKFOLDS_{norm}'
    for fold, fold_val in zip(models.folds.unique(), models_val.folds.unique()):
        model = models[(models['normalization']==norm) & (models['folds']==fold)]
        model_val = models_val[(models_val['normalization']==norm) & (models_val['folds']==fold_val)]
        plt.subplot(1,2,m+1)
        plt.plot(model[model['degree']=='2']['c'], model[model['degree']=='2']['min_dcf'], color='tan', label='$degree$ = 2 [eval]')
        plt.plot(model[model['degree']=='3']['c'], model[model['degree']=='3']['min_dcf'], color='teal', label='$degree$ = 3 [eval]')
        plt.plot(model[model['degree']=='4']['c'], model[model['degree']=='4']['min_dcf'], color='gray', label='$degree$ = 4 [eval]')
        plt.plot(model_val[model_val['degree']=='2']['c'], model_val[model_val['degree']=='2']['min_dcf'], linestyle='dashed', color='tan', label='$degree$ = 2 [val]')
        plt.plot(model_val[model_val['degree']=='3']['c'], model_val[model_val['degree']=='3']['min_dcf'], linestyle='dashed', color='teal', label='$degree$ = 3 [val]')
        plt.plot(model_val[model_val['degree']=='4']['c'], model_val[model_val['degree']=='4']['min_dcf'], linestyle='dashed', color='gray', label='$degree$ = 4 [val]')
        plt.tight_layout()
        plt.grid(linestyle='--')
        plt.xscale('log' )
        plt.xlabel('C')
        plt.ylabel("DCF")
        plt.legend()
        m+=1
    plt.savefig(os.path.join(figs_dir, save_name))



path = svm5_2_path
print(f'analyzing {path.split("/")[-1]}...')
save_name = path.split('/')[-1].split('.')[0] + '.txt'
svm = load_results(path)
models = pd.DataFrame(svm)
models['min_dcf'] = models['min_dcf'].map(lambda x: {0.5: round(x[0.5], 3), 0.9: round(x[0.9], 3), 0.1: round(x[0.1], 3)})
models.style.hide_index()
models['degree'] = models['degree'].map('degree: {}'.format)
models['c'] = models['c'].map('c: {}'.format)
check_path(os.path.join(tables_dir,save_name))
for pca in models.pca.unique():
    with open(os.path.join(tables_dir, save_name), 'a') as f:
        f.write('\t\t\t\t\t4-folds\n')
        f.write(f'\t\t\t\t\traw features - pca: {pca}\n')
        f.write(models[['normalization','balance','degree', 'c', 'min_dcf']].to_string(index=False))
        f.write('\n\n')
        f.write('------------------------------------------------------------------------------------')
        f.write('\n\n') 




# =============================================================================
# In the below codes we analyse the results of GMM.
# =============================================================================


path = gmm6_1_path
print(f'analyzing {path.split("/")[-1]}...')
gmm = load_results(path)
models = pd.DataFrame(gmm)
save_name = path.split('/')[-1].split('.')[0]
path_val = gmm6_1_path_val
gmm_val = load_results(path_val)
models_val = pd.DataFrame(gmm_val)
models_val['min_dcf'] = models_val['min_dcf'].map(lambda x: x[0.5])
models['min_dcf'] = models['min_dcf'].map(lambda x: x[0.5])
for norm in models.normalization.unique():
    m=0
    plt.figure(figsize=(13, 11), dpi=100)
    save_name = f'part6_upFULL_rightTIED__{norm}'
    for model in models.model.unique():
        mod = models[(models['model']==model) & (models['normalization']==norm)]
        mod_val = models_val[(models_val['model']==model) & (models_val['normalization']==norm)]
        width = 0.15
        x = np.arange(len(mod[mod['pca']==8]['n_component']))
        plt.subplot(2,2,m+1)
        plt.bar(x - width/2, mod[mod['pca']==8]['min_dcf'], width, label='minDcf ($\pi$ = 0.5) - PCA (m=8) [eval]',facecolor = 'darkcyan', edgecolor='#169acf')
        plt.bar(x - 3 * width/2, mod_val[mod_val['pca']==8]['min_dcf'], width, label='minDcf ($\pi$ = 0.5) - PCA (m=8) [val]', hatch='/', facecolor = 'cyan', edgecolor='#169acf')
        plt.bar(x + 3 * width/2, mod[mod['pca']==False]['min_dcf'], width, label='minDcf ($\pi$ = 0.5) - without PCA [eval]',facecolor = 'darkkhaki', edgecolor='#169acf')
        plt.bar(x + width/2, mod_val[mod_val['pca']==False]['min_dcf'], width, label='minDcf ($\pi$ = 0.5) - without PCA [val]', hatch='/', facecolor = 'khaki', edgecolor='#169acf')
        plt.xticks(x, mod[mod['pca']==8]['n_component'])
        plt.tight_layout()
        plt.xlabel('GMM Components')
        plt.ylabel("DCF")
        plt.legend()
        plt.grid(linestyle='--')
        m+=1
    plt.savefig(os.path.join(figs_dir, save_name))







path = gmm6_1_path
save_name = path.split('/')[-1].split('.')[0] + '.txt'
svm = load_results(path)
models = pd.DataFrame(svm)
models['min_dcf'] = models['min_dcf'].map(lambda x: {0.5: round(x[0.5], 3), 0.9: round(x[0.9], 3), 0.1: round(x[0.1], 3)})
models = models[models['normalization']==False]
models.style.hide_index()
models['n_component'] = models['n_component'].map('n_component: {}'.format)
check_path(os.path.join(tables_dir,save_name))
for pca in models.pca.unique():
    with open(os.path.join(tables_dir, save_name), 'a') as f:
        model = models[models['pca']==pca]
        f.write('\t\t\t\t\t4-folds\n')
        f.write(f'\t\t\t\t\traw features - pca: {pca}\n')
        f.write(model[['model', 'n_component', 'min_dcf']].to_string(index=False))
        f.write('\n\n')
        f.write('------------------------------------------------------------------------------------')
        f.write('\n\n') 



# =============================================================================
# Calibration
# =============================================================================


'''
For this part we extract the best models with gmm and svm.
For gmm we have:
    full_tied   n_component: 8  {0.1: 0.2220238095238095, 0.5: 0.06825396825396826, 0.9: 0.19742063492063497}
    
For SVM we have(RBF):
    kfolds: gamma: 0.01 c: 10 {0.1: 0.33690476190476193, 0.5: 0.10892857142857143, 0.9: 0.3244047619047619}
    single-fold: gamma: 0.01 c: 10 {0.1: 0.3545454545454545, 0.5: 0.0890909090909091, 0.9: 0.38484848484848494}
'''
print('analyzing calibration ...')
path = svm4_2_path
svm = load_results(path)
models_svm = pd.DataFrame(svm)
models_svm['min_dcf'] = models_svm['min_dcf'].map(lambda x: {0.5: round(x[0.5], 3), 0.9: round(x[0.9], 3), 0.1: round(x[0.1], 3)})
models_svm = models_svm[(models_svm['pca']==False) & (models_svm['normalization']==False)]
models_svm.style.hide_index()
models_svm['gamma'] = models_svm['gamma'].map('gamma: {}'.format)
models_svm['c'] = models_svm['c'].map('c: {}'.format)

path = gmm6_1_path
save_name = 'part7_calib_best.txt'
gmm = load_results(path)
models_gmm = pd.DataFrame(gmm)
models_gmm['min_dcf'] = models_gmm['min_dcf'].map(lambda x: {0.5: round(x[0.5], 3), 0.9: round(x[0.9], 3), 0.1: round(x[0.1], 3)})
models_gmm = models_gmm[(models_gmm['pca']==False) & (models_gmm['model']=='full_tied') & (models_gmm['n_component']==8) & (models_gmm['normalization']==False)]
models_gmm.style.hide_index()
check_path(os.path.join(tables_dir,save_name))
for fold in models_svm.folds.unique():
    with open(os.path.join(tables_dir, save_name), 'a') as f:
        model_svm = models_svm[models_svm['folds']==fold]
        model_gmm = models_gmm[models_gmm['folds']==fold]
        f.write(f'\t\t\t\t\t{fold}\n')
        f.write('\t\t\t\t\traw features - pca: No pca\n')
        f.write('\n_GMM_')
        f.write(model_gmm[['model', 'n_component', 'min_dcf' , 'act_dcf']].to_string(index=False))
        f.write('\n_SVM_')
        f.write(model_svm[['balance', 'gamma', 'c', 'min_dcf', 'act_dcf']].to_string(index=False))
        f.write('\n\n')
        f.write('------------------------------------------------------------------------------------')
        f.write('\n\n') 


'''
Now we have the results, we go for calibration.
in the below codes we collect scores of best gmm and svm models.
'''

x_train, y_train = read_data(file_name=train_path)
x_test, y_test = read_data(file_name=test_path)
train_folds = k_folds(n_folds, x_train, y_train)



scores_train_svm = []
labels_train_svm = []
svm_valid = SVM(kernel='rbf')  
for fold in train_folds:
    x_train = fold[0]
    x_test = fold[2]
    y_train = fold[1]
    y_test = fold[3]           
    svm_valid.fit(x_train, y_train, gamma=0.001, c=10) 
    scores_train_svm.append(svm_valid.predict(x_test))
    labels_train_svm.append(y_test)
scores_train_svm = vrow(np.hstack(scores_train_svm))
labels_train_svm = vrow(np.hstack(labels_train_svm))

scores_train_gmm_full = []
labels_train_gmm_full = []
gmm_valid_full = GMM(model='full', tied=False)
for fold in train_folds:
    x_train = fold[0]
    x_test = fold[2]
    y_train = fold[1]
    y_test = fold[3]           
    gmm_valid_full.fit(x_train, y_train, n_components=4)
    scores_train_gmm_full.append(gmm_valid_full.predict(x_test))
    labels_train_gmm_full.append(y_test)
scores_train_gmm_full = vrow(np.hstack(scores_train_gmm_full))
labels_train_gmm_full = vrow(np.hstack(labels_train_gmm_full))

scores_train_gmm = []
labels_train_gmm = []
gmm_valid = GMM(model='full', tied=True)
for fold in train_folds:
    x_train = fold[0]
    x_test = fold[2]
    y_train = fold[1]
    y_test = fold[3]           
    gmm_valid.fit(x_train, y_train, n_components=4)
    scores_train_gmm.append(gmm_valid.predict(x_test))
    labels_train_gmm.append(y_test)
scores_train_gmm = vrow(np.hstack(scores_train_gmm))
labels_train_gmm = vrow(np.hstack(labels_train_gmm))



x_train, y_train = read_data(file_name=train_path)
x_test, y_test = read_data(file_name=test_path)



gmm_full = GMM(model='full', tied=False)
gmm_full.fit(x_train, y_train, n_components=4)
scores_gmm_full = gmm_full.predict(x_test).reshape(1,-1)
labels_gmm_full = y_test
dcf_gmm_full = DCF(cfn,cfp, scores_gmm_full, labels_gmm_full)

svm = SVM(kernel='rbf')  
svm.fit(x_train, y_train, gamma=0.001, c=10) 
scores_svm = svm.predict(x_test)
labels_svm = y_test
dcf_svm = DCF(cfn,cfp, scores_svm, labels_svm)

gmm = GMM(model='full', tied=True)
gmm.fit(x_train, y_train, n_components=4)
scores_gmm = gmm.predict(x_test).reshape(1,-1)
labels_gmm = y_test
dcf_gmm = DCF(cfn,cfp, scores_gmm, labels_gmm)


'''
Now we plot DET figure (with fusion). The first fig is not cut. The
second one is cut for better visualization.
'''


labels = y_test
scores = np.vstack(( scores_train_gmm, scores_train_gmm_full))
scores_test = np.vstack(( scores_gmm, scores_gmm_full))
model = fusion()
model.fit(scores,labels_train_gmm)
fusion_scores = model.predict(scores_test)
fusion_labels = labels
dcf_fusion = DCF(cfn,cfp, fusion_scores, fusion_labels)
print(f'fusion min dcf: {dcf_fusion.compute_min_dcf(0.5)}')

fnrs_svm, fprs_svm = dcf_svm.det_plot()
fnrs_gmm, fprs_gmm = dcf_gmm.det_plot()
fnrs_gmm_full, fprs_gmm_full = dcf_gmm_full.det_plot()
fnrs_fusion, fprs_fusion = dcf_fusion.det_plot()

save_name = 'part7_calibrated_fusion_det'
plt.figure(figsize=(7, 7), dpi=100)
plt.plot(fnrs_svm,fprs_svm, color='darkblue', label='SVM')
plt.plot(fnrs_gmm,fprs_gmm, color='darkred', label='GMM tied')
plt.plot(fnrs_gmm_full,fprs_gmm_full, color='darkorange', label='GMM full')
plt.plot(fnrs_fusion,fprs_fusion, color='darkgreen', label='Fusion (GMM full, GMM tied)')
plt.title('DET plot')
ax = plt.gca()
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xticks([1,2,5,10,20,40])
ax.set_yticks([1,2,5,10,20,40])
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.get_xaxis().set_tick_params(which='minor', size=0)
ax.get_xaxis().set_tick_params(which='minor', width=0) 
plt.xlim([1, 40])
plt.ylim([1, 40])
plt.xlabel('False Positive Rate')
plt.ylabel("False Negative Rate")
plt.grid(linestyle='--')
plt.tight_layout()
plt.legend()
plt.savefig(os.path.join(figs_dir, save_name))



'''
Now we calibrate both GMM and SVM.
'''


save_name = 'part7_calibrated_scores.txt'
priors = [0.1, 0.9, 0.5]
check_path(os.path.join(tables_dir,save_name))
with open(os.path.join(tables_dir, save_name), 'a') as f:
    f.write('\t\t\t\t\tcalibration - SVM - RBF - kfolds\n')
    f.write('\t\t\t\t\traw features - pca: no pca\n\n\n')
    f.write(f'min-dcfs \t\t min-dcf(0.5)={dcf_svm.compute_min_dcf(0.5)}\t\t min-dcf(0.1)={dcf_svm.compute_min_dcf(0.1)}\t\t min-dcf(0.9)={dcf_svm.compute_min_dcf(0.9)}\n')
    f.write(f'Not-calibrated \t\t act-dcf(0.5)={dcf_svm.compute_act_dcf(0.5)}\t\t act-dcf(0.1)={dcf_svm.compute_act_dcf(0.1)}\t\t act-dcf(0.9)={dcf_svm.compute_act_dcf(0.9)}\n')

for prior in priors:
    model = calibration(pi=prior)
    model.fit(scores_train_svm, labels_train_svm)
    svm_cal_scores = model.predict(scores_svm)
    svm_cal_labels = y_test
    dcf_cal_svm = DCF(cfn,cfp, svm_cal_scores, svm_cal_labels)
    with open(os.path.join(tables_dir, save_name), 'a') as f:
        f.write(f'log-reg-({prior}) \t\t act-dcf(0.5)={dcf_cal_svm.compute_act_dcf(0.5)}\t\t act-dcf(0.1)={dcf_cal_svm.compute_act_dcf(0.1)}\t\t act-dcf(0.9)={dcf_cal_svm.compute_act_dcf(0.9)}\n')
        
        
with open(os.path.join(tables_dir, save_name), 'a') as f:
    f.write('\n\n')
    f.write('------------------------------------------------------------------------------------')
    f.write('\n\n')
    f.write('\t\t\t\t\tcalibration - GMM tied\n')
    f.write('\t\t\t\t\traw features - pca: no pca\n\n\n')
    f.write(f'min-dcfs \t\t min-dcf(0.5)={dcf_gmm.compute_min_dcf(0.5)}\t\t min-dcf(0.1)={dcf_gmm.compute_min_dcf(0.1)}\t\t min-dcf(0.9)={dcf_gmm.compute_min_dcf(0.9)}\n')
    f.write(f'Not-calibrated \t\t act-dcf(0.5)={dcf_gmm.compute_act_dcf(0.5)}\t\t act-dcf(0.1)={dcf_gmm.compute_act_dcf(0.1)}\t\t act-dcf(0.9)={dcf_gmm.compute_act_dcf(0.9)}\n')

for prior in priors:
    model = calibration(pi=prior)
    model.fit(scores_train_gmm, labels_train_gmm)
    gmm_cal_scores = model.predict(scores_gmm)
    gmm_cal_labels = y_test
    dcf_cal_gmm = DCF(cfn,cfp, gmm_cal_scores, gmm_cal_labels)
    with open(os.path.join(tables_dir, save_name), 'a') as f:
        f.write(f'log-reg-({prior}) \t\t act-dcf(0.5)={dcf_cal_gmm.compute_act_dcf(0.5)}\t\t act-dcf(0.1)={dcf_cal_gmm.compute_act_dcf(0.1)}\t\t act-dcf(0.9)={dcf_cal_gmm.compute_act_dcf(0.9)}\n')

with open(os.path.join(tables_dir, save_name), 'a') as f:
    f.write('\n\n')
    f.write('------------------------------------------------------------------------------------')
    f.write('\n\n')
    f.write('\t\t\t\t\tcalibration - GMM full\n')
    f.write('\t\t\t\t\traw features - pca: no pca\n\n\n')
    f.write(f'min-dcfs \t\t min-dcf(0.5)={dcf_gmm_full.compute_min_dcf(0.5)}\t\t min-dcf(0.1)={dcf_gmm_full.compute_min_dcf(0.1)}\t\t min-dcf(0.9)={dcf_gmm_full.compute_min_dcf(0.9)}\n')
    f.write(f'Not-calibrated \t\t act-dcf(0.5)={dcf_gmm_full.compute_act_dcf(0.5)}\t\t act-dcf(0.1)={dcf_gmm_full.compute_act_dcf(0.1)}\t\t act-dcf(0.9)={dcf_gmm_full.compute_act_dcf(0.9)}\n')

for prior in priors:
    model = calibration(pi=prior)
    model.fit(scores_train_gmm_full, labels_train_gmm_full)
    gmm_cal_scores_full = model.predict(scores_gmm_full)
    gmm_cal_labels_full = y_test
    dcf_cal_gmm_full = DCF(cfn,cfp, gmm_cal_scores_full, gmm_cal_labels_full)
    with open(os.path.join(tables_dir, save_name), 'a') as f:
        f.write(f'log-reg-({prior}) \t\t act-dcf(0.5)={dcf_cal_gmm_full.compute_act_dcf(0.5)}\t\t act-dcf(0.1)={dcf_cal_gmm_full.compute_act_dcf(0.1)}\t\t act-dcf(0.9)={dcf_cal_gmm_full.compute_act_dcf(0.9)}\n')




model = calibration(pi=0.5)
model.fit(scores_train_gmm, labels_train_gmm)
gmm_cal_scores = model.predict(scores_gmm)
gmm_cal_labels = y_test
dcf_cal_gmm = DCF(cfn,cfp, gmm_cal_scores, gmm_cal_labels)


model = calibration(pi=0.5)
model.fit(scores_train_svm, labels_train_svm)
svm_cal_scores = model.predict(scores_svm)
svm_cal_labels = y_test
dcf_cal_svm = DCF(cfn,cfp, svm_cal_scores, svm_cal_labels)


model = calibration(pi=0.5)
model.fit(scores_train_gmm_full, labels_train_gmm_full)
gmm_cal_scores_full = model.predict(scores_gmm_full)
gmm_cal_labels_full = y_test
dcf_cal_gmm_full = DCF(cfn,cfp, gmm_cal_scores_full, gmm_cal_labels_full)

print(f'gmm tied min dcf (calibrated): {dcf_cal_gmm.compute_min_dcf(0.5)}')
print(f'gmm full min dcf (calibrated): {dcf_cal_gmm_full.compute_min_dcf(0.5)}')
print(f'svm min dcf (calibrated): {dcf_cal_svm.compute_min_dcf(0.5)}')

print(f'gmm tied min dcf (not-calibrated): {dcf_gmm.compute_min_dcf(0.5)}')
print(f'gmm full min dcf (not-calibrated): {dcf_cal_gmm_full.compute_min_dcf(0.5)}')
print(f'svm min dcf (not-calibrated): {dcf_svm.compute_min_dcf(0.5)}')


'''
Now we plot both calibrated and not calibrated scores.
'''



save_name = 'part7_calibrated'
plt.figure(figsize=(14, 7), dpi=100)
p = np.linspace(-3, 3, 22)
plt.subplot(1,2,1)
plt.plot(p, dcf_gmm.bayes_error_plot(p_array=p, min_cost=False), color='darkred', label='GMM tied - act_dcf')
plt.plot(p, dcf_gmm.bayes_error_plot(p_array=p, min_cost=True), color='darkred',linestyle='dashed', label='GMM tied - min_dcf')
plt.plot(p, dcf_svm.bayes_error_plot(p_array=p, min_cost=False), color='darkblue', label='SVM - act_dcf')
plt.plot(p, dcf_svm.bayes_error_plot(p_array=p, min_cost=True), color='darkblue',linestyle='dashed', label='SVM - min_dcf')
plt.plot(p, dcf_gmm_full.bayes_error_plot(p_array=p, min_cost=False), color='darkorange', label='GMM full - act_dcf')
plt.plot(p, dcf_gmm_full.bayes_error_plot(p_array=p, min_cost=True), color='darkorange',linestyle='dashed', label='GMM full - min_dcf')
plt.xlabel(r'{log($\frac{\pi}{1-\pi}$)}')
plt.ylabel("DCF")
plt.tight_layout()
plt.legend()
plt.grid(linestyle='--')
plt.subplot(1,2,2)
plt.plot(p, dcf_cal_gmm.bayes_error_plot(p_array=p, min_cost=False), color='darkred', label='GMM  tied (cal) - act_dcf')
plt.plot(p, dcf_cal_gmm_full.bayes_error_plot(p_array=p, min_cost=False), color='darkorange', label='GMM  full (cal) - act_dcf')
plt.plot(p, dcf_cal_svm.bayes_error_plot(p_array=p, min_cost=False), color='darkblue', label='SVM (cal) - act_dcf')
plt.plot(p, dcf_cal_gmm.bayes_error_plot(p_array=p, min_cost=True), color='darkred',linestyle='dashed', label='GMM tied (cal) - min_dcf')
plt.plot(p, dcf_cal_gmm_full.bayes_error_plot(p_array=p, min_cost=True), color='darkorange',linestyle='dashed', label='GMM full (cal) - min_dcf')
plt.plot(p, dcf_cal_svm.bayes_error_plot(p_array=p, min_cost=True), color='darkblue',linestyle='dashed', label='SVM (cal) - min_dcf')
plt.xlabel(r'{log($\frac{\pi}{1-\pi}$)}')
plt.ylabel("DCF")
plt.tight_layout()
plt.legend()
plt.grid(linestyle='--')
plt.savefig(os.path.join(figs_dir, save_name))





# =============================================================================
# Fusion
# =============================================================================



p = np.linspace(-3, 3, 22)
save_name = 'part7_calibrated_and_fusion'
plt.figure(figsize=(7, 7), dpi=100)
plt.plot(p, dcf_cal_gmm.bayes_error_plot(p_array=p, min_cost=False), color='darkred', label='GMM tied (cal) - act_dcf')
plt.plot(p, dcf_cal_gmm.bayes_error_plot(p_array=p, min_cost=True), color='darkred',linestyle='dashed', label='GMM tied (cal) - min_dcf')
plt.plot(p, dcf_cal_gmm_full.bayes_error_plot(p_array=p, min_cost=False), color='darkorange', label='GMM full (cal) - act_dcf')
plt.plot(p, dcf_cal_gmm_full.bayes_error_plot(p_array=p, min_cost=True), color='darkorange',linestyle='dashed', label='GMM full (cal) - min_dcf')
plt.plot(p, dcf_cal_svm.bayes_error_plot(p_array=p, min_cost=False), color='darkblue', label='SVM (cal) - act_dcf')
plt.plot(p, dcf_cal_svm.bayes_error_plot(p_array=p, min_cost=True), color='darkblue',linestyle='dashed', label='SVM (cal) - min_dcf')
plt.plot(p, dcf_fusion.bayes_error_plot(p_array=p, min_cost=False), color='darkgreen', label='Fusion - act_dcf')
plt.plot(p, dcf_fusion.bayes_error_plot(p_array=p, min_cost=True), color='darkgreen',linestyle='dashed', label='Fusion (GMM full, GMM tied) - min_dcf')
plt.xlabel(r'{log($\frac{\pi}{1-\pi}$)}')
plt.xlabel(r'log($\frac{\pi}{1-\pi}$)')
plt.ylabel("DCF")
plt.tight_layout()
plt.legend()
plt.grid(linestyle='--')
plt.savefig(os.path.join(figs_dir, save_name))



# =============================================================================
# First calibration (using minimum threshold for actDCF)
# =============================================================================

save_name = 'part7_calibrated_threshold'
check_path(os.path.join(tables_dir,save_name))
with open(os.path.join(tables_dir, save_name), 'a') as f:
    f.write('\n\n')
    f.write('\t\t\t\t\tcalibration threshold estimation\n')
    for prior in priors:
        svm = thresh_calibration(scores_train_svm, labels_train_svm, scores_svm, labels_svm, pi=prior)
        gmm = thresh_calibration(scores_train_gmm, labels_train_gmm, scores_gmm, labels_gmm, pi=prior)
        gmm_full = thresh_calibration(scores_train_gmm_full, labels_train_gmm_full, scores_gmm_full, labels_gmm_full, pi=prior)
        f.write(f'prior {prior}\t\t\n\n')
        f.write(f'SVM \t\t min_dcf: {svm[0]} \t\t\t act_dcf(emprical): {svm[1]} \t\t\t act_dcf(min_threshold): {svm[2]} \t\t\t\n')
        f.write(f'GMM tied \t\t min_dcf: {gmm[0]} \t\t\t act_dcf(emprical): {gmm[1]} \t\t\t act_dcf(min_threshold): {gmm[2]} \t\t\t\n')
        f.write(f'GMM full \t\t min_dcf: {gmm_full[0]} \t\t\t act_dcf(emprical): {gmm_full[1]} \t\t\t act_dcf(min_threshold): {gmm_full[2]} \t\t\t\n')
        f.write('\n\n')
        f.write('------------------------------------------------------------------------------------')
        f.write('\n\n')

        
save_name = 'part7_final'
check_path(os.path.join(tables_dir,save_name))
with open(os.path.join(tables_dir, save_name), 'a') as f:
    f.write('\n\n')
    f.write('\t\t\t\t\tcalibration threshold estimation\n')
    for prior in priors:
        f.write(f'prior {prior}\t\t\n\n')
        f.write(f'SVM (cal) \t\t min_dcf: {dcf_cal_gmm.compute_min_dcf(prior)} \t\t\t act_dcf: {dcf_cal_gmm.compute_act_dcf(prior)}\n')
        f.write(f'GMM tied (cal) \t\t min_dcf: {dcf_cal_svm.compute_min_dcf(prior)} \t\t\t act_dcf: {dcf_cal_svm.compute_act_dcf(prior)}\n')
        f.write(f'GMM full (cal) \t\t min_dcf: {dcf_cal_gmm_full.compute_min_dcf(prior)} \t\t\t act_dcf: {dcf_cal_gmm_full.compute_act_dcf(prior)}\n')
        f.write(f'Fusion \t\t min_dcf: {dcf_fusion.compute_min_dcf(prior)} \t\t\t act_dcf: {dcf_fusion.compute_act_dcf(prior)}\n')
        f.write('\n\n')
        f.write('------------------------------------------------------------------------------------')
        f.write('\n\n')
