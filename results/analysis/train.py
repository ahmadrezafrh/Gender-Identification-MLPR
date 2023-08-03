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
import warnings
import matplotlib.pyplot as plt
import matplotlib.ticker
import pandas as pd
sys.path.insert(1, './../../')
pd.set_option('max_colwidth', 800)

from utils import check_path
from utils import load_results
from utils import read_data
from utils import vrow
from utils import k_folds, single_fold
from postprocess import DCF
from postprocess import calibration, thresh_calibration
from postprocess import fusion
from classifiers import SVM, GMM

split_ratio = 0.8
n_folds = 4
n_features = 12
cfn = 1
cfp = 1
train_path = "./../../data/Train.txt"
results_dir = './../../results'


tables_dir = os.path.join(results_dir, "tables/train")
figs_dir = os.path.join(results_dir, "figs/train")
train_results_dir = os.path.join(results_dir, "train")


mvg1_1_path = os.path.join(train_results_dir, "part1_1_mvg.results")
mvg1_2_path = os.path.join(train_results_dir, "part1_2_mvg.results")
lr2_1_path = os.path.join(train_results_dir, "part2_1_lr_linear.results")
lr2_2_path = os.path.join(train_results_dir, "part2_2_lr_linear.results")
lr2_3_path = os.path.join(train_results_dir, "part2_3_lr_quad.results")
lr2_4_path = os.path.join(train_results_dir, "part2_4_lr_quad.results")
svm3_1_path = os.path.join(train_results_dir, "part3_1_svm_linear.results")
svm3_2_path = os.path.join(train_results_dir, "part3_2_svm_linear.results")
svm4_1_path = os.path.join(train_results_dir, "part4_1_rbf.results")
svm4_2_path = os.path.join(train_results_dir, "part4_2_rbf.results")
svm5_1_path = os.path.join(train_results_dir, "part5_1_poly.results")
svm5_2_path = os.path.join(train_results_dir, "part5_2_poly.results")
gmm6_1_path = os.path.join(train_results_dir, "part6_1_gmm.results")

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
    kfolds_zscore = models[(models['pca']==pca) & (models['folds']=='k-folds') & (models['normalization']=='z-score')]
    single_gauss = models[(models['pca']==pca) & (models['folds']=='single-fold') & (models['normalization']=='gaussianization')]
    kfolds_gauss = models[(models['pca']==pca) & (models['folds']=='k-folds') & (models['normalization']=='gaussianization')]
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
    kfolds = models[(models['pca']==pca) & (models['folds']=='k-folds')]
    with open(os.path.join(tables_dir,save_name), 'a') as f:
        f.write('\t\t\t\t\tNOT_NORMALIZED\n')
        f.write(f'\t\t\t\t\tpca: {pca}\n')
        f.write('\t\t\t\t\tsingle fold\n')
        f.write(single[['model', 'min_dcf']].to_string(index=False))
        f.write('\n')
        f.write('\n\t\t\t\t\tk-folds\n')
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
save_name = 'part2_linear_upZ_midG_downR_leftKFOLDS'
models = pd.DataFrame(lr)
models['lambda'] = models['lambda'].map(lambda x: float(x))
models = models[models['pca']==False]
plt.figure(figsize=(10, 13), dpi=100)
m=0
for norm in models.normalization.unique():
    mods = models[models['normalization']==norm]
    for fold in mods.folds.unique():
        model = mods[(mods['normalization']==norm) & (mods['folds']==fold)]
        plt.subplot(3,2,m+1)
        plt.plot(model['lambda'], model['min_dcf'].map(lambda x: x[0.5]), color='darkred', label='minDcf ($\pi$ = 0.5)')
        plt.plot(model['lambda'], model['min_dcf'].map(lambda x: x[0.1]), color='darkblue', label='minDcf ($\pi$ = 0.1)')
        plt.plot(model['lambda'], model['min_dcf'].map(lambda x: x[0.9]), color='darkgreen', label='minDcf ($\pi$ = 0.9)')
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
models = models[models['folds']=='k-folds']
models['balance'] = models['balance'].map(lambda x: ast.literal_eval(x))
models.style.hide_index()
check_path(os.path.join(tables_dir,save_name))
for norm in models.normalization.unique():
    with open(os.path.join(tables_dir, save_name), 'a') as f:
        f.write(f'\t\t\t___________________normalization: {norm}________________________\n')
    for pca in models.pca.unique():
        mods = models[(models['pca']==pca) & (models['normalization']==norm)]
        
        with open(os.path.join(tables_dir, save_name), 'a') as f:
            f.write('\t\t\t\t\t4-folds\n')
            f.write(f'\t\t\t\t\traw features - pca: {pca}\n')
            f.write(mods[['lambda', 'balance', 'min_dcf']].to_string(index=False))
            f.write('\n\n')
            f.write('------------------------------------------------------------------------------------')
            f.write('\n\n') 

# =============================================================================
# In the below codes we analyse the results of Linear Regression models.
# =============================================================================

path = lr2_3_path
print(f'analyzing {path.split("/")[-1]}...')
save_name = 'part2_quad_upZ_midG_downR_leftKFOLDS'
lr = load_results(path)
models = pd.DataFrame(lr)
models['lambda'] = models['lambda'].map(lambda x: float(x))
models = models[models['pca']==False]
plt.figure(figsize=(10, 13), dpi=100)
m=0
for norm in models.normalization.unique():
    mods = models[models['normalization']==norm]
    for fold in mods.folds.unique():
        model = mods[(mods['normalization']==norm) & (mods['folds']==fold)]
        plt.subplot(3,2,m+1)
        plt.plot(model['lambda'], model['min_dcf'].map(lambda x: x[0.5]), color='darkred', label='minDcf ($\pi$ = 0.5)')
        plt.plot(model['lambda'], model['min_dcf'].map(lambda x: x[0.1]), color='darkblue', label='minDcf ($\pi$ = 0.1)')
        plt.plot(model['lambda'], model['min_dcf'].map(lambda x: x[0.9]), color='darkgreen', label='minDcf ($\pi$ = 0.9)')
        plt.tight_layout()
        plt.xscale('log' )
        plt.xlabel('λ')
        plt.ylabel("DCF")
        plt.legend(loc ="upper left")
        plt.grid(linestyle='--')
        m+=1
plt.savefig(os.path.join(figs_dir, save_name))


path = lr2_4_path
save_name = path.split('/')[-1].split('.')[0] + '.txt'
lr = load_results(path)
models = pd.DataFrame(lr)
models['min_dcf'] = models['min_dcf'].map(lambda x: {0.5: round(x[0.5], 3), 0.9: round(x[0.9], 3), 0.1: round(x[0.1], 3)})
models = models[models['folds']=='k-folds']
models['balance'] = models['balance'].map(lambda x: ast.literal_eval(x))
models.style.hide_index()
check_path(os.path.join(tables_dir,save_name))
for norm in models.normalization.unique():
    with open(os.path.join(tables_dir, save_name), 'a') as f:
        f.write(f'\t\t\t___________________normalization: {norm}________________________\n')
    for pca in models.pca.unique():
        mods = models[(models['pca']==pca) & (models['normalization']==norm)]
        
        with open(os.path.join(tables_dir, save_name), 'a') as f:
            f.write('\t\t\t\t\t4-folds\n')
            f.write(f'\t\t\t\t\traw features - pca: {pca}\n')
            f.write(mods[['lambda', 'balance', 'min_dcf']].to_string(index=False))
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
for norm in models.normalization.unique():
    mods = models[models['normalization']==norm]
    m=0
    plt.figure(figsize=(11, 11), dpi=100)
    save_name = f'part3_upKFOLDS_rightIMB_{norm}'
    for fold in mods.folds.unique():
        for balance in mods.balance.unique():
            model = mods[(mods['balance']==balance) & (mods['folds']==fold)]
            
            plt.subplot(2,2,m+1)
            plt.plot(model['c'], model['min_dcf'].map(lambda x: x[0.5]), color='darkred', label='minDcf ($\pi$ = 0.5)')
            plt.plot(model['c'], model['min_dcf'].map(lambda x: x[0.1]), color='darkblue', label='minDcf ($\pi$ = 0.1)')
            plt.plot(model['c'], model['min_dcf'].map(lambda x: x[0.9]), color='darkgreen', label='minDcf ($\pi$ = 0.9)')
            plt.tight_layout()
            plt.xscale('log' )
            plt.xlabel('C')
            plt.ylabel("DCF")
            plt.legend()
            plt.grid(linestyle='--')
            m+=1
    plt.savefig(os.path.join(figs_dir, save_name))

path = svm3_2_path
print(f'analyzing {path.split("/")[-1]}...')
save_name = path.split('/')[-1].split('.')[0] + '.txt'
svm = load_results(path)
models = pd.DataFrame(svm)

models['min_dcf'] = models['min_dcf'].map(lambda x: {0.5: round(x[0.5], 3), 0.9: round(x[0.9], 3), 0.1: round(x[0.1], 3)})
models = models[models['folds']=='k-folds']
models['balance'] = models['balance'].map(lambda x: ast.literal_eval(x))
models.style.hide_index()
check_path(os.path.join(tables_dir,save_name))
for norm in models.normalization.unique():
    mods = models[models['normalization']==norm]
    with open(os.path.join(tables_dir, save_name), 'a') as f:
        f.write('\t\t\t\t\t4-folds\n')
        f.write(f'\t\t\t\t\t{norm} - pca: No PCA\n')
        f.write(mods[['c','balance', 'min_dcf']].to_string(index=False))
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
models['c'] = models['c'].map(lambda x: float(x))
models['min_dcf'] = models['min_dcf'].map(lambda x: x[0.5])
models = models[models['pca']==False]
for norm in models.normalization.unique():
    plt.figure(figsize=(12, 6), dpi=100)
    save_name = f'part4_leftKFOLDS_{norm}'
    m=0 
    for fold in models.folds.unique():
        model = models[(models['normalization']==norm) & (models['folds']==fold)]
        
        plt.subplot(1,2,m+1)
        plt.plot(model[model['gamma']=='0.0001']['c'], model[model['gamma']=='0.0001']['min_dcf'], color='sienna', label='$\log\gamma$ = -4')
        plt.plot(model[model['gamma']=='0.001']['c'], model[model['gamma']=='0.001']['min_dcf'], color='tan', label='$\log\gamma$ = -3')
        plt.plot(model[model['gamma']=='0.01']['c'], model[model['gamma']=='0.01']['min_dcf'], color='teal', label='$\log\gamma$ = -2')
        plt.plot(model[model['gamma']=='0.1']['c'], model[model['gamma']=='0.1']['min_dcf'], color='gray', label='$\log\gamma$ = -1')
        plt.tight_layout()
        plt.grid(linestyle='--')
        plt.xscale('log' )
        plt.xlabel('C')
        plt.ylabel("DCF")
        plt.legend()
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
for norm in models.normalization.unique():
    m=0
    plt.figure(figsize=(12, 6), dpi=100)
    save_name = f'part5_leftKFOLDS_{norm}'
    for fold in models.folds.unique():
        model = models[(models['normalization']==norm) & (models['folds']==fold)]
        plt.subplot(1,2,m+1)
        plt.plot(model[model['degree']=='2']['c'], model[model['degree']=='2']['min_dcf'], color='tan', label='$degree$ = 2')
        plt.plot(model[model['degree']=='3']['c'], model[model['degree']=='3']['min_dcf'], color='teal', label='$degree$ = 3')
        plt.plot(model[model['degree']=='4']['c'], model[model['degree']=='4']['min_dcf'], color='gray', label='$degree$ = 4')
        plt.tight_layout()
        plt.grid(linestyle='--')
        plt.xscale('log' )
        plt.xlabel('C')
        plt.ylabel("DCF")
        plt.legend()
        m+=1
    plt.savefig(os.path.join(figs_dir, save_name))


path = svm5_1_path
print(f'analyzing {path.split("/")[-1]}...')
svm = load_results(path)
models = pd.DataFrame(svm)
models = models[(models['balance']=="['imbalanced', 0.5]") & (models['degree']=='2')]
models['c'] = models['c'].map(lambda x: float(x))
for norm in models.normalization.unique():
    mods = models[models['normalization']==norm]
    m=0
    plt.figure(figsize=(12, 6), dpi=100)
    save_name = f'part5_leftKFOLDS_pi_{norm}'
    for fold in mods.folds.unique():
        model = mods[mods['folds']==fold]
        plt.subplot(1,2,m+1)
        plt.plot(model['c'], model['min_dcf'].map(lambda x: x[0.5]), color='darkred', label='minDcf ($\pi$ = 0.5)')
        plt.plot(model['c'], model['min_dcf'].map(lambda x: x[0.1]), color='darkblue', label='minDcf ($\pi$ = 0.1)')
        plt.plot(model['c'], model['min_dcf'].map(lambda x: x[0.9]), color='darkgreen', label='minDcf ($\pi$ = 0.9)')
        plt.tight_layout()
        plt.xscale('log' )
        plt.xlabel('C')
        plt.ylabel("DCF")
        plt.legend()
        plt.grid(linestyle='--')
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
for norm in models.normalization.unique():
    with open(os.path.join(tables_dir, save_name), 'a') as f:
        f.write(f'\t\t\t___________________normalization: {norm}________________________\n')
    model = models[models['normalization']==norm]
    with open(os.path.join(tables_dir, save_name), 'a') as f:
        f.write('\t\t\t\t\t4-folds\n')
        f.write('\t\t\t\t\traw features - pca: False\n')
        f.write(model[['balance', 'degree', 'degree', 'c', 'min_dcf']].to_string(index=False))
        f.write('\n\n\n')
    with open(os.path.join(tables_dir, save_name), 'a') as f:   
        f.write('------------------------------------------------------------------------------------')
        f.write('\n\n') 


# =============================================================================
# In the below codes we analyse the results of GMM.
# =============================================================================


path = gmm6_1_path
print(f'analyzing {path.split("/")[-1]}...')
svm = load_results(path)
models = pd.DataFrame(svm)
models['min_dcf'] = models['min_dcf'].map(lambda x: x[0.5])
for norm in models.normalization.unique():
    m=0
    save_name = f'part6_upFULL_rightTIED_{norm}'
    plt.figure(figsize=(13, 11), dpi=100)
    for model in models.model.unique():
        mod = models[(models['model']==model) & (models['normalization']==norm)]
        width = 0.15
        x = np.arange(len(mod[mod['pca']==8]['n_component']))
        plt.subplot(2,2,m+1)
        plt.bar(x - width/2, mod[mod['pca']==8]['min_dcf'], width, label='minDcf ($\pi$ = 0.5) - PCA (m=8)',facecolor = 'darkcyan', edgecolor='#169acf')
        plt.bar(x + width/2, mod[mod['pca']==False]['min_dcf'], width, label='minDcf ($\pi$ = 0.5) - without PCA',facecolor = 'darkkhaki', edgecolor='#169acf')
        plt.xticks(x, mod[mod['pca']==8]['n_component'])
        plt.tight_layout()
        plt.xlabel('GMM Components')
        plt.ylabel("DCF")
        plt.grid(linestyle='--')
        plt.legend()
        m+=1
    plt.savefig(os.path.join(figs_dir, save_name))




path = gmm6_1_path
save_name = path.split('/')[-1].split('.')[0] + '.txt'
svm = load_results(path)
models = pd.DataFrame(svm)
models['min_dcf'] = models['min_dcf'].map(lambda x: {0.5: round(x[0.5], 3), 0.9: round(x[0.9], 3), 0.1: round(x[0.1], 3)})
models.style.hide_index()
models['n_component'] = models['n_component'].map('n_component: {}'.format)
check_path(os.path.join(tables_dir,save_name))
for norm in models.normalization.unique():
    with open(os.path.join(tables_dir, save_name), 'a') as f:
        f.write(f'\t\t\t___________________normalization: {norm}________________________\n')
    for pca in models.pca.unique():
        with open(os.path.join(tables_dir, save_name), 'a') as f:
            model = models[(models['pca']==pca) & (models['normalization']==norm)]
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
models_svm['act_dcf'] = models_svm['act_dcf'].map(lambda x: {0.5: round(x[0.5], 3), 0.9: round(x[0.9], 3), 0.1: round(x[0.1], 3)})
models_svm = models_svm[(models_svm['pca']==False) & (models_svm['normalization']==False)]
models_svm.style.hide_index()
models_svm['gamma'] = models_svm['gamma'].map('gamma: {}'.format)
models_svm['c'] = models_svm['c'].map('c: {}'.format)

path = gmm6_1_path
save_name = 'part7_calib_best.txt'
gmm = load_results(path)
models_gmm = pd.DataFrame(gmm)
models_gmm['min_dcf'] = models_gmm['min_dcf'].map(lambda x: {0.5: round(x[0.5], 3), 0.9: round(x[0.9], 3), 0.1: round(x[0.1], 3)})
models_gmm['act_dcf'] = models_gmm['act_dcf'].map(lambda x: {0.5: round(x[0.5], 3), 0.9: round(x[0.9], 3), 0.1: round(x[0.1], 3)})
models_gmm_tied = models_gmm[(models_gmm['pca']==False) & (models_gmm['model']=='full_tied') & (models_gmm['n_component']==4) & (models_gmm['normalization']==False)]
models_gmm_full = models_gmm[(models_gmm['pca']==False) & (models_gmm['model']=='full') & (models_gmm['n_component']==4) & (models_gmm['normalization']==False)]
models_gmm_tied.style.hide_index()
models_gmm_full.style.hide_index()
check_path(os.path.join(tables_dir,save_name))
model_svm = models_svm[models_svm['folds']=='k-folds']
models_gmm_tied = models_gmm_tied[models_gmm_tied['folds']=='k-folds']
model_gmm_full = models_gmm_full[models_gmm_full['folds']=='k-folds']
with open(os.path.join(tables_dir, save_name), 'a') as f:
    f.write('\t\t\t\t\traw features - pca: No pca\n')
    f.write('\n_GMM_full_tied')
    f.write(models_gmm_tied[['model', 'n_component', 'min_dcf' , 'act_dcf']].to_string(index=False))
    f.write('\n_GMM_full')
    f.write(model_gmm_full[['model', 'n_component', 'min_dcf' , 'act_dcf']].to_string(index=False))
    f.write('\n_SVM_')
    f.write(model_svm[['balance','gamma', 'c', 'min_dcf', 'act_dcf']].to_string(index=False))
    f.write('\n\n')
    f.write('------------------------------------------------------------------------------------')
    f.write('\n\n') 


'''
Now we have the results, we go for calibration.
in the below codes we collect scores of best gmm and svm models.
'''

x, y = read_data(file_name=train_path)
folds = k_folds(n_folds, x, y)
single_split = single_fold(split_ratio, x, y)


scores = []
labels = []
svm = SVM(kernel='rbf')
for fold in folds:
    x_train = fold[0]
    x_test = fold[2]
    y_train = fold[1]
    y_test = fold[3]           
    svm.fit(x_train, y_train, gamma=0.001, c=10) 
    scores.append(svm.predict(x_test))
    labels.append(y_test)
scores_svm = vrow(np.hstack(scores))
labels_svm = vrow(np.hstack(labels))
dcf_svm = DCF(cfn,cfp, scores_svm, labels_svm)


scores = []
labels = []
gmm = GMM(model='full', tied=True)
for fold in folds:
    x_train = fold[0]
    x_test = fold[2]
    y_train = fold[1]
    y_test = fold[3]           
    gmm.fit(x_train, y_train, n_components=4)
    scores.append(gmm.predict(x_test))
    labels.append(y_test)
scores_gmm = vrow(np.hstack(scores))
labels_gmm = vrow(np.hstack(labels))
dcf_gmm = DCF(cfn,cfp, scores_gmm, labels_gmm)

scores = []
labels = []
gmm = GMM(model='full', tied=False)
for fold in folds:
    x_train = fold[0]
    x_test = fold[2]
    y_train = fold[1]
    y_test = fold[3]           
    gmm.fit(x_train, y_train, n_components=4)
    scores.append(gmm.predict(x_test))
    labels.append(y_test)
scores_gmm_full = vrow(np.hstack(scores))
labels_gmm_full = vrow(np.hstack(labels))
dcf_gmm_full = DCF(cfn,cfp, scores_gmm_full, labels_gmm_full)


'''
Now we plot DET figure. The first fig is not cut. The
second one is cut for better visualization.
'''



fnrs_gmm, fprs_gmm = dcf_gmm.det_plot()
fnrs_svm, fprs_svm = dcf_svm.det_plot()
fnrs_gmm_full, fprs_gmm_full = dcf_gmm_full.det_plot()
save_name = 'part7_not_calibrated_det'
plt.figure(figsize=(7, 7), dpi=100)
plt.plot(fnrs_svm,fprs_svm, color='darkblue', label='SVM')
plt.plot(fnrs_gmm,fprs_gmm, color='darkred', label='GMM tied')
plt.plot(fnrs_gmm_full,fprs_gmm_full, color='darkorange', label='GMM full')
plt.title('DET plot')
plt.xlabel('False Positive Rate')
plt.ylabel("False Negative Rate")
ax = plt.gca()
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xticks([1,2,5,10,20,40])
ax.set_yticks([1,2,5,10,20,40])
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.get_xaxis().set_tick_params(which='minor', size=0)
ax.get_xaxis().set_tick_params(which='minor', width=0) 
plt.legend()
plt.xlim([1, 40])
plt.ylim(1, 40)
plt.grid(linestyle='--')
plt.tight_layout()
plt.savefig(os.path.join(figs_dir, save_name))





'''
The below figure is not calibrated scores bayes plot.
'''

p = np.linspace(-3, 3, 22)
save_name = 'part7_not_calibrated'
plt.figure(figsize=(18, 6), dpi=100)
plt.subplot(1,3,1)
plt.plot(p, dcf_gmm.bayes_error_plot(p_array=p, min_cost=False), color='darkred', label='GMM tied - act_dcf')
plt.plot(p, dcf_gmm.bayes_error_plot(p_array=p, min_cost=True), color='darkred',linestyle='dashed', label='GMM tied - min_dcf')
plt.xlabel(r'log($\frac{\pi}{1-\pi}$)')
plt.ylabel("DCF")
plt.legend()
plt.grid(linestyle='--')
plt.tight_layout()
plt.subplot(1,3,2)
plt.plot(p, dcf_svm.bayes_error_plot(p_array=p, min_cost=False), color='darkblue', label='SVM - act_dcf')
plt.plot(p, dcf_svm.bayes_error_plot(p_array=p, min_cost=True), color='darkblue',linestyle='dashed', label='SVM - min_dcf')
plt.xlabel(r'log($\frac{\pi}{1-\pi}$)')
plt.ylabel("DCF")
plt.legend()
plt.grid(linestyle='--')
plt.tight_layout()
plt.subplot(1,3,3)
plt.plot(p, dcf_gmm_full.bayes_error_plot(p_array=p, min_cost=False), color='darkorange', label='GMM full - act_dcf')
plt.plot(p, dcf_gmm_full.bayes_error_plot(p_array=p, min_cost=True), color='darkorange',linestyle='dashed', label='GMM full - min_dcf')
plt.xlabel(r'log($\frac{\pi}{1-\pi}$)')
plt.ylabel("DCF")
plt.legend()
plt.grid(linestyle='--')
plt.tight_layout()
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

scores_folds = k_folds(4, scores_svm, labels_svm)
for prior in priors:
    svm_cal_scores = []
    svm_cal_labels = []
    model = calibration(pi=prior)
    for fold in scores_folds:
        model.fit(fold[0], fold[1])
        svm_cal_scores.append(model.predict(fold[2]))
        svm_cal_labels.append(fold[3])
    svm_cal_scores = vrow(np.hstack(svm_cal_scores))
    svm_cal_labels = vrow(np.hstack(svm_cal_labels))
    dcf_cal_svm = DCF(cfn,cfp, svm_cal_scores, svm_cal_labels)
    with open(os.path.join(tables_dir, save_name), 'a') as f:
        f.write(f'log-reg-({prior}) \t\t act-dcf(0.5)={dcf_cal_svm.compute_act_dcf(0.5)}\t\t act-dcf(0.1)={dcf_cal_svm.compute_act_dcf(0.1)}\t\t act-dcf(0.9)={dcf_cal_svm.compute_act_dcf(0.9)}\n')
        
        
        
with open(os.path.join(tables_dir, save_name), 'a') as f:
    f.write('\n\n')
    f.write('------------------------------------------------------------------------------------')
    f.write('\n\n')
    f.write('\t\t\t\t\tcalibration - GMM\n')
    f.write('\t\t\t\t\traw features - pca: no pca\n\n\n')
    f.write(f'min-dcfs \t\t min-dcf(0.5)={dcf_gmm.compute_min_dcf(0.5)}\t\t min-dcf(0.1)={dcf_gmm.compute_min_dcf(0.1)}\t\t min-dcf(0.9)={dcf_gmm.compute_min_dcf(0.9)}\n')
    f.write(f'Not-calibrated \t\t act-dcf(0.5)={dcf_gmm.compute_act_dcf(0.5)}\t\t act-dcf(0.1)={dcf_gmm.compute_act_dcf(0.1)}\t\t act-dcf(0.9)={dcf_gmm.compute_act_dcf(0.9)}\n')

scores_folds = k_folds(4, scores_gmm, labels_gmm)
for prior in priors:
    gmm_cal_scores = []
    gmm_cal_labels = []
    model = calibration(pi=prior)
    for fold in scores_folds:
        model.fit(fold[0], fold[1])
        gmm_cal_scores.append(model.predict(fold[2]))
        gmm_cal_labels.append(fold[3])
    gmm_cal_scores = vrow(np.hstack(gmm_cal_scores))
    gmm_cal_labels = vrow(np.hstack(gmm_cal_labels))
    dcf_cal_gmm = DCF(cfn,cfp, gmm_cal_scores, gmm_cal_labels)
    with open(os.path.join(tables_dir, save_name), 'a') as f:
        f.write(f'log-reg-({prior}) \t\t act-dcf(0.5)={dcf_cal_gmm.compute_act_dcf(0.5)}\t\t act-dcf(0.1)={dcf_cal_gmm.compute_act_dcf(0.1)}\t\t act-dcf(0.9)={dcf_cal_gmm.compute_act_dcf(0.9)}\n')




scores_folds = k_folds(4, scores_gmm, labels_gmm)
gmm_cal_scores = []
gmm_cal_labels = []
model = calibration()
for fold in scores_folds:
    model.fit(fold[0], fold[1])
    gmm_cal_scores.append(model.predict(fold[2]))
    gmm_cal_labels.append(fold[3])
gmm_cal_scores = vrow(np.hstack(gmm_cal_scores))
gmm_cal_labels = vrow(np.hstack(gmm_cal_labels))
dcf_cal_gmm = DCF(cfn,cfp, gmm_cal_scores, gmm_cal_labels)


scores_folds = k_folds(4, scores_svm, labels_svm)
svm_cal_scores = []
svm_cal_labels = []
model = calibration()
for fold in scores_folds:
    model.fit(fold[0], fold[1])
    svm_cal_scores.append(model.predict(fold[2]))
    svm_cal_labels.append(fold[3])
svm_cal_scores = vrow(np.hstack(svm_cal_scores))
svm_cal_labels = vrow(np.hstack(svm_cal_labels))
dcf_cal_svm = DCF(cfn,cfp, svm_cal_scores, svm_cal_labels)

scores_folds = k_folds(4, scores_gmm_full, labels_gmm_full)
gmm_cal_scores_full = []
gmm_cal_labels_full = []
model = calibration()
for fold in scores_folds:
    model.fit(fold[0], fold[1])
    gmm_cal_scores_full.append(model.predict(fold[2]))
    gmm_cal_labels_full.append(fold[3])
gmm_cal_scores_full = vrow(np.hstack(gmm_cal_scores_full))
gmm_cal_labels_full = vrow(np.hstack(gmm_cal_labels_full))
dcf_cal_gmm_full = DCF(cfn,cfp, gmm_cal_scores_full, gmm_cal_labels_full)

print(f'gmm tied min dcf (calibrated): {dcf_cal_gmm.compute_min_dcf(0.5)}')
print(f'gmm full min dcf (calibrated): {dcf_cal_gmm_full.compute_min_dcf(0.5)}')
print(f'svm min dcf (calibrated): {dcf_cal_svm.compute_min_dcf(0.5)}')

print(f'gmm tied min dcf (not-calibrated): {dcf_gmm.compute_min_dcf(0.5)}')
print(f'gmm full min dcf (calibrated): {dcf_cal_gmm_full.compute_min_dcf(0.5)}')
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
plt.xlabel(r'log($\frac{\pi}{1-\pi}$)')
plt.ylabel("DCF")
plt.legend()
plt.grid(linestyle='--')
plt.tight_layout()
plt.subplot(1,2,2)
plt.plot(p, dcf_cal_gmm.bayes_error_plot(p_array=p, min_cost=False), color='darkred', label='GMM tied - act_dcf')
plt.plot(p, dcf_cal_svm.bayes_error_plot(p_array=p, min_cost=False), color='darkblue', label='SVM - act_dcf')
plt.plot(p, dcf_cal_gmm.bayes_error_plot(p_array=p, min_cost=True), color='darkred',linestyle='dashed', label='GMM tied - min_dcf')
plt.plot(p, dcf_cal_svm.bayes_error_plot(p_array=p, min_cost=True), color='darkblue',linestyle='dashed', label='SVM - min_dcf')
plt.plot(p, dcf_cal_gmm_full.bayes_error_plot(p_array=p, min_cost=True), color='darkorange',linestyle='dashed', label='GMM full - min_dcf')
plt.plot(p, dcf_cal_gmm_full.bayes_error_plot(p_array=p, min_cost=False), color='darkorange', label='GMM full - act_dcf')
plt.xlabel(r'log($\frac{\pi}{1-\pi}$)')
plt.ylabel("DCF")
plt.legend()
plt.grid(linestyle='--')
plt.tight_layout()
plt.savefig(os.path.join(figs_dir, save_name))





# =============================================================================
# Fusion
# =============================================================================

labels = labels_gmm
scores = np.vstack((scores_gmm_full, scores_gmm))
scores_folds = k_folds(4, scores, labels)
fusion_scores = []
fusion_labels = []
model = fusion()
for fold in scores_folds:
    model.fit(fold[0], fold[1])
    fusion_scores.append(model.predict(fold[2]))
    fusion_labels.append(fold[3])
fusion_scores = vrow(np.hstack(fusion_scores))
fusion_labels = vrow(np.hstack(fusion_labels))
dcf_fusion = DCF(cfn,cfp, fusion_scores, fusion_labels)
print(f'fusion min dcf: {dcf_fusion.compute_min_dcf(0.5)}')

fnrs_svm, fprs_svm = dcf_svm.det_plot()
fnrs_gmm, fprs_gmm = dcf_gmm.det_plot()
fnrs_fusion, fprs_fusion = dcf_fusion.det_plot()


save_name = 'part7_calibrated_det'
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
plt.ylim(1, 40)
plt.xlabel('False Positive Rate')
plt.ylabel("False Negative Rate")
plt.grid(linestyle='--')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(figs_dir, save_name))




p = np.linspace(-3, 3, 22)
save_name = 'part7_calibrated_and_fusion'
plt.figure(figsize=(7, 7), dpi=100)
plt.plot(p, dcf_cal_gmm.bayes_error_plot(p_array=p, min_cost=False), color='darkred', label='GMM tied (cal) - act_dcf')
plt.plot(p, dcf_cal_gmm.bayes_error_plot(p_array=p, min_cost=True), color='darkred',linestyle='dashed', label='GMM tied (cal) - min_dcf')
plt.plot(p, dcf_cal_svm.bayes_error_plot(p_array=p, min_cost=False), color='darkblue', label='SVM (cal) - act_dcf')
plt.plot(p, dcf_cal_svm.bayes_error_plot(p_array=p, min_cost=True), color='darkblue',linestyle='dashed', label='SVM (cal) - min_dcf')
plt.plot(p, dcf_cal_gmm_full.bayes_error_plot(p_array=p, min_cost=False), color='darkorange', label='GMM full (cal) - act_dcf')
plt.plot(p, dcf_cal_gmm_full.bayes_error_plot(p_array=p, min_cost=True), color='darkorange',linestyle='dashed', label='GMM full (cal) - min_dcf')
plt.plot(p, dcf_fusion.bayes_error_plot(p_array=p, min_cost=False), color='darkgreen', label='Fusion - act_dcf')
plt.plot(p, dcf_fusion.bayes_error_plot(p_array=p, min_cost=True), color='darkgreen',linestyle='dashed', label='Fusion (GMM full, GMM tied) - min_dcf')
plt.xlabel(r'log($\frac{\pi}{1-\pi}$)')
plt.ylabel("DCF")
plt.legend()
plt.grid(linestyle='--')
plt.tight_layout()
plt.savefig(os.path.join(figs_dir, save_name))



th=dcf_svm.compute_min_dcf_threshold(0.5)
dcf_svm.compute_min_dcf(0.5)
dcf_svm.compute_act_dcf(pi=0.5, th=th)

# =============================================================================
# First calibration (using minimum threshold for actDCF)
# =============================================================================
folds_svm = single_fold(split_ratio=0.8, x=scores_svm, y=labels_svm)
folds_gmm = single_fold(split_ratio=0.8, x=scores_gmm, y=labels_gmm)
folds_gmm_full = single_fold(split_ratio=0.8, x=scores_gmm_full, y=labels_gmm_full)
save_name = 'part7_calibrated_threshold'
check_path(os.path.join(tables_dir,save_name))
with open(os.path.join(tables_dir, save_name), 'a') as f:
    f.write('\n\n')
    f.write('\t\t\t\t\tcalibration threshold estimation\n')
    for prior in priors:
        svm = thresh_calibration(folds_svm[0], folds_svm[1], folds_svm[2], folds_svm[3], pi=prior)
        gmm = thresh_calibration(folds_gmm[0], folds_gmm[1], folds_gmm[2], folds_gmm[3], pi=prior)
        gmm_full = thresh_calibration(folds_gmm_full[0], folds_gmm_full[1], folds_gmm_full[2], folds_gmm_full[3], pi=prior)
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
