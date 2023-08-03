# -*- coding: utf-8 -*-

# =============================================================================
# Created By    : Ahmadreza Farmahini Farahani
# Created Date  : 2023/4
# Project       : This project is developed for "Machin Learnin and Pattern Recognition" course
# Supervisor    : Prof. Sandro Cumani
# Universsity   : Politecnico di Torino
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(1, './../../')

from utils import read_data
from utils import plot_hist
from preprocess import normalization
from preprocess import gaussianization


if __name__ == "__main__":
    
    root_dir = './../../results/figs/train'
    gauss_name = 'part0_distribution_gauss'
    heatmap_name = 'part0_heatmap'
    zscore_name = 'part0_distribution_zscore'
    raw_name = 'part0_distribution_raw'
    
    print("Calculating heatmap plots for train data...")
    x, y = read_data(file_name= "./../../data/Train.txt")
    n_features = x.shape[0]
    corr_all = np.corrcoef(x)
    corr_1 = np.corrcoef(x[:, y[0]==1])
    corr_0 = np.corrcoef(x[:, y[0]==0])
    plt.figure(figsize=(12, 16), dpi=60)
    plt.subplot(1,3,1)
    plt.imshow(corr_0, cmap='YlGn', interpolation='nearest')
    plt.subplot(1,3,2)
    plt.imshow(corr_1, cmap='PuBu', interpolation='nearest')
    plt.subplot(1,3,3)
    plt.imshow(corr_all, cmap='Reds', interpolation='nearest')
    plt.savefig(os.path.join(root_dir, heatmap_name))
    plt.tight_layout()

    
    
    
    print("Calculating histogram plots for raw distributions...")
    plot_hist(x,y,n_features, save_path=os.path.join(root_dir, raw_name))
    
    print("Calculating histogram plots for z-score distributions...")
    norm = normalization()
    x_norm = norm.fit_transform(x)
    plot_hist(x_norm,y,n_features, save_path=os.path.join(root_dir, zscore_name))

    print("Calculating histogram plots for gaussianized distributions...")
    gauss = gaussianization()
    x_gauss = gauss.fit_transform(x)
    plot_hist(x_gauss,y,n_features, save_path=os.path.join(root_dir, gauss_name))
