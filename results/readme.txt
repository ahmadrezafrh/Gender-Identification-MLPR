This folder contains all the results during training. We have two kind of analysis
over training data. One for analysis if the trained results and another for analysis
of test data. The training results are based on the model trained only on training data
and validation on a part of training data. The test results are based on training on the 
training data and testing on the test data. The test results are part of the experimental
analysis. With the test results we want to see how much are training results assumptions 
and beliefs are correct about an unknown application.


below tree describes all the directories.

├── analysis
│   ├── readme.txt
│   ├── test.py
│   └── train.py
├── configues
│   ├── test
│   │   ├── part1_1_mvg.json
│   │   ├── part1_2_mvg.json
│   │   ├── part2_1_lr.json
│   │   ├── part2_2_lr.json
│   │   ├── part3_1_svm_linear.json
│   │   ├── part3_2_svm_linear.json
│   │   ├── part4_1_rbf.json
│   │   ├── part4_2_rbf.json
│   │   ├── part5_1_poly.json
│   │   ├── part5_2_poly.json
│   │   └── part6_1_gmm.json
│   └── train
│       ├── part1_1_mvg.json
│       ├── part1_2_mvg.json
│       ├── part2_1_lr.json
│       ├── part2_2_lr.json
│       ├── part3_1_svm_linear.json
│       ├── part3_2_svm_linear.json
│       ├── part4_1_rbf.json
│       ├── part4_2_rbf.json
│       ├── part5_1_poly.json
│       ├── part5_2_poly.json
│       └── part6_1_gmm.json
├── figs
│   ├── test
│   └── train
│       ├── part0_distributions.png
│       ├── part0_heatmaps_all.png
│       ├── part0_normalized_distributions.png
│       ├── part2_noPCA_upKFOLDS_leftNORMALIZED.png
│       ├── part2_PCA8_upKFOLDS_leftNORMALIZED.png
│       ├── part3_notNORMALIZED_upKFOLDS_leftPCA8.png
│       ├── part4_notNORMALIZED_upKFOLDS_leftPCA8.png
│       ├── part5_notNORMALIZED_upKFOLDS_leftPCA8.png
│       ├── part6_notNORMALIZED_upFULL_rightTIED.png
│       ├── part7_1_rbf_kfolds.png
│       ├── part7_2_gmm.png
│       └── part7_3_calibrated_gmm_svm.png
|
├── tables
│   ├── test
│   └── train
│       ├── part1_1_mvg.txt
│       ├── part2_2_lr.txt
│       ├── part3_2_svm_linear.txt
│       ├── part4_2_rbf.txt
│       ├── part5_2_poly.txt
│       ├── part6_1_gmm.txt
│       ├── part7_calib_best.txt
│       └── part7_calibrated_scores.txt
├── test
│   ├── part1_1_mvg.results
│   ├── part1_2_mvg.results
│   ├── part2_1_lr.results
│   ├── part2_2_lr.results
│   ├── part3_1_svm_linear.results
│   ├── part3_2_svm_linear.results
│   ├── part4_1_rbf.results
│   ├── part4_2_rbf.results
│   ├── part5_1_poly.results
│   └── part6_1_gmm.results
└── train
    ├── part1_1_mvg.results
    ├── part1_2_mvg.results
    ├── part2_1_lr.results
    ├── part2_2_lr.results
    ├── part3_1_svm_linear.results
    ├── part3_2_svm_linear.results
    ├── part4_1_rbf.results
    ├── part4_2_rbf.results
    ├── part5_1_poly_norm.results
    ├── part5_1_poly.results
    ├── part5_2_poly.results
    └── part6_1_gmm.results