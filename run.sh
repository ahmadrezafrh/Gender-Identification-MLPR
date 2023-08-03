#!/bin/bash

if [ "$1" = "-t" ]; then
    shift
    for arg in "$@"; do
        if [ "$arg" = "test" ] || [ "$arg" = "train" ]; then
            for mode in "$arg" test; do
                cd "$mode"
                python3 gmm.py part6_1_gmm.json
    		    python3 lr_linear.py part2_1_lr_linear.json
    		    python3 lr_quad.py part2_3_lr_quad.json
    		    python3 mvg.py part1_1_mvg.json
    		    python3 mvg.py part1_2_mvg.json
    		    python3 svm_linear.py part3_1_svm_linear.json
    		    python3 svm_polynomial.py part5_1_poly.json
    		    python3 svm_rbf.py part4_1_rbf.json
    		    python3 lr_linear.py part2_2_lr_linear.json
    		    python3 svm_rbf.py part4_2_rbf.json
    		    python3 svm_polynomial.py part5_2_poly.json
    		    python3 svm_linear.py part3_2_svm_linear.json
    		    python3 lr_quad.py part2_4_lr_quad.json
    		    cd ..
            done
        else
            echo "Invalid argument: $arg. Skipping..."
        fi
    done
else
    echo "Invalid arguments. Please provide '-t' followed by a list of 'test' or 'train' arguments."
fi
