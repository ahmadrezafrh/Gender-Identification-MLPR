#!/bin/bash

cd results/analysis
if [ "$1" = "-b" ]; then
    shift
    for arg in "$@"; do
        if [ "$arg" = "test" ] || [ "$arg" = "train" ]; then
            for mode in "$arg" test; do
                cd "$mode"
                python3 data.py
                python3 "$2".py
    		    cd ..
            done
        else
            echo "Invalid argument: $arg. Skipping..."
        fi
    done
else
    echo "Invalid arguments. Please provide '-t' followed by a list of 'test' or 'train' arguments."
fi