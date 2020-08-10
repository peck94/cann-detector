#!/bin/bash

declare -a models=("SimpleCNN" "ResNet50" "DenseNet121" "Xception")
for model in "${models[@]}"
do
    python evaluate.py mnist "$model" --eps .3
    python attack.py mnist "$model" --eps .3
    python mahalanobis.py mnist "$model" --eps .3
    python deepknn.py mnist "$model" --eps .3
done
