#!/bin/bash

declare -a models=("SimpleCNN" "ResNet50" "DenseNet121" "Xception")
for model in "${models[@]}"
do
    python evaluate.py fashion "$model" --eps .1
    python attack.py fashion "$model" --eps .1
    python mahalanobis.py fashion "$model" --eps .1
    python deepknn.py fashion "$model" --eps .1
done
