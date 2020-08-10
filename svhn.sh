#!/bin/bash

declare -a models=("SimpleCNN" "ResNet50" "DenseNet121" "Xception")
for model in "${models[@]}"
do
    python evaluate.py svhn "$model" --eps .03
    python attack.py svhn "$model" --eps .03
    python mahalanobis.py svhn "$model" --eps .03
    python deepknn.py svhn "$model" --eps .03
done
