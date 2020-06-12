#!/bin/bash

declare -a models=("SimpleCNN" "ResNet50" "DenseNet121" "Xception")
for model in "${models[@]}"
do
    python evaluate.py cifar10 "$model" --eps .03
    python attack.py cifar10 "$model" --eps .03
done
