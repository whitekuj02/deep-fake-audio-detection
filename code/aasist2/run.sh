#!/bin/bash

mkdir ./experiments
mkdir ./output

python train.py --data_path "/root/data/aasist/denoise"
python test.py --model_path "./experiments/swa_params.pt" --data_path "/root/data/aasist/denoise"
