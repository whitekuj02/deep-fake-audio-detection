#!/bin/bash

mkdir -p "/root/code/4_ensemble/input" \
         "/root/code/4_ensemble/output"

python ensemble.py --input_path "/root/code/4_ensemble/input" --output_path "/root/code/4_ensemble/output"

# python weighted_ensemble.py