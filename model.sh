#!/bin/bash

# data 다운
sh data.sh

# 먼저 data 처리
python3 /root/code/rawboost/main.py

# unlabel csv 만들기
python3 /root/code/experiment/unlabel.py

# test masking
python3 /root/code/experiment/no_speech.py

# training
python3 /root/code/aasist/main.py --config /root/code/aasist/config/AASIST.conf

# ensemble
python3 /root/code/experiment/ensemble.py

# weight ensemble
python3 /root/code/experiment/weight_ensemble.py