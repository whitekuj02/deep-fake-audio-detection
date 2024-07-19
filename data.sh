#!/bin/bash

# Update package lists
apt-get update
apt update

# Install unzip
apt install -y unzip

# Install Python packages
pip install gdown

# Download file from Google Drive
gdown 1hi1dibkHyFbaxAteLlZJw6r3g9ddd4Lf -O /root/

# Unzip the downloaded file
unzip /root/open.zip -d /root/data
rm /root/open.zip

# Install additional Python packages
pip install deepfilternet librosa soundfile pandas pydub torchcontrib tensorboard

mkdir -p /root/asset/