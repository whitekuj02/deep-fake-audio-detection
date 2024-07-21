#!/bin/bash
set -e

base_data_path="/root/data"

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
unzip -q /root/open.zip -d "${base_data_path}"
rm /root/open.zip


# Prepare data directories
mkdir -p "${base_data_path}/data16k_denoised/train" \
         "${base_data_path}/data16k_denoised/test" \
         "${base_data_path}/data16k_denoised/val" \
         "${base_data_path}/data16k_rawboost/train" \
         "${base_data_path}/data16k/train" \
         "${base_data_path}/data16k/test" \
         "${base_data_path}/data16k/mix" \
         "${base_data_path}/aasist/denoise" \
         "${base_data_path}/aasist/rawboost"