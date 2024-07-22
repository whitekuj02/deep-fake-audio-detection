# Install additional Python packages
set -e

pip install deepfilternet librosa soundfile pandas pydub torchcontrib tensorboard torchaudio tqdm transformers

python resample.py --base_data_path "/root/data" --sr 16000

python denoiser.py --base_data_path "/root/data" --sr 16000

python audiomix.py --base_data_path "/root/data" --mix_samples 50000 --source "data16k_denoised"

python audiomix.py --base_data_path "/root/data" --mix_samples 50000 --source "data16k"

python train_valid_test.py --base_data_path "/root/data" --train_ratio 0.8

python rawboost.py --base_data_path "/root/data" --train_ratio 0.8 

python nospeech.py --base_data_path "/root/data" --output_path "/root/data/nospeech.csv" --batch_size 32