# Install additional Python packages
pip install deepfilternet librosa soundfile pandas pydub torchcontrib tensorboard torchaudio tqdm

python resample.py --base_data_path "/root/data" --sr 16000

python denoiser.py --base_data_path "/root/data" --sr 16000

python audiomix.py --base_data_path "/root/data" --mix_samples 50000

python train_valid_test.py --base_data_path "/root/data" --train_ratio 0.8
