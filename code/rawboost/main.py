import os
import glob
import random
import shutil
import librosa
import argparse
import soundfile as sf
import pandas as pd

from tqdm import tqdm
from pydub import AudioSegment
from RawBoost import process_Rawboost_feature

BASE_DATA_PATH = "/root/data/"
NUM_MIX_SAMPLE = 50000
DATA_TRN_RATIO = 0.8

def change_sr_wav(audio_path, target_path, target_sr=16000):
    """
    Convert the sample rate of an audio file to 16000 Hz and save as a WAV file.

    Parameters:
    audio_path (str): Path to the input audio file.
    target_path (str): Path to save the converted audio file.
    target_sr (int): Target sample rate. Default is 16000.
    """

    y, sr = librosa.load(audio_path, sr=target_sr)
    assert sr == target_sr
    sf.write(target_path, y, target_sr, format="WAV")

def overlay_random(audio1_path, audio2_path):
    """
    Overlay two audio files with random volume adjustment and starting position.

    Parameters:
    audio1_path (str): Path to the first audio file.
    audio2_path (str): Path to the second audio file.

    Returns:
    AudioSegment: Combined audio segment.
    """
    audio1 = AudioSegment.from_file(audio1_path)
    audio2 = AudioSegment.from_file(audio2_path)

    len_audio1 = len(audio1)
    len_audio2 = len(audio2)

    start_point = random.randint(0, max(len_audio1, len_audio2) - min(len_audio1, len_audio2))

    # Randomly reduce the volume of one audio
    audio1 = audio1 - random.uniform(0, 30)
    
    # Overlay the two audios
    if len_audio1 >= len_audio2:
        combined = audio1.overlay(audio2, position=start_point)
    else:
        combined = audio2.overlay(audio1, position=start_point)

    return combined

def change_sample_rate(mode="train"):
    """
    Change the sample rate of all audio files in the dataset to 16000 Hz.

    Parameters:
    mode (str): The dataset mode, either 'train' or 'test'.
    """
    with open(os.path.join(BASE_DATA_PATH, f"{mode}.csv"), "r") as f, \
         open(os.path.join(BASE_DATA_PATH, f"data16k/{mode}.csv"), "w") as nf:

        if mode == "train":
            nf.write("id,path,real,fake\n")
        elif mode == "test":
            nf.write("id,path\n")
        elif mode == "unlabeled_data":
            nf.write("id,path\n")

        f.readline()
        for line in tqdm(f.readlines(), desc=f"{mode}"):
            splitted = line.strip().split(",")
            _id = splitted[0]
            tr_audio_path = os.path.join(BASE_DATA_PATH, f"{mode}/{_id}.ogg")
            target_audio_path = os.path.join(BASE_DATA_PATH, f"data16k/{mode}/{_id}.wav")
            change_sr_wav(tr_audio_path, target_audio_path, target_sr=16000)

            if mode == "train":
                real = 1 if splitted[-1] == "real" else 0
                fake = 1 if splitted[-1] == "fake" else 0
                nf.write(f"{_id},{target_audio_path},{real},{fake}\n")
            elif mode == "test":
                nf.write(f"{_id},{target_audio_path}\n")
            elif mode == "unlabeled_data":
                nf.write(f"{_id},{target_audio_path}\n")

def mix_train_dataset():
    """
    Mix audio files from the training dataset to create additional samples.
    """
    data, combined = [], []

    # Read training audio names and labels
    with open(os.path.join(BASE_DATA_PATH, "train.csv"), "r") as f:
        f.readline()
        for line in f.readlines():
            audio_name, _, label = line.strip().split(",")
            data.append((audio_name, label))
        data = data[1:]

    # Perform audio mixing
    for _ in tqdm(range(NUM_MIX_SAMPLE), desc="Audio mix process"):
        idx1, idx2 = 0, 0
        while idx1 == idx2:
            idx1 = random.randint(0, len(data) - 1)
            idx2 = random.randint(0, len(data) - 1)

        audio1_name, audio1_label = data[idx1]
        audio2_name, audio2_label = data[idx2]

        audio1_path = os.path.join(BASE_DATA_PATH, f"data16k/train/{audio1_name}.wav")
        audio2_path = os.path.join(BASE_DATA_PATH, f"data16k/train/{audio2_name}.wav")

        combined_name = f"{audio1_name}-{audio2_name}"
        combined_path = os.path.join(BASE_DATA_PATH, f"data16k/mix/{combined_name}.wav")

        combined_real = 1 if audio1_label == "real" or audio2_label == "real" else 0
        combined_fake = 1 if audio1_label == "fake" or audio2_label == "fake" else 0

        combined_audio = overlay_random(audio1_path, audio2_path)
        combined_audio.export(combined_path, format="wav")
        combined.append((combined_name, combined_path, combined_real, combined_fake))

    # Save the combined dataset information
    train_mixed_df = pd.DataFrame(combined, columns=["audio_name", "audio_path", "real", "fake"])
    train_mixed_df.to_csv(os.path.join(BASE_DATA_PATH, "data16k/mix.csv"), index=False)

def train_test_val(args):
    """
    Process the training, validation, and test datasets for the RawBoost system.

    Parameters:
    args: Command-line arguments for RawBoost processing.
    """
    data_labels = []

    # Read original training data
    with open(os.path.join(BASE_DATA_PATH, "data16k/train.csv"), "r") as trf:
        trf.readline()
        for line in trf.readlines():
            audio_name, path, real, fake = line.strip().split(",")
            data_labels.append((audio_name, path, real, fake))

    # Read mixed training data
    with open(os.path.join(BASE_DATA_PATH, "data16k/mix.csv"), "r") as mxf:
        mxf.readline()
        for line in mxf.readlines():
            audio_name, audio_path, real, fake = line.strip().split(",")
            data_labels.append((audio_name, audio_path, real, fake))

    random.shuffle(data_labels)

    # Rawboost processing
    options = range(9) # [0, 8]
    batch = len(data_labels) // len(options)

    with open(os.path.join(BASE_DATA_PATH, "rawboost/train_all.csv"), "w") as trf:
        trf.write("id,path,real,fake\n")
        for i in options:
            start = batch * i

            if i == len(options) - 1:
                current_batch = data_labels[start:]
            else:
                current_batch = data_labels[start:start+batch]

            for audio_name, audio_path, real, fake in tqdm(current_batch, desc=f"RawBoost mode={i}"):
                y, sr = librosa.load(audio_path, sr=None)
                y = process_Rawboost_feature(y, sr, args, i)

                output_path = os.path.join(BASE_DATA_PATH, f"rawboost/train_all/{audio_name}.wav")
                sf.write(output_path, y, sr, format="WAV")
                trf.write(f"{audio_name},{output_path},{real},{fake}\n")

    # Split into train and validation sets
    rawboost_trn_data = []
    with open(os.path.join(BASE_DATA_PATH, "rawboost/train_all.csv"), "r") as trf:
        trf.readline()
        for line in trf.readlines():
            _id, _path, _real, _fake = line.strip().split(",")
            rawboost_trn_data.append((_id, _path, _real, _fake))
    random.shuffle(rawboost_trn_data)

    trn_sz = int(len(rawboost_trn_data) * DATA_TRN_RATIO)
    trn_data = rawboost_trn_data[:trn_sz]
    with open(os.path.join(BASE_DATA_PATH, "rawboost/train.csv"), "w") as trf:
        trf.write("id,path,real,fake\n")
        for _id, _path, _real, _fake in trn_data:
            new_path = os.path.join(BASE_DATA_PATH, f"rawboost/train/{_id}.wav")
            shutil.copyfile(_path, new_path)
            trf.write(f"{_id},{new_path},{_real},{_fake}\n")

    vld_data = rawboost_trn_data[trn_sz:]
    with open(os.path.join(BASE_DATA_PATH, "rawboost/val.csv"), "w") as vdf:
        vdf.write("id,path,real,fake\n")
        for _id, _path, _real, _fake in vld_data:
            new_path = os.path.join(BASE_DATA_PATH, f"rawboost/val/{_id}.wav")
            shutil.copyfile(_path, new_path)
            vdf.write(f"{_id},{new_path},{_real},{_fake}\n")

    # Process unlabeled_data data
    with open(os.path.join(BASE_DATA_PATH, "data16k/unlabeled_data.csv"), "r") as tsf, \
         open(os.path.join(BASE_DATA_PATH, "rawboost/unlabeled_data.csv"), "w") as rtsf:

        rtsf.write("id,path\n")
        tsf.readline()

        for line in tsf.readlines():
            _id, _path = line.strip().split(",")
            new_path = os.path.join(BASE_DATA_PATH, f"rawboost/unlabeled_data/{_id}.wav")
            shutil.copyfile(_path, new_path)
            rtsf.write(f"{_id},{new_path}\n")

    # Process test data
    with open(os.path.join(BASE_DATA_PATH, "data16k/test.csv"), "r") as tsf, \
         open(os.path.join(BASE_DATA_PATH, "rawboost/test.csv"), "w") as rtsf:

        rtsf.write("id,path\n")
        tsf.readline()

        for line in tsf.readlines():
            _id, _path = line.strip().split(",")
            new_path = os.path.join(BASE_DATA_PATH, f"rawboost/test/{_id}.wav")
            shutil.copyfile(_path, new_path)
            rtsf.write(f"{_id},{new_path}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RawBoost system')

    # LnL_convolutive_noise parameters 
    parser.add_argument('--nBands', type=int, default=5, 
                        help='Number of notch filters. The higher the number of bands, the more aggressive the distortion. [default=5]')
    parser.add_argument('--minF', type=int, default=20, 
                        help='Minimum center frequency [Hz] of notch filter. [default=20]')
    parser.add_argument('--maxF', type=int, default=8000, 
                        help='Maximum center frequency [Hz] (<sr/2) of notch filter. [default=8000]')
    parser.add_argument('--minBW', type=int, default=100, 
                        help='Minimum width [Hz] of filter. [default=100]')
    parser.add_argument('--maxBW', type=int, default=1000, 
                        help='Maximum width [Hz] of filter. [default=1000]')
    parser.add_argument('--minCoeff', type=int, default=10, 
                        help='Minimum filter coefficients. More coefficients result in a more ideal filter slope. [default=10]')
    parser.add_argument('--maxCoeff', type=int, default=100, 
                        help='Maximum filter coefficients. More coefficients result in a more ideal filter slope. [default=100]')
    parser.add_argument('--minG', type=int, default=0, 
                        help='Minimum gain factor of linear component. [default=0]')
    parser.add_argument('--maxG', type=int, default=0, 
                        help='Maximum gain factor of linear component. [default=0]')
    parser.add_argument('--minBiasLinNonLin', type=int, default=5, 
                        help='Minimum gain difference between linear and non-linear components. [default=5]')
    parser.add_argument('--maxBiasLinNonLin', type=int, default=20, 
                        help='Maximum gain difference between linear and non-linear components. [default=20]')
    parser.add_argument('--N_f', type=int, default=5, 
                        help='Order of the (non-)linearity where N_f=1 refers only to linear components. [default=5]')

    # ISD_additive_noise parameters
    parser.add_argument('--P', type=int, default=10, 
                        help='Maximum number of uniformly distributed samples in [%]. [default=10]')
    parser.add_argument('--g_sd', type=int, default=2, 
                        help='Gain parameter > 0. [default=2]')

    # SSI_additive_noise parameters
    parser.add_argument('--SNRmin', type=int, default=10, 
                        help='Minimum SNR value for colored additive noise. [default=10]')
    parser.add_argument('--SNRmax', type=int, default=40, 
                        help='Maximum SNR value for colored additive noise. [default=40]')

    args = parser.parse_args()

    # Create necessary directories
    os.makedirs(os.path.join(BASE_DATA_PATH, "data16k/train"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DATA_PATH, "data16k/test"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DATA_PATH, "data16k/mix"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DATA_PATH, "data16k/unlabeled_data"), exist_ok=True)

    # Step 1: Resample the audio files
    print("1. Resampling..")
    change_sample_rate(mode="train")
    change_sample_rate(mode="test")
    change_sample_rate(mode="unlabeled_data")

    # Step 2: Mix the training dataset
    print("2. Mixing train dataset..")
    mix_train_dataset()

    # Step 3: Apply RawBoost processing
    print("3. Make train / test / val dataset")
    os.makedirs(os.path.join(BASE_DATA_PATH, "rawboost/train_all"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DATA_PATH, "rawboost/train"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DATA_PATH, "rawboost/val"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DATA_PATH, "rawboost/test"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DATA_PATH, "rawboost/unlabeled_data"), exist_ok=True)
    train_test_val(args)
