import random
import numpy as np
import torchaudio
import soundfile as sf
import torch
from torch import Tensor
from torch.utils.data import Dataset

___author__ = "Hemlata Tak, Jee-weon Jung"
__email__ = "tak@eurecom.fr, jeeweon.jung@navercorp.com"


def genSpoof_list(dir_meta, is_train=False, is_eval=False, is_unlabel=False):

    d_meta = {}
    file_list = []

    with open(dir_meta, "r") as f:
        f.readline()
        l_meta = f.readlines()

    # train
    if is_train:
        for line in l_meta:
            _, key, label_real, label_fake = line.strip().split(",")
            file_list.append(key)
            d_meta[key] = [int(label_real), int(label_fake)]
        return d_meta, file_list

    # test
    elif is_eval:
        for line in l_meta:
            _, key = line.strip().split(",")
            #key = line.strip()
            file_list.append(key)
        return file_list

    # unlabel
    elif is_unlabel:
        for line in l_meta:
            _, key = line.strip().split(",")
            #key = line.strip()
            file_list.append(key)
        return file_list

    # val
    else:
        for line in l_meta:
            _, key, label_real, label_fake = line.strip().split(",")
            file_list.append(key)
            d_meta[key] = [int(label_real), int(label_fake)]
        return d_meta, file_list


def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


def pad_random(x: np.ndarray, max_len: int = 64600):
    x_len = x.shape[0]
    # if duration is already long enough
    if x_len > max_len:
        stt = np.random.randint(x_len - max_len)
        return x[stt:stt + max_len]

    # if too short
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (num_repeats))[:max_len]
    return padded_x

def add_white_noise(x, noise_level=0.005):
    """Add white noise to the signal."""
    noise = np.random.randn(len(x))
    augmented = x + noise_level * noise
    return augmented

def change_volume(x, volume_level=1.5):
    """Change the volume of the signal."""
    return x * volume_level

def time_stretch(x, rate=1.25):
    """Time stretch the signal."""
    augmented = torchaudio.transforms.Resample(orig_freq=16000, new_freq=int(16000 * rate))(Tensor(x)).numpy()
    return augmented

def spec_augment(x, num_mask=2, freq_masking_max_percentage=0.15, time_masking_max_percentage=0.3):
    """Apply SpecAugment to the signal."""
    x = torch.Tensor(x)
    for i in range(num_mask):
        all_frames_num, all_freqs_num = x.shape

        freq_percentage = random.uniform(0.0, freq_masking_max_percentage)

        num_freqs_to_mask = int(freq_percentage * all_freqs_num)
        f0 = random.randint(0, all_freqs_num - num_freqs_to_mask)

        x[:, f0:f0 + num_freqs_to_mask] = 0

        time_percentage = random.uniform(0.0, time_masking_max_percentage)

        num_frames_to_mask = int(time_percentage * all_frames_num)
        t0 = random.randint(0, all_frames_num - num_frames_to_mask)

        x[t0:t0 + num_frames_to_mask, :] = 0
    
    return x.numpy()

def perturbation(x, sigma=0.003):
    """Apply a small perturbation to the signal."""
    noise = np.random.normal(0, sigma, x.shape)
    return x + noise

class Dataset_Dacon2024_train(Dataset):
    def __init__(self, list_IDs, labels, base_dir):
        """self.list_IDs	: list of strings (each string: utt key),
           self.labels      : dictionary (key: utt key, value: label integer)"""
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.cut = 80000 # take ~4 sec audio (64600 samples)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        X, _ = sf.read(key)
        X_pad = pad_random(X, self.cut)
        x_inp = Tensor(X_pad)
        y = Tensor(self.labels[key])
        return x_inp, y


class Dataset_Dacon2024_devNeval(Dataset):
    def __init__(self, list_IDs, base_dir):
        """self.list_IDs	: list of strings (each string: utt key),
        """
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.cut = 80000  # take ~4 sec audio (64600 samples)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        X, _ = sf.read(key)
        X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)
        return x_inp, key
