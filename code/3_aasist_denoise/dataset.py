import torch
import numpy as np
import soundfile as sf
from tqdm import tqdm
from torch.utils.data import Dataset

def prepare_dataset_list(db_path, mode):

    csv_path = f"{db_path}/{mode}.csv"
    with open(csv_path, "r") as f:
        f.readline()
        lines = f.readlines()

    file_list, labels = [], {}

    # train
    if mode in ["train", "val"]:
        for line in tqdm(lines, desc="prepare dataset"):
            aid, apath, areal, afake = line.strip().split(",")
            file_list.append(apath)
            labels[apath] = [int(areal), int(afake)]

    elif mode == "test":
        for line in tqdm(lines, desc="prepare dataset"):
            aid, apath = line.strip().split(",")
            file_list.append(apath)
    
    return file_list, labels


def pad(x, max_len=80000):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


def pad_random(x: np.ndarray, max_len: int = 80000):
    x_len = x.shape[0]
    # if duration is already long enough
    if x_len > max_len:
        stt = np.random.randint(x_len - max_len)
        return x[stt:stt + max_len]

    # if too short
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (num_repeats))[:max_len]
    return padded_x


class DaconDataset(Dataset):
    def __init__(self, db_path, mode='train'):
        file_list, labels = prepare_dataset_list(db_path, mode=mode)
        self.file_list = file_list
        self.labels = labels
        self.mode = mode
        self.cut = 80000 # 16000sr * 5초 = 80000

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        audiopath = self.file_list[index]
        X, _ = sf.read(audiopath)

        if self.mode not in ["test", "unlabeled_data"]:
            X = torch.Tensor(pad_random(X, self.cut))
        else:
            X = torch.Tensor(X)[:self.cut]
            
        y = 0
        if len(self.labels) != 0:
            y = torch.Tensor(self.labels[audiopath])

        return X, y