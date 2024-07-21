import os
import shutil
import random
import argparse

BASE_DATA_PATH = "/root/data"

def aasist_train(args):
    metadata = []

    for mode in ["train", "mix"]:
        with open(os.path.join(args.base_data_path, f"data16k_denoised/{mode}.csv"), "r") as f:
            f.readline()
            for line in f.readlines():
                id, path, real, fake = line.strip().split(",")
                metadata.append((id, path, real, fake))

    random.shuffle(metadata)
    train_index = int(len(metadata) * args.train_ratio)

    trn_dataset = metadata[:train_index]
    val_dataset = metadata[train_index:]

    for mode, dataset in zip(["train", "val"], [trn_dataset, val_dataset]):
        with open(os.path.join(args.base_data_path, f"aasist/denoise/{mode}.csv"), "w") as f:
            f.write("id,path,real,fake\n")
            for id, path, real, fake in dataset:
                f.write(f"{id},{path},{real},{fake}\n")

def aasist_test(args):
    shutil.copyfile(
        os.path.join(args.base_data_path, "data16k_denoised/test.csv"),
        os.path.join(args.base_data_path, "aasist/denoise/test.csv")
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_data_path", type=str, default="/root/data")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    args = parser.parse_args()

    aasist_train(args)
    aasist_test(args)