import os
import shutil
import librosa
import argparse
import soundfile as sf
from tqdm import tqdm
from glob import glob

def create_unlab_csv(args):

    with open(os.path.join(args.base_data_path, "unlabeled_data.csv"), "w") as f:
        f.write("id,path\n")
        for path in glob(os.path.join(args.base_data_path, "unlabeled_data/*.ogg")):
            id = path.split("/")[-1][:-4]
            f.write(f"{id},{path}\n")

    shutil.copyfile(
        os.path.join(args.base_data_path, "unlabeled_data.csv"),
        os.path.join(args.base_data_path, "aasist/rawboost/unlabeled_data.csv")
    )

    

# 데이터를 16000으로 resampling 후, wav 형식으로 저장
def resample_audio(args, mode):

    if mode == "unlabeled_data":
        create_unlab_csv(args)

    with open(os.path.join(args.base_data_path, "{}.csv".format(mode)), "r") as f, \
        open(os.path.join(args.base_data_path, f"data16k/{mode}.csv"), "w") as wf:

        if mode == "train":
            wf.write("id,path,real,fake\n")
        elif mode == "test" or mode == "unlabeled_data":
            wf.write("id,path\n")

        f.readline()

        for line in tqdm(f.readlines(), desc=f"Resample({mode})"):
            audio_id = line.strip().split(",")[0]
            audio_path = os.path.join(args.base_data_path, mode, f"{audio_id}.ogg")
            output_path = os.path.join(args.base_data_path, "data16k", mode, f"{audio_id}.wav")
            y, _ = librosa.load(audio_path, sr=args.sr)
            sf.write(output_path, y, args.sr, format="WAV")

            if mode == "train":
                label = line.strip().split(",")[2]
                real = 1 if label == "real" else 0
                fake = 1 if label == "fake" else 0
                f.write(f"{audio_id},{output_path},{real},{fake}\n")

            elif mode == "test":
                f.write(f"{audio_id},{output_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_data_path", type=str, default="/root/data")
    parser.add_argument("--sr", type=int, default=16000)
    args = parser.parse_args()

    resample_audio(args, "train")
    resample_audio(args, "test")
    resample_audio(args, "unlabeled_data")