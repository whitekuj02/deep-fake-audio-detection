import os
import librosa
import argparse
import soundfile as sf
from tqdm import tqdm


# 데이터를 16000으로 resampling 후, wav 형식으로 저장
def resample_audio(args, mode):

    # new_f = open(os.path.join(args.base_data_path, f"/datak16k/{mode}.csv"))

    with open(os.path.join(args.base_data_path, "{}.csv".format(mode))) as f:
        f.readline()

        for line in tqdm(f.readlines(), desc=f"Resample({mode})"):
            audio_name = line.strip().split(",")[0]
            audio_path = os.path.join(args.base_data_path, mode, f"{audio_name}.ogg")
            output_path = os.path.join(args.base_data_path, "data16k", mode, f"{audio_name}.wav")
            y, _ = librosa.load(audio_path, sr=args.sr)
            sf.write(output_path, y, args.sr, format="WAV")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_data_path", type=str, default="/root/data")
    parser.add_argument("--sr", type=int, default=16000)
    args = parser.parse_args()

    resample_audio(args, "train")
    resample_audio(args, "test")