import os
import argparse
import warnings
import torchaudio
import soundfile as sf

from tqdm import tqdm
from torchaudio.functional import resample
from df.enhance import enhance, init_df

warnings.filterwarnings("ignore")


# Load default model
model, df_state, _ = init_df()
model_sr = df_state.sr()


def enhancer(path, output_path):
    # 음성 데이터 불러오기
    audio, sr = torchaudio.load(path)
    audio = resample(audio, orig_freq=sr, new_freq=model_sr)

    # Denoise the audio
    audio = enhance(model, df_state, audio)

    audio = resample(audio, orig_freq=model_sr, new_freq=args.sr)
    audio = audio.squeeze().numpy()

    sf.write(output_path, audio, sr, format="WAV")


def denoiser(args, mode="train"):
    f = open(os.path.join(args.base_data_path, f"data16k_denoised/{mode}.csv"), "w")

    if mode == "train":
        f.write("id,path,real,fake\n")
    elif mode == "test":
        f.write("id,path\n")

    with open(os.path.join(args.base_data_path, f"{mode}.csv"), "r") as csv:
        csv.readline()

        for line in tqdm(csv.readlines(), desc=f"Denoise({mode})"):
            audio_id = line.strip().split(",")[0]
            audio_path = os.path.join(args.base_data_path, mode, f"{audio_id}.ogg")
            output_path = os.path.join(
                args.base_data_path, f"data16k_denoised/{mode}/{audio_id}.wav"
            )
            enhancer(audio_path, output_path)

            if mode == "train":
                label = line.strip().split(",")[2]
                real = 1 if label == "real" else 0
                fake = 1 if label == "fake" else 0
                f.write(f"{audio_id},{output_path},{real},{fake}\n")

            elif mode == "test":
                f.write(f"{audio_id},{output_path}\n")

    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_data_path", type=str, default="/root/data")
    parser.add_argument("--sr", type=int, default=16000)
    args = parser.parse_args()

    denoiser(args, "train")
    denoiser(args, "test")
