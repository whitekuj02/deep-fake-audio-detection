import os
import random
import argparse

from tqdm import tqdm
from pydub import AudioSegment


def overlay_random(audio1_path, audio2_path):

    # 음성 파일 로드
    audio1 = AudioSegment.from_file(audio1_path)
    audio2 = AudioSegment.from_file(audio2_path)

    # 두 음성 파일의 길이 계산
    len_audio1 = len(audio1)
    len_audio2 = len(audio2)

    # 랜덤 시작 지점 설정
    start_point = random.randint(0, max(len_audio1, len_audio2) - min(len_audio1, len_audio2))

    # 한쪽 음성의 볼륨을 랜덤하게 낮춤 (-10 dB에서 -30 dB 사이)
    audio1 = audio1 - random.uniform(0, 30)
    
    # 두 음성을 겹침
    if len_audio1 >= len_audio2:
        combined = audio1.overlay(audio2, position=start_point)
    else:
        combined = audio2.overlay(audio1, position=start_point)

    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_data_path", type=str, default="/root/data")
    parser.add_argument("--mix_samples", type=int, default=50000)
    parser.add_argument("--source", type=str, default="data16k_denoised")
    args = parser.parse_args()

    os.makedirs(os.path.join(args.base_data_path, f"{args.source}/mix"), exist_ok=True)

    data, combined = [], []

    # train audio name들 가져오기
    with open(os.path.join(args.base_data_path, "train.csv"), "r") as f:
        f.readline()
        for line in f.readlines():
            audio_name, _, label = line.strip().split(",")
            data.append((audio_name, label))


    # Audio mix
    for _ in tqdm(range(args.mix_samples), desc="Audio mix"):
        idx1 = random.randint(0, len(data) - 1)
        idx2 = random.randint(0, len(data) - 1)

        audio1_name, audio1_label = data[idx1]
        audio2_name, audio2_label = data[idx2]

        audio1_path = os.path.join(args.base_data_path, f"{args.source}/train/{audio1_name}.wav")
        audio2_path = os.path.join(args.base_data_path, f"{args.source}/train/{audio2_name}.wav")

        combined_name = f"{audio1_name}-{audio2_name}"
        combined_path = os.path.join(args.base_data_path, f"{args.source}/mix/{combined_name}.wav")

        combined_real = 1 if audio1_label == "real" or audio2_label == "real" else 0
        combined_fake = 1 if audio1_label == "fake" or audio2_label == "fake" else 0

        combined_audio = overlay_random(audio1_path, audio2_path)
        combined_audio.export(combined_path, format="wav")
        combined.append((combined_name, combined_path, combined_real, combined_fake))

    # 정답 정보를 저장
    with open(os.path.join(args.base_data_path, f"{args.source}/mix.csv"), "w") as f:
        f.write("id,path,real,fake\n")
        for id, path, real, fake in combined:
            f.write(f"{id},{path},{real},{fake}\n")