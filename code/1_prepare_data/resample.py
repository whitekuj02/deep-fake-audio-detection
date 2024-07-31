import os
import shutil
import librosa
import argparse
import soundfile as sf
from tqdm import tqdm
from glob import glob

# Unlabeled 데이터의 CSV 파일을 생성하는 함수
def create_unlab_csv(args):
    # unlabeled_data.csv 파일 생성
    with open(os.path.join(args.base_data_path, "unlabeled_data.csv"), "w") as f:
        f.write("id,path\n")
        for path in glob(os.path.join(args.base_data_path, "unlabeled_data/*.ogg")):
            id = path.split("/")[-1][:-4]
            f.write(f"{id},{path}\n")

    # 생성된 CSV 파일을 aasist/rawboost 디렉토리로 복사
    shutil.copyfile(
        os.path.join(args.base_data_path, "unlabeled_data.csv"),
        os.path.join(args.base_data_path, "aasist/rawboost/unlabeled_data.csv")
    )

# 데이터를 16000Hz로 리샘플링 후, wav 형식으로 저장하는 함수
def resample_audio(args, mode):

    # 모드가 "unlabeled_data"일 경우, CSV 파일 생성
    if mode == "unlabeled_data":
        create_unlab_csv(args)

    # 입력 CSV 파일과 출력 CSV 파일을 연다
    with open(os.path.join(args.base_data_path, "{}.csv".format(mode)), "r") as f, \
        open(os.path.join(args.base_data_path, f"data16k/{mode}.csv"), "w") as wf:

        # 모드에 따라 CSV 파일 헤더 작성
        if mode == "train":
            wf.write("id,path,real,fake\n")
        elif mode == "test" or mode == "unlabeled_data":
            wf.write("id,path\n")

        # 첫 번째 라인은 헤더이므로 건너뜀
        f.readline()

        # 각 라인을 읽어와서 오디오 파일을 리샘플링하고 저장
        for line in tqdm(f.readlines(), desc=f"Resample({mode})"):
            audio_id = line.strip().split(",")[0]
            audio_path = os.path.join(args.base_data_path, mode, f"{audio_id}.ogg")
            output_path = os.path.join(args.base_data_path, "data16k", mode, f"{audio_id}.wav")
            y, _ = librosa.load(audio_path, sr=args.sr)
            sf.write(output_path, y, args.sr, format="WAV")

            # 모드가 "train"일 경우, 라벨을 추가하여 CSV 파일에 작성
            if mode == "train":
                label = line.strip().split(",")[2]
                real = 1 if label == "real" else 0
                fake = 1 if label == "fake" else 0
                wf.write(f"{audio_id},{output_path},{real},{fake}\n")

            # 모드가 "test"일 경우, 라벨 없이 CSV 파일에 작성
            elif mode == "test":
                wf.write(f"{audio_id},{output_path}\n")

if __name__ == "__main__":
    # 명령줄 인자를 파싱
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_data_path", type=str, default="/root/data")
    parser.add_argument("--sr", type=int, default=16000)
    args = parser.parse_args()

    # 각 모드에 대해 리샘플링 함수 호출
    resample_audio(args, "train")
    resample_audio(args, "test")
    resample_audio(args, "unlabeled_data")
