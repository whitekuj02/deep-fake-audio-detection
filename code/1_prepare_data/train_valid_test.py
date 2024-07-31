import os
import shutil
import random
import argparse

# 훈련 및 검증 데이터를 준비하는 함수
def aasist_train(args):
    metadata = []

    # train 및 mix 모드에 대해 데이터 로드
    for mode in ["train", "mix"]:
        with open(os.path.join(args.base_data_path, f"data16k_denoised/{mode}.csv"), "r") as f:
            f.readline()  # 헤더 건너뛰기
            for line in f.readlines():
                id, path, real, fake = line.strip().split(",")
                metadata.append((id, path, real, fake))

    # 데이터 섞기
    random.shuffle(metadata)
    train_index = int(len(metadata) * args.train_ratio)

    # 훈련 및 검증 데이터셋 분리
    trn_dataset = metadata[:train_index]
    val_dataset = metadata[train_index:]

    # 분리된 데이터셋을 train 및 val 파일로 저장
    for mode, dataset in zip(["train", "val"], [trn_dataset, val_dataset]):
        with open(os.path.join(args.base_data_path, f"aasist/denoise/{mode}.csv"), "w") as f:
            f.write("id,path,real,fake\n")
            for id, path, real, fake in dataset:
                f.write(f"{id},{path},{real},{fake}\n")

# 테스트 데이터를 준비하는 함수
def aasist_test(args):
    shutil.copyfile(
        os.path.join(args.base_data_path, "data16k_denoised/test.csv"),
        os.path.join(args.base_data_path, "aasist/denoise/test.csv")
    )

# 메인 함수
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_data_path", type=str, default="/root/data")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    args = parser.parse_args()

    # 훈련 및 테스트 데이터 준비 함수 호출
    aasist_train(args)
    aasist_test(args)
