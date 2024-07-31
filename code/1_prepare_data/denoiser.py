import os
import argparse
import warnings
import torchaudio
import soundfile as sf

from tqdm import tqdm
from torchaudio.functional import resample
from df.enhance import enhance, init_df

warnings.filterwarnings("ignore")

# 기본 모델 로드
model, df_state, _ = init_df()
model_sr = df_state.sr()

def enhancer(path, output_path):
    # 음성 데이터 불러오기
    audio, sr = torchaudio.load(path)
    audio = resample(audio, orig_freq=sr, new_freq=model_sr)

    # 음성 데이터 노이즈 제거
    audio = enhance(model, df_state, audio)

    # 다시 지정된 샘플링 레이트로 리샘플링
    audio = resample(audio, orig_freq=model_sr, new_freq=args.sr)
    audio = audio.squeeze().numpy()

    # 노이즈가 제거된 오디오 파일을 저장
    sf.write(output_path, audio, sr, format="WAV")

def denoiser(args, mode="train"):
    # 출력 CSV 파일 생성
    f = open(os.path.join(args.base_data_path, f"data16k_denoised/{mode}.csv"), "w")

    # 모드에 따라 CSV 파일 헤더 작성
    if mode == "train":
        f.write("id,path,real,fake\n")
    elif mode == "test":
        f.write("id,path\n")

    # 입력 CSV 파일을 읽어와서 각 라인을 처리
    with open(os.path.join(args.base_data_path, f"{mode}.csv"), "r") as csv:
        csv.readline()  # 첫 번째 라인은 헤더이므로 건너뜀

        for line in tqdm(csv.readlines(), desc=f"Denoise({mode})"):
            audio_id = line.strip().split(",")[0]
            audio_path = os.path.join(args.base_data_path, mode, f"{audio_id}.ogg")
            output_path = os.path.join(
                args.base_data_path, f"data16k_denoised/{mode}/{audio_id}.wav"
            )
            enhancer(audio_path, output_path)  # 노이즈 제거 함수 호출

            # 모드에 따라 결과를 CSV 파일에 작성
            if mode == "train":
                label = line.strip().split(",")[2]
                real = 1 if label == "real" else 0
                fake = 1 if label == "fake" else 0
                f.write(f"{audio_id},{output_path},{real},{fake}\n")
            elif mode == "test":
                f.write(f"{audio_id},{output_path}\n")

    f.close()

if __name__ == "__main__":
    # 명령줄 인자를 파싱
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_data_path", type=str, default="/root/data")
    parser.add_argument("--sr", type=int, default=16000)
    args = parser.parse_args()

    # 각 모드에 대해 노이즈 제거 함수 호출
    denoiser(args, "train")
    denoiser(args, "test")
