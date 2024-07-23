import os
import torch
import argparse
import torchaudio
import pandas as pd

from glob import glob
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import AutoFeatureExtractor, ASTForAudioClassification

# 데이터셋 클래스 정의
class AudioDataset(Dataset):
    def __init__(self, file_paths, feature_extractor, target_sampling_rate=16000):
        self.file_paths = file_paths
        self.feature_extractor = feature_extractor
        self.target_sampling_rate = target_sampling_rate

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        data, sr = torchaudio.load(file_path)
        data = torchaudio.functional.resample(data, orig_freq=sr, new_freq=self.target_sampling_rate)
        data = data.squeeze()
        inputs = self.feature_extractor(data, sampling_rate=self.target_sampling_rate, return_tensors="pt")
        inputs['input_values'] = inputs['input_values'].squeeze(0)  # 배치 차원 제거
        return inputs['input_values'], file_path

# 추론 함수
def inference_batch(model, dataloader):
    non_speeches = []

    for batch, paths in tqdm(dataloader):
        batch = batch.to('cuda:0')
        with torch.no_grad():
            outputs = model(input_values=batch).logits

        for i, logits in enumerate(outputs):
            logits = logits.squeeze()

            # logits을 정렬하여 인덱스를 얻음
            sorted_indices = torch.argsort(logits)

            # 0의 정렬된 인덱스에서의 위치를 찾음
            sorted_position = (sorted_indices == 0).nonzero(as_tuple=True)[0].item()

            # 뒤에서 몇 번째인지 계산
            reverse_position = len(logits) - sorted_position - 1
            zero_logit_value = logits[0].item()

            if reverse_position > 2 and zero_logit_value <= -2.5:
                audio_id = paths[i].split("/")[-1][:-4]
                non_speeches.append(audio_id)
    
    return sorted(non_speeches)

def main(args):
    # 모델 및 특징 추출기 로드
    feature_extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
    model = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
    model = model.to('cuda:0')
    model.eval()

    # 파라미터 설정
    file_paths = glob(os.path.join(args.base_data_path, "test/*.ogg"))
    dataset = AudioDataset(file_paths, feature_extractor)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # 추론 실행
    non_speeches = inference_batch(model, dataloader)

    # 데이터프레임 생성 및 CSV 저장
    pd.DataFrame(non_speeches, columns=['id']).to_csv(args.output_path, index=False)

if __name__ == "__main__":
    # 명령줄 인자 파싱
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--output_path', type=str, default="/root/asset/nospeech.csv")
    parser.add_argument("--base_data_path", type=str, default="/root/data")
    args = parser.parse_args()

    main(args)
