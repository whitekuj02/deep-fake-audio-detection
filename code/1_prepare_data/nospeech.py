import os
import torch
import argparse
import torchaudio
import pandas as pd

from glob import glob
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import AutoFeatureExtractor, ASTForAudioClassification


# Define dataset class
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
        inputs['input_values'] = inputs['input_values'].squeeze(0)  # Remove batch dimension
        return inputs['input_values'], file_path


# Inference function
def inference_batch(model, dataloader, k=5):
    non_speeches = []

    with open("/root/asset/test_only_speech_list_k5.txt", "w") as tf:
        for batch, paths in tqdm(dataloader):
            batch = batch.to('cuda:0')
            with torch.no_grad():
                outputs = model(input_values=batch).logits

            for i, logits in enumerate(outputs):
                logits = logits.squeeze()

                #list_of_lists = [str(tensor.tolist()) for tensor in sorted(logits)[-k:]]
                sorted_indices = torch.argsort(logits)

                # 0의 정렬된 인덱스에서의 위치를 찾음
                sorted_position = (sorted_indices == 0).nonzero(as_tuple=True)[0].item()

                # 뒤에서 몇 번째인지 계산
                reverse_position = len(logits) - sorted_position - 1
                zero_logit_value = logits[0].item()

                if reverse_position > 2 and zero_logit_value <= -2.5:
                    audio_id = paths[i].split("/")[-1][:-4]
                    non_speeches.append(audio_id)
                    
    return non_speeches


def main(args):
    # Load model and feature extractor
    feature_extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
    model = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
    model = model.to('cuda:0')
    model.eval()

    # Parameters
    file_paths = glob(os.path.join(args.base_data_path, "test/*.ogg"))
    dataset = AudioDataset(file_paths, feature_extractor)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Run inference
    non_speeches = inference_batch(model, dataloader, k=5, with_logit=True)

    # Create a DataFrame
    pd.DataFrame(non_speeches, columns=['id']).to_csv(args.output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--output_path', type=str, default="/root/asset/nospeech.csv")
    parser.add_argument("--base_data_path", type=str, default="/root/data")
    args = parser.parse_args()

    main(args)
