import torch
from torch.utils.data import DataLoader, Dataset
import torchaudio
from transformers import AutoFeatureExtractor, ASTForAudioClassification
from glob import glob
from collections import defaultdict
from tqdm import tqdm
import pandas as pd

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
def inference_batch(model, dataloader, k=5, with_logit=False):
    d = defaultdict(int)
    counts = 0
    tqdm_bar = tqdm(dataloader)

    with open("/root/asset/test_only_speech_list_k5.txt", "w") as tf:
        for batch, paths in tqdm_bar:
            batch = batch.to('cuda:0')
            with torch.no_grad():
                outputs = model(input_values=batch).logits

            for i, logits in enumerate(outputs):
                logits = logits.squeeze()
                predicted_class_ids = torch.argsort(logits)[-k:]
                predicted_labels = [model.config.id2label[_id.item()] for _id in predicted_class_ids]

                for label in predicted_labels:
                    d[label] += 1

                #list_of_lists = [str(tensor.tolist()) for tensor in sorted(logits)[-k:]]
                sorted_indices = torch.argsort(logits)

                # 0의 정렬된 인덱스에서의 위치를 찾음
                sorted_position = (sorted_indices == 0).nonzero(as_tuple=True)[0].item()

                # 뒤에서 몇 번째인지 계산
                reverse_position = len(logits) - sorted_position - 1
                zero_logit_value = logits[0].item()

                if reverse_position > 2 and zero_logit_value <= -2.5:
                    tf.write(paths[i] + " --> " + str(reverse_position) + " --> " + str(zero_logit_value) + "\n")
                    counts += 1
                    

            tqdm_bar.set_postfix(only_speech=d)

    return d, counts

# Load model and feature extractor
feature_extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model = model.to('cuda:0')
model.eval()

# Parameters
batch_size = 32  # Adjust batch size according to your GPU memory
file_paths = glob("/root/data/test/*.ogg")
dataset = AudioDataset(file_paths, feature_extractor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)


# Run inference
d, counts = inference_batch(model, dataloader, k=5, with_logit=True)
print("Finished processing. Total non-speech files:", counts)

file_ids = []

with open("/root/asset/test_only_speech_list_k5.txt", "r") as file:
    for line in file:
        file_path = line.split(' ')[0]
        file_id = file_path.split('/')[-1].split('.')[0]
        file_ids.append(file_id.strip("\n"))

# Create a DataFrame
df = pd.DataFrame(file_ids, columns=['id'])

# Save to CSV
csv_path = "/root/asset/infer_masking.csv"
df.to_csv(csv_path, index=False)

print(f"CSV file saved to {csv_path}")
