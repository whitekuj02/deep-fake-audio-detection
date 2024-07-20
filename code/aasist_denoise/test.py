
import torch
import argparse
import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm
from model import Model
from dataset import DaconDataset
from utils import get_nonspeech_list
from torch.utils.data import DataLoader

device = torch.device("cuda")


def main(args):
    # 데이터셋 설정
    half_batch = args.batch_size // 2
    dataset = DaconDataset(db_path=args.data_path, mode="test")
    dataloader = DataLoader(dataset, batch_size=half_batch, shuffle=False, pin_memory=True, drop_last=False)

    # 모델 설정
    model = Model().to(device)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    # 최종 마스킹 설정
    nonspeech_list = get_nonspeech_list()

    # 테스트 데이터셋 이름 준비
    audio_ids  = [audio_path.split("/")[-1][:-4] for audio_path in dataset.file_list]
    preds = []

    # 모델 추론
    with torch.no_grad():
        for batch, _ in tqdm(dataloader):
            batch = batch.to(device)
            pred = model(batch)
            preds.extend(pred.cpu().numpy().tolist())

    # 제출용 csv 파일 생성
    with open("./submission.csv", "w") as f:
        f.write("id,fake,real\n")
        for aid, (real, fake) in zip(audio_ids, preds):
            if aid in nonspeech_list:
                fake, real = 0.0, 0.0
            f.write("{},{},{}\n".format(aid, fake, real))

    tqdm.write("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--data_path", type=str, default="/root/data/aasist/denoise")
    parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()
    main(args)