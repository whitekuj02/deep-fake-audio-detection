import os
import sys
import torch
import argparse
from torch.utils.data import DataLoader

from tqdm import tqdm
from model import Model
from dataset import DaconDataset
from torchcontrib.optim import SWA
from utils import _get_scheduler, set_seed

MAX_PATIENCE = 3
EPOCH = 30
SEED = 990917

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tqdm.write(f"Current device: {device}")
set_seed(SEED)


def one_epoch(model, dataloader, criterion, optimizer=None):
    total_loss = 0

    tqdm_bar = tqdm(dataloader, file=sys.stdout)
    for x, y_true in tqdm_bar:
        x, y_true = x.to(device), y_true.to(device)

        y_pred = model(x)
        loss = criterion(y_pred, y_true)

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        tqdm_bar.set_postfix(loss=loss.item())

    mean_loss = total_loss / len(dataloader)

    return mean_loss


def main(args):

    exp_path = "./experiments"

    # 데이터셋 준비
    train_dataset = DaconDataset(args.data_path, mode="train")
    valid_dataset = DaconDataset(args.data_path, mode="val")

    train_loader = DataLoader(train_dataset, batch_size=32, drop_last=True, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, drop_last=False, shuffle=False, pin_memory=True)

    # 모델, 옵티마이저 준비
    model = Model().to(device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    tqdm.write("* No. model params:{}".format(nb_params))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)
    scheduler = _get_scheduler(optimizer, EPOCH, 0.0001, 0.000005, len(train_loader))
    criterion = torch.nn.BCELoss()
    optimizer_swa = SWA(optimizer)

    # 학습 및 검증 시작
    n_swa_update = 0
    best_loss = 1.0
    patience = 0

    for epoch in range(1, EPOCH):

        # 학습 - train
        model.train()
        tqdm.write(f"\n* EPOCH {epoch:03d} | Training started")
        train_loss = one_epoch(model, train_loader, criterion, optimizer)

        # 검증 - validation
        model.eval()
        tqdm.write(f"* EPOCH {epoch:03d} | Validation started")
        with torch.no_grad():
            val_loss = one_epoch(model, valid_loader, criterion)

        # memo
        tqdm.write(f'* EPOCH {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}')

        # 최적의 파라미터 저장
        if val_loss < best_loss:
            patience = 0
            tqdm.write(f'* EPOCH {epoch:03d} | Best Model founded! Saving model...')
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(exp_path, "params.pt"))

            optimizer_swa.update_swa()
            n_swa_update += 1
        else:
            patience += 1
            if patience == MAX_PATIENCE:
                tqdm.write(f"* EPOCH {epoch:03d} | Max patience reached. Early stopping..")
                break

        # optimizer 및 scheduler 업데이트
        scheduler.step()
        tqdm.write("")

    if n_swa_update > 0:
        optimizer_swa.swap_swa_sgd()
        optimizer_swa.bn_update(train_loader, model, device=device)
    torch.save(model.state_dict(), os.path.join(exp_path, "swa_params.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/root/data/aasist/denoise")
    args = parser.parse_args()
    main(args)
