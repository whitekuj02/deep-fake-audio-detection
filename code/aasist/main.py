"""
Main script that trains, validates, and evaluates
various models including AASIST.

AASIST
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
import argparse
import json
import os
import sys
import warnings
from importlib import import_module
from pathlib import Path
from shutil import copy
from typing import Dict, List, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from torchcontrib.optim import SWA
import torch.optim as optim

from data_utils import (Dataset_ASVspoof2019_train,
                        Dataset_ASVspoof2019_devNeval, genSpoof_list)
from evaluation import calculate_tDCF_EER
from utils import create_optimizer, seed_worker, set_seed, str_to_bool
from tqdm import tqdm
from dacon_eval import dacon_score

warnings.filterwarnings("ignore", category=FutureWarning)


def main(args: argparse.Namespace) -> None:
    """
    Main function.
    Trains, validates, and evaluates the ASVspoof detection model.
    """
    # load experiment configurations
    with open(args.config, "r") as f_json:
        config = json.loads(f_json.read())
    model_config = config["model_config"]
    optim_config = config["optim_config"]
    optim_config["epochs"] = config["num_epochs"]
    track = config["track"]
    assert track in ["LA", "PA", "DF"], "Invalid track given"
    if "eval_all_best" not in config:
        config["eval_all_best"] = "True"
    if "freq_aug" not in config:
        config["freq_aug"] = "False"

    # make experiment reproducible
    set_seed(args.seed, config)

    # define database related paths
    output_dir = Path(args.output_dir)
    database_path = Path(config["database_path"])
    dev_trial_path = database_path / "val.csv"
    eval_trial_path = database_path / "test.csv"

    # define model related paths
    model_tag = "{}_{}_ep{}_bs{}".format(
        track,
        os.path.splitext(os.path.basename(args.config))[0],
        config["num_epochs"], config["batch_size"])
    if args.comment:
        model_tag = model_tag + "_{}".format(args.comment)
    model_tag = output_dir / model_tag
    model_save_path = model_tag / "weights"
    eval_score_path = model_tag / "test"
    os.makedirs(eval_score_path, exist_ok=True)

    writer = SummaryWriter(model_tag)
    os.makedirs(model_save_path, exist_ok=True)
    
    copy(args.config, model_tag / "config.conf")

    # set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: {}".format(device))
    if device == "cpu":
        raise ValueError("GPU not detected!")

    # define model architecture
    model = get_model(model_config, device)

    # define dataloaders
    trn_loader, dev_loader, eval_loader, unlabel_loader = get_loader(
        database_path, args.seed, config)

    # evaluates pretrained model and exit script
    if args.eval:
        model.load_state_dict(
            torch.load(config["model_path"], map_location=device))
        print("Model loaded : {}".format(config["model_path"]))
        print("Start evaluation...")
        produce_evaluation_file(eval_loader, model, device,
                                eval_score_path, eval_trial_path)
        calculate_tDCF_EER(cm_scores_file=eval_score_path,
                           asv_score_file=database_path /
                           config["asv_score_path"],
                           output_file=model_tag / "t-DCF_EER.txt")
        print("DONE.")
        eval_eer, eval_tdcf = calculate_tDCF_EER(
            cm_scores_file=eval_score_path,
            asv_score_file=database_path / config["asv_score_path"],
            output_file=model_tag/"loaded_model_t-DCF_EER.txt")
        sys.exit(0)

    # get optimizer and scheduler
    optim_config["steps_per_epoch"] = len(trn_loader)
    optimizer, scheduler = create_optimizer(model.parameters(), optim_config)
    optimizer_swa = SWA(optimizer)

    best_dev_eer = 1.
    best_val_score = 100.
    best_eval_tdcf = 1.
    n_swa_update = 0  # number of snapshots of model to use in SWA
    f_log = open(model_tag / "metric_log.txt", "a")
    f_log.write("=" * 5 + "\n")

    # make directory for metric logging
    metric_path = model_tag / "metrics"
    os.makedirs(metric_path, exist_ok=True)

    # Training
    for epoch in range(config["num_epochs"]):
        print("Start training epoch{:03d}".format(epoch))
        running_loss = train_epoch(trn_loader, unlabel_loader, model, optimizer, device,
                                   scheduler, config)
        produce_evaluation_file(dev_loader, model, device,
                                metric_path/"val_pred.csv", dev_trial_path)
        val_score = dacon_score(
                        answer_path=database_path/"val.csv",
                        submission_path=metric_path/"val_pred.csv"
                    )

        print("DONE.\nLoss:{:.5f}, val_score: {:.3f}".format(running_loss, val_score))
        writer.add_scalar("loss", running_loss, epoch)
        writer.add_scalar("val_score", val_score, epoch)

        # torch.save(model.state_dict(),
        #             model_save_path / "epoch_{}_{:03.3f}.pth".format(epoch, val_score))

        # do evaluation whenever best model is renewed
        if str_to_bool(config["eval_all_best"]):
            produce_evaluation_file(eval_loader, model, device,
                                    eval_score_path/f"submission_ep{epoch}.csv", eval_trial_path)
            mask_zero(eval_score_path/f"submission_ep{epoch}.csv", epoch, val_score)

        print("Saving epoch {} for swa".format(epoch))
        optimizer_swa.update_swa()
        n_swa_update += 1

    print("Start final evaluation")
    epoch += 1
    if n_swa_update > 0:
        optimizer_swa.swap_swa_sgd()
        combined_loader = DataLoader(
            ConcatDataset([trn_loader.dataset, unlabel_loader.dataset]),
            batch_size=trn_loader.batch_size,
            shuffle=False
        )
        optim.bn_update(combined_loader, model, device=device)
    
    produce_evaluation_file(eval_loader, model, device, eval_score_path/"submission_last.csv", eval_trial_path)
    torch.save(model.state_dict(),
               model_save_path / "swa.pth")

    print("Training / inference done.")


ASSET_PATH = "/root/asset"

def mask_zero(submission_path, ep, val_score):

    non_speeches = None
    with open(os.path.join(ASSET_PATH, "nonspeech.csv")) as f:
        f.readline()
        non_speeches = [non_speech.strip() for non_speech in f.readlines()]

    with open(os.path.join(submission_path), "r") as f, \
         open(os.path.join(ASSET_PATH, "masked_submission-ep{}-val{}.csv".format(ep,val_score)), "w") as wf:
        
        wf.write(f.readline())
        for line in f.readlines():
            _id, _fake, _real = line.strip().split(",")
            if _id in non_speeches:
                _fake, _real = 0., 0.
            wf.write("{},{},{}\n".format(_id, _fake, _real))
        wf.close()

    print("Masking (post-processing) done!")



def get_model(model_config: Dict, device: torch.device):
    """Define DNN model architecture"""
    module = import_module("models.{}".format(model_config["architecture"]))
    _model = getattr(module, "Model")
    model = _model(model_config).to(device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print("no. model params:{}".format(nb_params))

    return model


def get_loader(
        database_path: str,
        seed: int,
        config: dict) -> List[torch.utils.data.DataLoader]:
    """Make PyTorch DataLoaders for train / developement / evaluation"""

    trn_database_path = database_path / "train"
    dev_database_path = database_path / "val"
    domain_database_path = database_path / "unlabeled_data"
    eval_database_path = database_path / "test"

    trn_list_path = database_path / "train.csv"
    dev_trial_path = database_path / "val.csv"
    domain_list_path = database_path / "unlabeled_data.csv"
    eval_trial_path = database_path / "test.csv"

    # train
    d_label_trn, file_train = genSpoof_list(dir_meta=trn_list_path,
                                            is_train=True,
                                            is_eval=False)
    print("no. training files:", len(file_train))

    train_set = Dataset_ASVspoof2019_train(list_IDs=file_train,
                                           labels=d_label_trn,
                                           base_dir=trn_database_path)
    gen = torch.Generator()
    gen.manual_seed(seed)
    trn_loader = DataLoader(train_set,
                            batch_size=config["batch_size"],
                            shuffle=True,
                            drop_last=True,
                            pin_memory=True,
                            worker_init_fn=seed_worker,
                            generator=gen)

    # val
    _, file_dev = genSpoof_list(dir_meta=dev_trial_path,
                                is_train=False,
                                is_eval=False)
    print("no. validation files:", len(file_dev))

    dev_set = Dataset_ASVspoof2019_devNeval(list_IDs=file_dev,
                                            base_dir=dev_database_path)
    dev_loader = DataLoader(dev_set,
                            batch_size=config["batch_size"],
                            shuffle=False,
                            drop_last=False,
                            pin_memory=True)

    # unlabel
    file_unlabel = genSpoof_list(dir_meta=domain_list_path,
                              is_train=False,
                              is_eval=False,
                              is_unlabel=True)

    print("no. unlabel files:", len(file_unlabel))

    unlabel_set = Dataset_ASVspoof2019_devNeval(list_IDs=file_unlabel,
                                             base_dir=eval_database_path)
    unlabel_loader = DataLoader(unlabel_set,
                             batch_size=config["batch_size"],
                             shuffle=True,
                             drop_last=False,
                             pin_memory=True)


    # test
    file_eval = genSpoof_list(dir_meta=eval_trial_path,
                              is_train=False,
                              is_eval=True)
    
    print("no. test files:", len(file_eval))
    eval_set = Dataset_ASVspoof2019_devNeval(list_IDs=file_eval,
                                             base_dir=eval_database_path)
    eval_loader = DataLoader(eval_set,
                             batch_size=config["batch_size"],
                             shuffle=False,
                             drop_last=False,
                             pin_memory=True)

    return trn_loader, dev_loader, eval_loader, unlabel_loader


def produce_evaluation_file(
    data_loader: DataLoader,
    model,
    device: torch.device,
    save_path: str,
    trial_path: str) -> None:
    """Perform evaluation and save the score to a file"""
    model.eval()
    with open(trial_path, "r") as f_trl:
        f_trl.readline()
        trial_lines = f_trl.readlines()
    fname_list = []
    score_list = []
    for batch_x, utt_id in tqdm(data_loader, desc="mode=[VAL/TEST]"):

        batch_x = batch_x.to(device)
        with torch.no_grad():
            _, batch_out, _ = model(batch_x)
            # print(batch_out)
            # batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()
        
        # add outputs
        fname_list.extend(utt_id)
        score_list.extend(batch_out.tolist())

    assert len(trial_lines) == len(fname_list) == len(score_list)
    with open(save_path, "w") as fh:
        fh.write("id,fake,real\n")
        for (real, fake), trl in zip(score_list, trial_lines):
            metadata = trl.strip().split(",")
            audio_name = metadata[0]
            fh.write("{},{},{}\n".format(audio_name, fake, real))
    print("Scores saved to {}".format(save_path))


# def train_epoch(
#     trn_loader: DataLoader,
#     model,
#     optim: Union[torch.optim.SGD, torch.optim.Adam],
#     device: torch.device,
#     scheduler: torch.optim.lr_scheduler,
#     config: argparse.Namespace):
#     """Train the model for one epoch"""
#     running_loss = 0
#     num_total = 0.0
#     ii = 0
#     model.train()

#     # set objective (Loss) functions
#     weight = torch.FloatTensor([0.1, 0.9]).to(device)
#     criterion = nn.BCELoss()
#     tqdm_bar = tqdm(trn_loader, desc="mode=[TRAIN]")
#     for batch_x, batch_y in tqdm_bar:
#         batch_size = batch_x.size(0)
#         num_total += batch_size
#         ii += 1
#         batch_x = batch_x.to(device)
#         batch_y = batch_y.to(device)

#         _, batch_out = model(batch_x, Freq_aug=str_to_bool(config["freq_aug"]))


#         batch_loss = criterion(batch_out, batch_y)
#         tqdm_bar.set_postfix(batch_loss=batch_loss.item())
#         running_loss += batch_loss.item() * batch_size
#         optim.zero_grad()
#         batch_loss.backward()
#         optim.step()

#         if config["optim_config"]["scheduler"] in ["cosine", "keras_decay"]:
#             scheduler.step()
#         elif scheduler is None:
#             pass
#         else:
#             raise ValueError("scheduler error, got:{}".format(scheduler))

#     running_loss /= num_total
#     return running_loss

def train_epoch(
    trn_loader: DataLoader,
    domain_loader: DataLoader,
    model,
    optim: Union[torch.optim.SGD, torch.optim.Adam],
    device: torch.device,
    scheduler: torch.optim.lr_scheduler,
    config: argparse.Namespace):
    """Train the model for one epoch"""
    running_loss = 0
    num_total = 0.0
    ii = 0
    model.train()

    # set objective (Loss) functions
    criterion = nn.BCELoss()
    domain_criterion = nn.BCELoss()

    # Training with trn_loader (includes x and y)
    tqdm_bar = tqdm(trn_loader, desc="mode=[TRAIN]")
    for batch in tqdm_bar:
        batch_x = batch[0].to(device)
        batch_y = batch[1].to(device)
        domain_labels = torch.zeros(batch_x.size(0), 2).to(device)
        domain_labels[:, 0] = 1

        batch_size = batch_x.size(0)
        num_total += batch_size
        ii += 1

        _, batch_out, domain_out = model(batch_x, Freq_aug=str_to_bool(config["freq_aug"]))

        batch_loss = criterion(batch_out, batch_y)
        domain_loss = domain_criterion(domain_out, domain_labels)
        total_loss = batch_loss + domain_loss
        running_loss += total_loss.item() * batch_size
        tqdm_bar.set_postfix(batch_loss=batch_loss.item(), domain_loss=domain_loss.item(), total_loss=total_loss.item())

        optim.zero_grad()
        total_loss.backward()
        optim.step()

        if config["optim_config"]["scheduler"] in ["cosine", "keras_decay"]:
            scheduler.step()
        elif scheduler is None:
            pass
        else:
            raise ValueError("scheduler error, got:{}".format(scheduler))

    # Training with domain_loader (includes only x)
    for i in range(40):
        tqdm_bar = tqdm(domain_loader, desc="mode=[DOMAIN_TRAIN]")
        for batch in tqdm_bar:
            batch_x = batch[0].to(device)
            domain_labels = torch.zeros(batch_x.size(0), 2).to(device)
            domain_labels[:, 1] = 1

            batch_size = batch_x.size(0)
            num_total += batch_size
            ii += 1

            # 모델 예측
            _, _, domain_out = model(batch_x, Freq_aug=str_to_bool(config["freq_aug"]))

            # 도메인 분류 손실 계산
            domain_loss = domain_criterion(domain_out, domain_labels)
            total_loss = domain_loss

            running_loss += total_loss.item() * batch_size
            tqdm_bar.set_postfix(repeat=i, domain_loss=domain_loss.item(), total_loss=total_loss.item())

            optim.zero_grad()
            total_loss.backward()
            optim.step()

            if config["optim_config"]["scheduler"] in ["cosine", "keras_decay"]:
                scheduler.step()
            elif scheduler is None:
                pass
            else:
                raise ValueError("scheduler error, got:{}".format(scheduler))

    running_loss /= num_total
    return running_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASVspoof detection system")
    parser.add_argument("--config",
                        dest="config",
                        type=str,
                        help="configuration file",
                        required=True)
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        type=str,
        help="output directory for results",
        default="./exp_result_rawboost",
    )
    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="random seed (default: 1234)")
    parser.add_argument(
        "--eval",
        action="store_true",
        help="when this flag is given, evaluates given model and exit")
    parser.add_argument("--comment",
                        type=str,
                        default=None,
                        help="comment to describe the saved model")
    parser.add_argument("--eval_model_weights",
                        type=str,
                        default=None,
                        help="directory to the model weight file (can be also given in the config file)")
    main(parser.parse_args())
