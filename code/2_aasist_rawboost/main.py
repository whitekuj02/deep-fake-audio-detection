"""
Main script that trains, validates, and evaluates
various models including AASIST.

AASIST
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
import os
import sys
import json
import warnings
import argparse
import itertools

from tqdm import tqdm
from shutil import copy
from pathlib import Path
from dacon_eval import dacon_score
from importlib import import_module
from typing import Dict, List, Union
from utils import create_optimizer, seed_worker, set_seed, str_to_bool, create_exp
from data_utils import Dataset_Dacon2024_train, Dataset_Dacon2024_devNeval, genSpoof_list

import torch
import torch.nn as nn
from torchcontrib.optim import SWA
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast


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
    alpha = 0

    if "eval_all_best" not in config:
        config["eval_all_best"] = "True"
    if "freq_aug" not in config:
        config["freq_aug"] = "False"

    # make experiment reproducible
    set_seed(args.seed, config)

    # define database related paths
    database_path = Path(config["database_path"])
    dev_trial_path = database_path / "val.csv"
    eval_trial_path = database_path / "test.csv"

    # define model related paths
    model_tag = create_exp("experiments")
    model_tag = Path(model_tag)
    model_save_path = model_tag / "weights"
    eval_score_path = model_tag / "test"
    metric_path = model_tag / "metrics"
    
    writer = SummaryWriter(model_tag)

    os.makedirs(eval_score_path, exist_ok=True)
    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(metric_path, exist_ok=True)
    
    copy(args.config, model_tag / "config.conf")

    # set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tqdm.write("Device: {}".format(device))
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
        tqdm.write("Model loaded : {}".format(config["model_path"]))
        tqdm.write("Start evaluation...")
        produce_evaluation_file(eval_loader, model, device, eval_score_path/f"submission.csv", eval_trial_path, alpha)
        tqdm.write("DONE.")
        sys.exit(0)

    # get optimizer and scheduler
    optim_config["steps_per_epoch"] = len(trn_loader)
    optimizer, scheduler = create_optimizer(model.parameters(), optim_config)
    optimizer_swa = SWA(optimizer)

    initial_alpha = 0.2  # 초기 alpha 값
    max_alpha = 0.8  # 최종 alpha 값
    n_swa_update = 0  # number of snapshots of model to use in SWA

    # Training
    for epoch in range(config["num_epochs"]):
        tqdm.write("Start training epoch{:03d}".format(epoch))
        alpha = initial_alpha + (max_alpha - initial_alpha) * (epoch / config["num_epochs"])
        running_loss = train_epoch(trn_loader, unlabel_loader, model, optimizer, device,
                                   scheduler, config, alpha)
        produce_evaluation_file(dev_loader, model, device,
                                metric_path/"val_pred.csv", dev_trial_path, alpha)
        val_score = dacon_score(
                        answer_path=database_path/"val.csv",
                        submission_path=metric_path/"val_pred.csv"
                    )

        tqdm.write("DONE.\nLoss:{:.5f}, val_score: {:.3f}".format(running_loss, val_score))
        writer.add_scalar("loss", running_loss, epoch)
        writer.add_scalar("val_score", val_score, epoch)

        torch.save(model.state_dict(), model_save_path / "params.pth")

        # do evaluation whenever best model is renewed
        if str_to_bool(config["eval_all_best"]):
            produce_evaluation_file(eval_loader, model, device,
                                    eval_score_path/"pre_submission.csv", eval_trial_path, alpha)
            mask_zero(eval_score_path/"pre_submission.csv", epoch, eval_score_path)

        tqdm.write("Saving epoch {} for swa".format(epoch))
        optimizer_swa.update_swa()
        n_swa_update += 1

    tqdm.write("Start final evaluation")
    if n_swa_update > 0:
        optimizer_swa.swap_swa_sgd()
        combined_loader = itertools.chain(trn_loader, unlabel_loader)
        optimizer_swa.bn_update(combined_loader, model, device=device)
    
    produce_evaluation_file(eval_loader, model, device, eval_score_path/"pre_submission.csv", eval_trial_path, alpha)
    mask_zero(eval_score_path/"pre_submission.csv", epoch, eval_score_path)
    torch.save(model.state_dict(), model_save_path / "swa_params.pth")

    tqdm.write("Training / inference done.")


def mask_zero(submission_path, ep, eval_score_path):
    

    non_speeches = None
    with open(os.path.join("/root/data/nospeech.csv")) as f:
        f.readline()
        non_speeches = [non_speech.strip() for non_speech in f.readlines()]

    if ep > 5 and ep <= 10:
        with open(os.path.join(submission_path), "r") as f, \
             open(os.path.join(eval_score_path, "submission.csv".format(ep)), "w") as wf:
            
            wf.write(f.readline())
            for line in f.readlines():
                _id, _fake, _real = line.strip().split(",")
                if _id in non_speeches:
                    _fake, _real = 0., 0.
                wf.write("{},{},{}\n".format(_id, _fake, _real))
            wf.close()

    tqdm.write("Masking (post-processing) done!")



def get_model(model_config: Dict, device: torch.device):
    """Define DNN model architecture"""
    module = import_module("models.{}".format(model_config["architecture"]))
    _model = getattr(module, "Model")
    model = _model(model_config).to(device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    tqdm.write("no. model params:{}".format(nb_params))

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
    tqdm.write(f"no. training files: {len(file_train)}")
    

    train_set = Dataset_Dacon2024_train(list_IDs=file_train,
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
    tqdm.write(f"no. validation files: {len(file_dev)}")

    dev_set = Dataset_Dacon2024_devNeval(list_IDs=file_dev,
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

    tqdm.write(f"no. unlabel files: {len(file_unlabel)}")

    unlabel_set = Dataset_Dacon2024_devNeval(list_IDs=file_unlabel,
                                             base_dir=domain_database_path)
    unlabel_loader = DataLoader(unlabel_set,
                             batch_size=config["batch_size"],
                             shuffle=True,
                             drop_last=False,
                             pin_memory=True)


    # test
    file_eval = genSpoof_list(dir_meta=eval_trial_path,
                              is_train=False,
                              is_eval=True)
    
    tqdm.write(f"no. test files: {len(file_eval)}")
    eval_set = Dataset_Dacon2024_devNeval(list_IDs=file_eval,
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
    trial_path: str,
    alpha) -> None:
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
            _, batch_out, _ = model(batch_x, alpha)
            batch_out = nn.Sigmoid()(batch_out)
        
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
    tqdm.write("Scores saved to {}".format(save_path))

        
def train_epoch(
    trn_loader: DataLoader,
    domain_loader: DataLoader,
    model,
    optim: Union[torch.optim.SGD, torch.optim.Adam],
    device: torch.device,
    scheduler: torch.optim.lr_scheduler,
    config: argparse.Namespace,
    alpha):
    """Train the model for one epoch"""
    running_loss = 0
    num_total = 0.0
    ii = 0
    count = 1
    model.train()

    # set objective (Loss) functions
    criterion = nn.BCEWithLogitsLoss()
    domain_criterion = nn.BCEWithLogitsLoss()

    # amp scaler
    scaler = GradScaler()

    train_iter = iter(trn_loader)
    domain_iter = itertools.cycle(domain_loader)

    # Training with trn_loader (includes x and y)
    tqdm_bar = tqdm(zip(train_iter, domain_iter), total=len(trn_loader), desc="mode=[TRAIN]")
    for batch, domain_batch in tqdm_bar:
        batch_x = batch[0].to(device)
        batch_y = batch[1].to(device)
        domain_x = domain_batch[0].to(device)
        domain_labels = torch.zeros(batch_x.size(0), 1).to(device)
        _domain_labels = torch.ones(domain_x.size(0), 1).to(device)

        batch_size = batch_x.size(0)
        num_total += batch_size
        ii += 1

        with autocast():
            _, batch_out, domain_out = model(batch_x, alpha, Freq_aug=str_to_bool(config["freq_aug"]))
            _, _, _domain_out = model(domain_x, alpha, Freq_aug=str_to_bool(config["freq_aug"]))

            batch_loss = criterion(batch_out, batch_y)
            domain_loss = domain_criterion(domain_out, domain_labels)
            _domain_loss = domain_criterion(_domain_out, _domain_labels)


            total_loss = batch_loss + domain_loss + _domain_loss
            running_loss += total_loss.item() * batch_size
            tqdm_bar.set_postfix(alpha=alpha, batch_loss=batch_loss.item(), domain_0_loss=domain_loss.item(), domain_1_loss = _domain_loss.item(), total_loss=total_loss.item())

            optim.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.step(optim)
            scaler.update()


        if config["optim_config"]["scheduler"] in ["cosine", "keras_decay", "sgdr"]:
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
    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="random seed (default: 1234)")
    parser.add_argument(
        "--eval",
        action="store_true",
        help="when this flag is given, evaluates given model and exit")
    parser.add_argument("--eval_model_weights",
                        type=str,
                        default=None,
                        help="directory to the model weight file (can be also given in the config file)")
    main(parser.parse_args())
