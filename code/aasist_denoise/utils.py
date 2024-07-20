"""
Utilization functions
"""

import os
import pytz
import torch
import random
import numpy as np

import datetime


def set_requires_grad(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad
        

def cosine_annealing(step, total_steps, lr_max, lr_min):
    """Cosine Annealing for learning rate decay scheduler"""
    return lr_min + (lr_max -
                     lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))



def _get_scheduler(optimizer, epochs, base_lr, min_lr, tot_steps):
    """
    Defines learning rate scheduler according to the given config
    """

    total_steps = epochs * tot_steps

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_annealing(
            step,
            total_steps,
            1,  # since lr_lambda computes multiplicative factor
            min_lr / base_lr))
    
    return scheduler



def set_seed(seed):
    """ 
    set initial seed for reproduction
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_experiment():
    
    base_dir = "/root/code/DomainAdaptation/experiments"
    os.makedirs(base_dir, exist_ok=True)

    # 현재 날짜와 시간 가져오기
    folder_name = datetime.datetime.now().strftime("exp-%m-%d-%H-%M")
    exp_path = os.path.join(base_dir, folder_name)
    os.makedirs(exp_path, exist_ok=True)
    os.makedirs(os.path.join(exp_path, "weights"), exist_ok=True)

    return exp_path


def loop_iterable(dataloader):
    while True:
        yield from dataloader

def combined_dataloader(loader1, loader2):
    for combined in zip(loader1, loader2):
        return combined


def get_time():
    # 현재 UTC 시간 가져오기
    utc_now = datetime.datetime.now(datetime.timezone.utc)
    
    # KST 시간대로 변환
    kst_timezone = pytz.timezone('Asia/Seoul')
    kst_now = utc_now.astimezone(kst_timezone)
    
    # hh:ss:mm 형식으로 반환
    return kst_now.strftime('%H:%M:%S')


def get_nonspeech_list():
    nonspeech_list = []

    print("newmaksing")
    with open("/root/asset/new_masking.csv", "r") as f:
        f.readline()
        for line in f.readlines():
            nonspeech_list.append(line.strip())

    return nonspeech_list