import os, math
from natsort import natsorted
from glob import glob
from subprocess import call
from torch import load

from .define import *

def avg_and_std(arr: list):
    if len(arr) == 0:
        return (0, 0)
    
    mean = sum(arr) / len(arr)
    
    variance = sum((x - mean) ** 2 for x in arr) / (len(arr) - 1)
    std_dev = math.sqrt(variance)
    
    return (mean, std_dev)

def load_model(model, model_dir: str, wandb_run_id: str, device="cpu", eval=True, optimizer=None):
    ckpt_fn = natsorted(
        glob(f"{model_dir}/{wandb_run_id}/*/ckp_*.pth.tar")
    )[-1]
    checkpoint = load(ckpt_fn, map_location=device)
    model.load_state_dict(checkpoint["state_dict"], strict=False)

    if eval:
        model.eval()    

    if "optimizer" in checkpoint and optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        return model, optimizer

    return model

"""def move_dirs(source_dir: str, target_dir: str):
    if len(glob(f'{source_dir}/*')) > 0:
        os.makedirs(target_dir, exist_ok=True)
        call(f'mv {source_dir}/* {target_dir}/', shell=True)
        print("Moved folders")
    else:
        print(f"Folder {source_dir} empty")"""