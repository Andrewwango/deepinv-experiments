from natsort import natsorted
from glob import glob
from torch import load

from .define import *

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