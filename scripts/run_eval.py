import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from subprocess import call
from glob import glob
from argparse import ArgumentParser
from munch import DefaultMunch
from natsort import natsorted
from tqdm import tqdm

import wandb
import numpy as np
import torch
import deepinv as dinv
from utils import *

parser = ArgumentParser()
parser.add_argument("-s", "--sample", nargs="+", type=int, default=[0], help="Which images to plot from dataloader")
parser.add_argument("-r", "--runs", type=str, default="", help="Name of run set from run_eval.cfg")
parser.add_argument("--model_dir_local", type=str, default="/home/s2558406/Repos/deepinv-experiments/models", help="Dir where models saved")
parser.add_argument("--model_dir_remote", type=str, default="/home/s2558406/RDS/models/deepinv-experiments", help="Dir for results to be saved to")
parser.add_argument("--plot", action="store_true", help="Whether plot image results")
parser.add_argument("--save_recon", action="store_true", help="Whether save reconstructed images as numpy volumes")

args = parser.parse_args()

def avg(arr):
    return sum(arr) / len(arr) if len(arr) > 0 else 0

def move_models_to_datastore():
    if len(glob(f'{args.local_store}/*')) > 0:
        os.makedirs(args.remote_store, exist_ok=True)
        #move_command = f'[ ! -f {local_store}/* ] || mv {local_store}/* {remote_store}/'
        move_command = f'mv {args.local_store}/* {args.remote_store}/'
        print(move_command)
        call(move_command, shell=True)
        print("Moved folders")
    else:
        print(f"Local model folder {args.local_store} empty")
    return args.remote_store

def plot_results(*args, save_fn=None, dpi=(300), **kwargs):
    fig = dinv.utils.plot_inset(*args, show=False, save_fn=None, return_fig=True, **kwargs)
    for d in dpi:
        fig.savefig(f"{save_fn}_{d}", dpi=d)

wandb_runs = wandb.Api().runs("ei-experiments")
with open("scripts/run_eval.cfg.json", "r") as f:
    args_runs = json.load(f)[args.runs]
runs = {
    run.id: DefaultMunch.fromDict(run.config, None) for run in wandb_runs
    if run.id in list(args_runs.keys())
}
runs = {k: runs[k] for k in args_runs.keys()}

print(f"Evaluating {len(runs)} runs")

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

physics = ... #TODO

test_dataloader = torch.utils.data.DataLoader(..., shuffle=False, batch_size=1) #TODO

model = ....to(device) #TODO

metrics = [dinv.loss.PSNR()]

model_dir = move_models_to_datastore()

results = {}

ite = 0
plot_x_hat_images, plot_x_hat_labels, plot_x_hat_results = [], [], []
for id in runs.keys():
    if id not in os.listdir(f"{model_dir}"): continue
    ite += 1
    
    config = runs[id]
    
    base_fn = f"{model_dir}/{id}/*/ckp_"
    ckpt_fn = natsorted(
        glob(f"{model_dir}/{id}/*/ckp_*.pth.tar")
    )[-1]
    checkpoint = torch.load(ckpt_fn, map_location=device)
    
    print(f"{ite}: {id} epochs {config.epochs} ckpt {os.path.basename(ckpt_fn)}")

    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.eval()

    metrics_x_hat, metrics_x_init, plot_x_hat, plot_x, plot_x_init, plot_y = [], [], [], [], [], [], [], []

    for i, (x, y) in tqdm(enumerate(test_dataloader)):
        with torch.no_grad():
            x = x.to(device)
            y = y.to(device)

            x1 = model(y, physics)
            x_init = physics.A_adjoint(y)

            if i in args.sample:
                plot_x_hat.append(x1)
                plot_x_init.append(x_init)
                plot_x.append(x)
                plot_y.append(y)

            metrics_x_init.append([metric(x_init, x) for metric in metrics])
            metrics_x_hat.append([metric(x1, x) for metric in metrics])
    
    plot_x, plot_x_init, plot_y = [torch.cat(imgs) if len(imgs) > 0 else [] for imgs in (plot_x, plot_x_init, plot_y)] #gets replaced on each ite
    plot_x_hat_images += [torch.cat(plot_x_hat)]

    config["title"] = args_runs[id]
    config["metrics"] = avg(list(map(list, zip(*metrics_x_hat)))) #transpose list of lists
    config["metrics_init"] = avg(list(map(list, zip(*metrics_x_init))))
    
    plot_x_hat_labels.append(args_runs[id])
    plot_x_hat_results.append(round(config["metrics"], 2))

    results[id] = config

base_fn = f"{model_dir}/eval_{args.runs}"

with open(f"{base_fn}.json", "w") as f:
    json.dump(results, f)

if args.plot:
    plot_results(
        [
            plot_y,
            plot_x_init,
            plot_x, 
            *plot_x_hat_images
        ], 
        titles=["y (measurements)", "No learning", "x (GT)", *plot_x_hat_labels],
        labels=[None, round(config["metrics_init"], 2), None, *plot_x_hat_results],
        save_fn=base_fn,
    )

if args.save_recon:
    save_dict = {}
    for (x_hat, title) in zip(plot_x_hat_images, plot_x_hat_labels):
        save_dict[title] = x_hat.detach().cpu().numpy()
    save_dict["x"] = plot_x.detach().cpu().numpy()
    save_dict["x_init"] = plot_x_init.detach().cpu().numpy()
    save_dict["y"] = plot_y.detach().cpu().numpy()
    
    np.savez(base_fn, **save_dict)
    
    