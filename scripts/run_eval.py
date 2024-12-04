## Script to evaluate multiple runs, plot images, calculate metrics and save reconstructions.

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from argparse import ArgumentParser
from munch import DefaultMunch
from tqdm import tqdm

import wandb
import numpy as np
import torch
import deepinv as dinv
from utils import *

parser = ArgumentParser()
parser.add_argument("-s", "--sample", nargs="+", type=int, default=[0], help="Which images to plot from dataloader")
parser.add_argument("-r", "--runs", type=str, default="", help="Name of run set from run_eval.cfg")
parser.add_argument("--model_dir", type=str, default="models", help="Dir where models saved")
parser.add_argument("--plot", action="store_true", help="Whether plot image results")
parser.add_argument("--save_recon", action="store_true", help="Whether save reconstructed images as numpy volumes")
parser.add_argument("--skip_metrics", action="store_true")
args = parser.parse_args()

## Load and choose runs
wandb_runs = wandb.Api().runs("deepinv-experiments")
with open("scripts/run_eval.cfg.json", "r") as f:
    args_runs = json.load(f)[args.runs]
runs = {
    run.id: DefaultMunch.fromDict(run.config, None) for run in wandb_runs
    if run.id in list(args_runs.keys())
}
runs = {k: runs[k] for k in args_runs.keys()}

print(f"Evaluating {len(runs)} runs")

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

## Iterate through runs
results = {}
ite = 0
plot_x_hat_images, plot_x_hat_labels, plot_x_hat_results = [], [], []
for id in runs.keys():
    if id not in os.listdir(f"{args.model_dir}"): continue
    ite += 1    
    config = runs[id]
    print(f"{ite}: {id} trained for epochs {config.epochs}")    

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    generator = torch.Generator().manual_seed(config.seed)

    ## Define experiment
    physics = define_physics(config, device=device, generator=generator)
    _, test_dataloader = define_data(config, random_split_seed=config.seed, batch_size=1, physics=physics, generator=generator, device=device)
    metrics = define_metrics(config)
    model = define_model(config, device=device)
    _, model = define_loss(config, model=model, device=device)
    model = load_model(model, args.model_dir, id, device=device)

    ## Iterate through dataset
    metrics_x_hat, metrics_x_init = [dinv.utils.AverageMeter(m.__class__.__name__) for m in metrics], [dinv.utils.AverageMeter(m.__class__.__name__) for m in metrics]
    plot_x_hat, plot_x, plot_x_init, plot_y = [], [], [], []
    for i, (x, y) in tqdm(enumerate(test_dataloader)):
        with torch.no_grad():
            if i in args.sample or not args.skip_metrics:
                x = x.to(device)
                y = y.to(device)
                #physics.update_parameters(mask=mask.to(device))

                x1 = model(y, physics)
                x_init = physics.A_adjoint(y)

            if i in args.sample:
                plot_x_hat.append(x1)
                plot_x_init.append(x_init)
                plot_x.append(x)
                plot_y.append(y)

            for m, meter_x_hat, meter_x_init in zip(metrics, metrics_x_hat, metrics_x_init):
                meter_x_hat .update(m(x1,     x).detach().cpu().numpy())
                meter_x_init.update(m(x_init, x).detach().cpu().numpy())
    
    ## Log results
    plot_x, plot_x_init, plot_y = [torch.cat(imgs) if len(imgs) > 0 else [] for imgs in (plot_x, plot_x_init, plot_y)] #gets replaced on each ite
    plot_x_hat_images += [torch.cat(plot_x_hat)]

    config["title"] = args_runs[id]
    if not args.skip_metrics:
        config["metrics"]      = [(meter.avg, meter.std) for meter in metrics_x_hat]
        config["metrics_init"] = [(meter.avg, meter.std) for meter in metrics_x_init]
    
    plot_x_hat_labels.append(args_runs[id])
    plot_x_hat_results.append(round(config["metrics"][0][0], 2)) # NOTE this picks the first metric by default
    plot_x_init_result = round(config["metrics_init"][0][0], 2) # gets replaced on every ite

    results[id] = config

## Save metrics
base_fn = f"{args.model_dir}/eval_{args.runs}"

if not args.skip_metrics:
    with open(f"{base_fn}.json", "w") as f:
        json.dump(results, f)

## Save plots
if args.plot:
    dinv.utils.plot_inset(
        [
            plot_y,
            plot_x_init,
            plot_x, 
            *plot_x_hat_images
        ], 
        titles=["y (measurements)", "No learning", "x (GT)", *plot_x_hat_labels],
        labels=[None, plot_x_init_result, None, *plot_x_hat_results],
        save_fn=base_fn,
        show=False,
        dpi=300,
    )

## Save reconstructions
if args.save_recon:
    save_dict = {}
    for (x_hat, title, result) in zip(plot_x_hat_images, plot_x_hat_labels, plot_x_hat_results):
        save_dict[title] = x_hat.detach().cpu().numpy()
        save_dict[title + "_result"] = str(result)
    
    np.savez(
        base_fn,
        x=plot_x.detach().cpu().numpy(),
        y=plot_y.detach().cpu().numpy(),
        x_init=plot_x_init.detach().cpu().numpy(),
        x_result="",
        y_result="",
        x_init_result=str(plot_x_init_result),
        **save_dict
    )
    
    
