import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from argparse import ArgumentParser
from munch import DefaultMunch
import wandb

import numpy as np
import torch
import deepinv as dinv

from utils import *

parser = ArgumentParser()
parser.add_argument("-e", "--epochs", type=int, default=10)
parser.add_argument("-b", "--batch_size", type=int, default=1)
parser.add_argument("-l", "--learning_rate", type=float, default=1e-3)
parser.add_argument("--model_dir", type=str, default="models", help="Dir where models saved")
parser.add_argument("--ckpt", type=str, default=None, help="wandb run id for checkpoint to load")

args = parser.parse_args()

config = DefaultMunch(
    epochs=args.epochs,
    batch_size=args.batch_size,
    lr_init=args.learning_rate,
    seed=0,
)

project_name="deepinv-experiments"
os.makedirs(args.model_dir, exist_ok=True)

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

torch.manual_seed(config.seed)
np.random.seed(config.seed)
generator = torch.Generator(device=device).manual_seed(config.seed)

with wandb.init(project=project_name, config=config, dir="./wandb"):
    config = wandb.config

    ## Define experiment
    physics = define_physics(config, device=device, generator=generator)
    train_dataloader, test_dataloader = define_data(config, random_split_seed=config.seed, physics=physics, generator=generator, device=device)
    metrics = define_metrics(config)
    model = define_model(config, device=device)
    loss, model = define_loss(config, model=model, device=device)
    optimizer, scheduler = define_optimizer_scheduler(model, config)

    if args.ckpt is not None:
        model, optimizer = load_model(model, args.model_dir, args.ckpt, eval=False, optimizer=optimizer, device=device)

    # Define trainer
    trainer = dinv.training.Trainer(
        model = model,
        physics = physics,
        optimizer = optimizer,
        train_dataloader = train_dataloader,
        eval_dataloader = test_dataloader,
        epochs = config.epochs,
        losses = loss,
        scheduler = scheduler,
        metrics = metrics,
        online_measurements = ...,
        ckp_interval = 1000,
        device = device,
        eval_interval = 1,
        save_path = f"{args.model_dir}/{wandb.run.id}",
        plot_images = False,
        wandb_vis = True,
    )

    trainer.train()