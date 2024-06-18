import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from argparse import ArgumentParser
from pathlib import Path
from munch import DefaultMunch
import wandb

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import deepinv as dinv

from utils import *

parser = ArgumentParser()
parser.add_argument("-e", "--epochs", type=int, default=2)
parser.add_argument("-b", "--batch_size", type=int, default=1)
parser.add_argument("-l", "--learning_rate", type=float, default=1e-3)
parser.add_argument("--arg2", action="store_true") #Boolean arg defaulting to False

args = parser.parse_args()

config = DefaultMunch(
    epochs=args.epochs,
    batch_size=args.batch_size,
    lr_init=args.learning_rate,
    seed=0,
)

project_name="deepinv-experiments"

torch.manual_seed(config.seed)
np.random.seed(config.seed)
device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

with wandb.init(project=project_name, config=config, dir="./wandb"):
    config = wandb.config

    dataset = ... #TODO
    
    physics = dinv.physics.MRI((64, 64))

    train_dataset, test_dataset = random_split(dataset, (0.8, 0.2))

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, num_workers=0, pin_memory=True, shuffle=True)
    test_dataloader  = DataLoader(dataset= test_dataset, batch_size=config.batch_size, num_workers=0, pin_memory=True, shuffle=False)

    losses = dinv.loss.SupLoss()

    model = dinv.models.UNet().to(device)

    trainer = dinv.training.Trainer(
        model = model,
        physics = physics,
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr_init),
        train_dataloader = train_dataloader,
        eval_dataloader = test_dataloader,
        epochs = config.epochs,
        losses = losses,
        scheduler = None,
        metrics = dinv.loss.PSNR(),
        online_measurements = False,
        ckp_interval = 1000,
        device = device,
        eval_interval = 1,
        save_path = f"models/{wandb.run.id}",
        plot_images = True,
        wandb_vis = True,
    )

    trainer.train()