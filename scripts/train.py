import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from argparse import ArgumentParser
from pathlib import Path
from munch import DefaultMunch
import wandb

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, ToTensor, Resize
from torchvision.datasets.utils import download_and_extract_archive
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

    # Define physics
    physics = dinv.physics.Inpainting((3, 256, 256))

    # Download Urban100 dataset
    download_and_extract_archive(
        "https://huggingface.co/datasets/eugenesiow/Urban100/resolve/main/data/Urban100_HR.tar.gz?download=true",
        "Urban100",
        filename="Urban100_HR.tar.gz",
        md5="65d9d84a34b72c6f7ca1e26a12df1e4c",
    )

    train_dataset, test_dataset = random_split(
        ImageFolder(
            "Urban100", transform=Compose([ToTensor(), Resize(256)])
        ),
        (0.8, 0.2),
    )

    # Prepare dataset of images and measurements
    dataset_path = dinv.datasets.generate_dataset(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        physics=physics,
        device=device,
        save_dir="Urban100",
    )

    train_dataloader = DataLoader(
        dinv.datasets.HDF5Dataset(dataset_path, train=True), shuffle=True, batch_size=config.batch_size,
    )
    test_dataloader = DataLoader(
        dinv.datasets.HDF5Dataset(dataset_path, train=False), shuffle=False, batch_size=config.batch_size,
    )
    
    # Define loss
    losses = dinv.loss.SupLoss()

    # Define model
    model = dinv.models.UNet().to(device)

    # Define trainer
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