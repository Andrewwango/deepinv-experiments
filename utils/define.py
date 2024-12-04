from typing import List, Union, Tuple
from torch import Generator
from torch.nn import Module
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, ToTensor, Resize, CenterCrop

import deepinv as dinv
from deepinv.physics import Physics, LinearPhysics
from deepinv.loss import Loss, Metric

## Edit these definitions for your own experiment

def define_model(config: dict, device="cpu") -> Module:
    return dinv.models.DnCNN(depth=2, nf=8, pretrained=None, device=device).to(device)

def define_loss(config: dict, model: Module = None, device="cpu") -> Union[Loss, List[Loss]]:
    return [dinv.loss.SupLoss()], model

def define_metrics(config: dict, reduction=None) -> Union[Metric, List[Metric]]:
    return [dinv.loss.PSNR(reduction=reduction), dinv.loss.SSIM(reduction=reduction)]

def define_data(config: dict, random_split_seed: int, data_dir: str = ".", batch_size: int = None, physics: Physics = None, generator: Generator = None, device="cpu") -> Tuple[DataLoader]:
    batch_size = batch_size if batch_size is not None else config.batch_size

    random_split_generator = Generator(device="cpu").manual_seed(random_split_seed) # separate generator for random_split to a) stay on cpu and b) prevent generator conditional fork

    train_dataset, test_dataset = random_split(
        dinv.datasets.Urban100HR(data_dir, download=True, transform=Compose([
            ToTensor(),
            CenterCrop(128),
            Resize(64, antialias=True),
        ])),
        (0.8, 0.2),
        generator=random_split_generator
    )

    dataset_path = dinv.datasets.generate_dataset(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        physics=physics,
        device=device,
        save_dir=f"{data_dir}/Urban100",
    )

    train_dataloader = DataLoader(
        dinv.datasets.HDF5Dataset(dataset_path, train=True), shuffle=True, batch_size=batch_size, generator=generator
    )
    test_dataloader = DataLoader(
        dinv.datasets.HDF5Dataset(dataset_path, train=False), shuffle=False, batch_size=batch_size,
    )
    return train_dataloader, test_dataloader

def define_physics(config: dict, device="cpu", generator: Generator = None) -> Union[LinearPhysics, Physics]:
    return dinv.physics.Inpainting((3, 64, 64), mask=0.6)

def define_optimizer_scheduler(model: Module, config: dict) -> Tuple[Optimizer, LRScheduler]:
    return Adam(model.parameters(), lr=config.lr_init), None