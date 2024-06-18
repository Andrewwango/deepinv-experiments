# DeepInverse experiments template
Template for deep learning experiments using the DeepInverse library

## Get started

- Clone repository: `git clone https://github.com/Andrewwango/deepinv-experiments.git && cd deepinv-experiments`
- Create a local venv: `python -m venv venv` and activate with `source venv/Scripts/activate` (deactivate with `deactivate`)
- Alternatively create a conda env: `conda create -n myvenv python=3.11` and activate with `conda activate myvenv`
- Install requirements: `pip install -r requirements.txt`
- Login to Weights & Biases `wandb login`

### Installation notes
- `deepinv` may need to be installed from GitHub for the latest version: `pip install git+https://github.com/deepinv/deepinv.git#egg=deepinv`
- `torch` may need to be installed with lower CUDA compatibility: `pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu118`

## Usage

### Run training

- `bash scripts/train.sh`

Replace data, physics, model and losses in `scripts/train.py` for your own problem. Log into Weights & Biases to see the results.

### Run eval

TBC

## Notes

### Folder structure

- `docs`: Quarto rendered HTML notebooks
- `models`: torch models saved by deepinv training.
- `notebooks`: IPython notebooks for experiments.
- `results`: evaluation files and visualisations from eval script
- `scripts`: scripts for launching training and evaluation
- `utils`: Python functions and classes for your own experiment code
- `wandb`: local Weights & Biases files saved during `wandb` logging

### Render notebooks

- [Install quarto](https://quarto.org/docs/get-started/) and `quarto render`
- View HTML files in `docs/notebooks` or optionally deploy to a static site on GitHub Pages