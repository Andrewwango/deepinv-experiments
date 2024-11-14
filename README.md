# DeepInverse experiments template
Template for deep learning experiments using the [DeepInverse](https://deepinv.github.io/) library

## Get started

- Clone repository: `git clone https://github.com/Andrewwango/deepinv-experiments.git && cd deepinv-experiments`
- Create a local venv: `python -m venv venv` and activate with `source venv/Scripts/activate` (deactivate with `deactivate`). Alternatively create an environment with `conda`.
- Alternatively create a conda env: `conda create -n myvenv python=3.11` and activate with `conda activate myvenv`
- Install requirements: `pip install -r requirements.txt`
- Login to Weights & Biases `wandb login`

### Installation notes
- `deepinv` may need to be installed from GitHub for the latest version: `pip install git+https://github.com/deepinv/deepinv.git#egg=deepinv`
- `torch` may need to be installed with lower CUDA compatibility: `pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu118`

## Usage

### Run training

This runs a full training, see `scripts/train.py` for full parameters, logs progress to Weights & Biases (wandb), and saves model weights to a directory in `--model_dir`. Replace data, physics, model and losses in `scripts/train.py` for your own problem. Log into Weights & Biases to see the results under project name `deepinv-experiments`.

`python scripts/train.py --epochs 2`

Alternatively, collect scripts in a bash file to run multiple scripts:

`bash scripts/train.sh`

### Run eval

This runs an evaluation on multiple wandb runs given in `run_eval.cfg.json`, collects test metrics in a `json` file, plots and saves sample reconstructions and outputs into a specified remote directory.

`python scripts/run_eval.py --runs "test" --plot --save_recon --sample 0 1 2`

Alternatively, collect scripts in a bash file to run multiple scripts:

`bash scripts/run_eval.sh`

## View results

You can view results locally using the [vis_results.ipynb](notebooks/vis_results.ipynb) notebook, plot or export to markdown or latex.

## Notes

### Folder structure

- `docs`: Quarto rendered HTML notebooks
- `notebooks`: IPython notebooks for experiments.
- `results`: evaluation files and visualisations from eval script
- `scripts`: scripts for launching training and evaluation
- `utils`: Python functions and classes for your own experiment code
- `wandb`: local Weights & Biases files saved during `wandb` logging

### Render notebooks

- [Install quarto](https://quarto.org/docs/get-started/) and `quarto render`
- View HTML files in `docs/notebooks` or optionally deploy to a static site on GitHub Pages
