# ICED: In-Context Density Estimation for Tabular Data

This repository is built on the first version of TabPFN and [ZEUS](https://github.com/gmum/zeus) codebase. The corresponding license can be found in the [legal](legal) directory. For the latest version, see the TabPFN2 repository: https://github.com/PriorLabs/TabPFN

## Abstract
Code repository for [https://arxiv.org/abs/2608.09348](https://arxiv.org/abs/2608.09348)

Density estimation underlies many unsupervised tasks on tabular data such as anomaly detection, 
out-of-distribution detection, and data augmentation. Although all these problems reduce to questions 
about where probability mass lies, they are typically solved individually by fitting a separate model to each dataset, 
with its own hyperparameters and tuning budget. We introduce ICED, an in-context, energy-based density estimator 
that removes this per-dataset cost. ICED is a transformer-based model pretrained once on a synthetic prior 
built specifically for density estimation under an objective that fits log-density where it is informative 
and preserves its ordering elsewhere. In the inference, it reads a dataset as context and returns an unnormalized 
log-density for any query point in a single forward pass, with no fitting, sampling, or hyperparameter selection. 
A single frozen ICED model then drives four tasks usually handled by four specialized pipelines: density estimation, 
out-of-distribution detection, unsupervised anomaly detection, and generative augmentation. Across all four, 
it is competitive with the strongest task-specific method, while being the only approach that needs no retraining, 
no tuning, and no labels to move between them.

## Setup
Create a conda environment and install dependencies:
```
conda create -n iced python=3.10
conda activate iced
pip install -r requirements.txt
```

## Pre-training
Pre-training is launched by running `pretrain.py`. Configuration is handled via [OmegaConf](https://omegaconf.readthedocs.io/) CLI overrides on top of the defaults in [src/configs.py](src/configs.py) (e.g. `device`, `dim`, `n_layers`, `nr_epochs`, `output_dir`, `model_path` to resume from a checkpoint):
```
python pretrain.py device=cuda nr_epochs=1000 output_dir=./output
```
Checkpoints, plots, and logs (via Weights & Biases) are written under `output_dir`.

## Model checkpoint
The best ICED checkpoint used in the paper, along with the required TabICLv2 checkpoint, can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1ZS2p6qtxNdrt2IZ186t34RL-y9PQhzum?usp=sharing)

ICED relies on a frozen TabICL encoder ([src/model/icl_encoder.py](src/model/icl_encoder.py)), so the TabICL checkpoint is required to run any pre-training or evaluation, not just to reproduce the paper's results. Place the downloaded files at:
```
checkpoints/iced.pt
checkpoints/tabicl/tabicl-classifier-v2-20260212.ckpt
```
`checkpoints/iced.pt` is the default `model_path` used by `pretrain.py`/`evaluation.py`; the TabICL path is currently hardcoded in [src/model/icl_encoder.py](src/model/icl_encoder.py).

## AdBench datasets
Evaluating anomaly and out-of-distribution detection (`evaluate_anomalies`) requires the [AdBench](https://drive.google.com/drive/folders/1ZS2p6qtxNdrt2IZ186t34RL-y9PQhzum?usp=sharing) real-world datasets. Download them and place the `.npz` files under [src/evaluation/anomalies/datasets/real](src/evaluation/anomalies/datasets/real), split into the five expected subfolders:
```
src/evaluation/anomalies/datasets/real/Classical/*.npz
src/evaluation/anomalies/datasets/real/CV_by_ResNet18/*.npz
src/evaluation/anomalies/datasets/real/CV_by_ViT/*.npz
src/evaluation/anomalies/datasets/real/NLP_by_BERT/*.npz
src/evaluation/anomalies/datasets/real/NLP_by_RoBERTa/*.npz
```

## Synthetic evaluation datasets
The fixed, pre-generated GMM test sets used for reproducible density evaluation can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1ZS2p6qtxNdrt2IZ186t34RL-y9PQhzum?usp=sharing) and placed under [density_sets](density_sets). Each file is a `torch.save`d list of `(support, query, log_probs)` tuples loadable directly by `testsets_path`, and its filename encodes the generation config it was sampled with (see the corresponding enums in [src/configs.py](src/configs.py)):
`net_<network_type>_aug_<augment_type>_moment_<augment_moment>_mix_<mixture_type>_cat_<prob_cat>.pt`

## Evaluation
Evaluation is run via `evaluation.py`, using the same OmegaConf CLI-override style as pre-training. Point `model_path` at a checkpoint produced by pre-training:
```
python evaluation.py model_path=checkpoints/iced.pt device=cuda
```

