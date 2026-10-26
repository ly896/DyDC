#  Dynamic Graph Anomaly Detection via Dual-Memory Self-Distillation and Long–Short-Term Behavioral Consistency Learning
#  Dynamic Graph Distillation and Consistency，DyDC

This repository contains a PyTorch implementation of a robust dynamic graph anomaly detection model. The model compares short-term temporal neighborhood behavior with long-term node memory and uses an EMA teacher encoder for stable self-distillation.

## Repository Structure

```text
data/                 CSV datasets used by the training script
evaluation/           Evaluation utilities
figure/               Plotting scripts for paper figures
model/                Temporal attention and model definition
modules/              Memory, message, and embedding modules
utils/                Data loading and graph neighborhood utilities
train.py              Main training entry point
```

## Installation

Create a Python environment and install the dependencies:

```bash
pip install -r requirements.txt
```

`torch-scatter` must match your PyTorch and CUDA version. If the default installation fails, install it from the official PyTorch Geometric wheel index for your environment.

## Data Format

Datasets are expected in `data/ml_<dataset>.csv` and should contain the following columns:

```text
u,i,ts,label,idx
```
