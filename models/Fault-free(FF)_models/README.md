
# Fault-Free Models

## Overview

This folder contains the final fault-free CNN models used throughout the redundancy evaluation presented in the paper.

## Model Format

All models are provided as serialized TorchScript (`.pt`) files.

The models can be loaded using:

```python
import torch

model = torch.jit.load("model_name.pt", map_location="cpu")
model.eval()
```

## Purpose

These fault-free models serve as the reference (pristine) replicas used by the redundancy architectures:

- Dual Modular Redundancy (DMR)
- Triple Modular Redundancy (TMR)
- Proposed Hybrid Redundancy

During fault-injection experiments, one replica is replaced with a corresponding fault-injected model, while the remaining replicas use the fault-free models contained in this folder.

## Notes

- the specification of pruning and quantization are descripted in detail in the paper.

- The models are quantized and optimized for CPU-only inference.
- The models were generated using the training, pruning, and quantization procedures provided in the corresponding folders of this repository.
- These models are intended for deployment and evaluation only. To retrain or regenerate the models, refer to the Training, Pruning, and Quantization folders.
