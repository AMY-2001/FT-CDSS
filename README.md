This repository contains the implementation and supporting materials for the proposed fault-tolerant multimodal Clinical Decision Support System for edge-AI computing.

The system integrates multiple CNN-based diagnostic units for:
- Colon cancer histopathology
- Brain tumor MRI
- Pneumonia chest X-ray

The implementation includes model training, pruning, quantization, redundancy mechanisms, fault injection, and the pipeline-parallel execution framework used in the study.

## Repository Structure

- `training/` — Model training scripts and notebooks.
- `pruning/` — Model pruning implementations and configurations.
- `Quantization/` — Model quantization procedures.
- `Redundancy/` — DMR, TMR, and hybrid redundancy implementations.
- `FI_code/` — Fault-injection implementation and experimental configurations.
- `Pipeline/` — Pipeline-parallel execution and performance evaluation.
- `models/` — Pretrained model files used in the experiments.
- `Requirement` — Required Python packages and dependencies.

## Reproducibility

The repository provides the implementation required to reproduce the main training, model optimization, redundancy, fault-injection, and pipeline evaluation procedures reported in the study.

The experiments were conducted on a Raspberry Pi 4 Model B using PyTorch and the QNNPACK quantized backend.

## Dataset

The medical datasets used in this study are publicly available from their respective sources and are not redistributed in this repository. Dataset preparation and partition information are provided where applicable.
