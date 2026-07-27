
# Fault Injection

This folder contains the implementation of the fault injection framework used in the paper.

The proposed implementation injects faults into the quantized INT8 activation tensors of the neural network using the PyTorch FX graph. The framework supports three fault models:

- Stuck-at-0 (SA-0)
- Stuck-at-1 (SA-1)
- Bit Flip (Transient Upset)

## Injection Method

Faults are injected after feature extraction by modifying the INT8 activation tensor using `torch.int_repr()`. The modified tensor is then reconstructed using the original quantization parameters (scale and zero-point) before continuing the inference process.

The fault injection process consists of:

1. Loading the trained quantized CNN.
2. Selecting the target activation tensor.
3. Randomly selecting the affected activation channels.
4. Generating the fault mask according to the selected fault model.
5. Injecting the fault using bitwise operations.
6. Reconstructing the quantized tensor.
7. Saving the faulty model for evaluation.

## Configurable Parameters

The following parameters can be configured before each fault injection campaign:

- Fault model (SA-0, SA-1, or Bit Flip)
- Target bit position(s)
- Percentage of affected activation channels
- Random seed
- Target CNN architecture

## Supported CNN Models

- ResNet18
- MobileNetV2
- RegNet-Y-800MF

## Reproducibility

Random fault locations are generated using a fixed random seed to ensure reproducibility of the reported experiments.

The generated faulty model is exported as a TorchScript model and can be directly evaluated using the evaluation notebooks provided in this repository.
