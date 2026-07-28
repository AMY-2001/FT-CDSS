
# Faulty Models

This folder contains the faulty CNN models generated using the fault injection framework.

The models were produced by injecting faults into the quantized INT8 activation tensors of the original compressed CNNs. Each faulty model represents a specific fault injection campaign with user-defined parameters.

## Fault Types

The generated faulty models support the following fault models:

- Stuck-at-0 (SA-0)
- Stuck-at-1 (SA-1)
- Bit Flip (Transient Upset)

## Configurable Parameters

Each faulty model can be generated using different configurations, including:

- Fault type
- Target bit position(s)
- Fault intensity (percentage of affected activation channels)
- Random seed
- Target CNN architecture

## Supported CNN Models

Faulty versions can be generated for:

- ResNet18
- MobileNetV2
- RegNet-Y-800MF

## Usage

The generated faulty models are used during the fault injection evaluation to assess the robustness of the redundancy architectures (DMR, TMR, and the proposed Hybrid architecture).

The faulty models are exported as TorchScript models and can be directly loaded for inference and evaluation.
