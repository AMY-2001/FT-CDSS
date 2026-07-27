
# Redundancy Evaluation

This folder contains the implementation of the redundancy architectures proposed in the paper.

The notebook evaluates the behavior of three redundant neural network configurations using the quantized CNN models generated during the previous stages of the workflow.

## Implemented Architectures

The following redundancy schemes are implemented:

- **Dual Modular Redundancy (DMR)**
- **Triple Modular Redundancy (TMR)**
- **Proposed Hybrid Redundancy**

### DMR

The DMR architecture executes two identical CNN replicas in parallel.

- If both replicas agree, the prediction is accepted.
- If a disagreement occurs, the sample is flagged as a detected fault.

### TMR

The TMR architecture executes three identical CNN replicas simultaneously.

The final prediction is obtained using majority voting.

### Proposed Hybrid Architecture

The proposed hybrid redundancy operates in two stages:

1. Two CNN replicas execute in parallel (DMR mode).
2. If both replicas agree, the prediction is immediately accepted.
3. If a disagreement is detected, the spare CNN is activated and a majority vote is performed.

This implementation allows the spare network to remain inactive during fault-free operation, reducing computational cost while preserving fault tolerance.

## Inputs

The notebook requires:

- Fault-free quantized TorchScript model (`fault_free_model.pt`)
- Fault-injected TorchScript model (`faulty_model.pt`)
- Test dataset prepared using the preprocessing described in the paper

## Evaluation

The notebook evaluates the redundant architectures by reporting:

- Classification accuracy
- Agreement and disagreement counts (DMR)
- Fault detection coverage
- Fault recovery coverage
- Silent Data Corruption (SDC)
- False-positive rate (FPR)
- False-negative rate (FNR)
- End-to-end diagnostic accuracy
- DMR/TMR path utilization (Hybrid architecture)

## Fault Assumptions

The redundancy mechanism assumes a **single faulty neural network** at a time. The remaining replicas are assumed to operate correctly.

The proposed architecture targets **mutually independent network-level transient faults**. Common-mode failures and simultaneous faults affecting multiple replicas are outside the scope of the current implementation.

## Reproducibility

The notebook reproduces the redundancy experiments reported in the paper using the supplied fault-free and fault-injected CNN models. Different fault campaigns can be evaluated by replacing the faulty model while keeping the remaining evaluation procedure unchanged.
