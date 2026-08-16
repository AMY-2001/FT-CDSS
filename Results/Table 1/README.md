
# README — Table 1 Experimental Results

## Purpose

This directory contains the experimental log and supporting information required to reproduce the results reported in **Table 1** of the manuscript.

The results cover:

* Model size before pruning
* Model size after pruning
* Mult-add operations before and after pruning
* FLOPs reduction
* Model size after quantization
* Classification accuracy
* Pruning configurations for each dataset and CNN architecture

## Models

The evaluated CNN architectures are:

* **RN18** — ResNet18
* **RGY** — RegNetY-800MF
* **MNV2** — MobileNetV2

The models were evaluated on:

* Colon cancer
* MRI brain tumor
* Chest X-ray

A separate classifier model used for image routing is also reported.

## Experimental Log

The file:

`table_1_experimental_log.txt`

contains the raw numerical results used to construct Table 1, including the original model sizes, pruning results, FLOPs-related measurements, quantized model sizes, and accuracies.

## Pruning

The pruning ratio is specified manually in the pruning configuration. For example:

```python
importance = tp.importance.MagnitudeImportance(p=2)

pruner = tp.pruner.MagnitudePruner(
    model=model,
    example_inputs=example_inputs,
    importance=importance,
    pruning_ratio=0.7,
    ignored_layers=[m for _, m in ignored],
)
```

The pruning ratio is adjusted according to the model and dataset.

## FLOPs Calculation

The computational complexity is obtained using `torchinfo`:

```python
from torchinfo import summary

summary(model, input_size=(1, 3, 224, 224))
```

The reported **total mult-add operations** are multiplied by 2 to obtain FLOPs:

```text
FLOPs = Total mult-add operations × 2
```

## Model Size

Model size refers to the storage occupied by the serialized `.pt` model file on disk.

The reported size therefore represents the actual serialized model storage rather than the number of parameters or runtime memory consumption.

## Accuracy

Classification accuracy is calculated as:

```text
Accuracy (%) =
(Number of correct predictions / Total number of predictions) × 100
```

The accuracy results in the experimental log correspond to the values reported in Table 1.

## Reproducibility

The numerical values in `table_1_experimental_log.txt` are provided as the experimental record for Table 1. The corresponding pruning, model evaluation, FLOPs measurement, and quantization scripts are available in the repository.

To reproduce Table 1:

1. Load the corresponding trained CNN model.
2. Apply the dataset-specific pruning ratio specified in the pruning configuration.
3. Measure the post-pruning model size and mult-add operations.
4. Convert mult-add operations to FLOPs by multiplying by 2.
5. Quantize the pruned model.
6. Measure the serialized `.pt` file size after quantization.
7. Evaluate the resulting model on the corresponding test set.
8. Use the recorded values to regenerate the Table 1 results.
