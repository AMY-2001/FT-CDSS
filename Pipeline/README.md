
# Multiprocessing Deployment Pipeline

## Overview

This folder contains the multiprocessing implementation of the proposed multimodal clinical decision support system (CDSS) deployed on a Raspberry Pi 4.

The objective of this implementation is to execute three disease-specific units concurrently using the four available CPU cores of the Raspberry Pi. The pipeline enables automatic routing of incoming medical images to the corresponding diagnostic model while evaluating different redundancy architectures (Baseline (simplex), DMR, TMR, and the proposed Hybrid redundancy).

The implementation follows the deployment architecture presented in the manuscript.

---

# Pipeline Overview
The pipeline is implemented using Python multiprocessing, where every diagnostic unit executes independently.

---

# Hardware Platform

The implementation targets:

- Raspberry Pi 4
- CPU-only execution
- Quantized TorchScript models
- QNNPACK inference engine

Each process is pinned to a dedicated CPU core using CPU affinity.

---

# Code Organization

The inference backend is configured to use the QNNPACK engine

```python
torch.backends.quantized.engine = "qnnpack"
```

which is optimized for quantized inference on ARM processors.

---

## 2. Selecting the Deployment Mode

The deployment supports multiple redundancy architectures.

The desired architecture is selected by changing

```python
INFERENCE_MODE
```

Possible values are

```python
BASELINE
DMR
TMR
```

Each mode loads a different TorchScript model while keeping the remaining deployment pipeline unchanged.

This allows all architectures to be evaluated under identical execution conditions.

---

## 3. Selecting the Routing Strategy

The pipeline supports two operating modes controlled by

```python
USE_CLASSIFIER
```

### USE_CLASSIFIER = True

When enabled, every incoming image is first processed by a lightweight CNN classifier.

The classifier determines whether the image belongs to

- Colon histopathology
- Brain MRI
- Chest X-ray

Based on the predicted modality, the image is automatically forwarded to the corresponding diagnostic queue.

This configuration represents the complete multimodal CDSS proposed in the manuscript.

---

### USE_CLASSIFIER = False

When disabled, the classifier is bypassed.

Instead, images are directly assigned to the appropriate diagnostic queue according to their dataset location.

This mode is primarily intended for debugging and evaluation of the diagnostic units without including the routing stage.

---

## 4. Loading the Diagnostic Models

Each diagnostic unit loads its own TorchScript model.

---

## 5. Image Preprocessing

Two preprocessing pipelines are implemented.

### Colon Histopathology

### MRI and Chest X-ray

Because these datasets contain grayscale images, they are first converted into RGB format before applying resizing and normalization.
same preprocessing employed during model training is used.
This ensures compatibility with the CNN architectures.

---

## 6. Loading Class Labels

The implementation automatically retrieves the class names for each dataset using

```python
datasets.ImageFolder()
```

These labels are later used to convert the numerical CNN predictions into human-readable diagnostic labels.

---

## 7. Image Loading

The function

```python
load_image()
```

is responsible for

- opening an image
- applying the appropriate preprocessing pipeline
- adding the batch dimension required by PyTorch

Every image passes through this function before inference.

---

## 8. Diagnostic Process

The function

```python
run_model()
```

implements a single diagnostic process.

Each diagnostic process performs the following operations:

1. Pins itself to a dedicated CPU core.
2. Loads its TorchScript model.
3. Waits for incoming image paths from its queue.
4. Loads and preprocesses each image.
5. Executes CNN inference.
6. Interprets the output according to the selected redundancy architecture.
7. Sends the final prediction to the result queue.

The same function is reused for the Colon, MRI, and Chest X-ray diagnostic units.

---

## 9. Image Modality Classifier

The function

```python
classifier_process()
```

implements the image routing stage.

Its objective is **not** disease diagnosis.

Instead, it determines the imaging modality of each incoming image.

The classifier predicts one of three modalities:

- Colon
- Brain MRI
- Chest X-ray

According to the prediction, the image path is placed into one of the three multiprocessing queues.

When all images have been processed, the classifier inserts a termination signal (`None`) into every queue to indicate that no additional images remain.

---

## 10. Queue-Based Communication

Communication between processes is implemented using Python multiprocessing queues.

Four queues are created:

- Colon queue
- MRI queue
- Chest X-ray queue
- Result queue

The classifier places image paths into the appropriate diagnostic queue.

Each diagnostic process continuously retrieves image paths from its own queue, performs inference, and stores the prediction inside the result queue.

This producer-consumer structure allows all diagnostic processes to execute concurrently.

---

## 11. CPU Affinity

Each process is explicitly assigned to a dedicated CPU core using

```python
psutil.Process().cpu_affinity()
```

The core assignment is

Image Classifier --> Core 0 
Colon Diagnosis --> Core 1 
MRI Diagnosis --> Core 2 
Chest X-ray Diagnosis --> Core 3 

This minimizes unnecessary process migration during execution.

---

## 12. Thread Configuration

Each diagnostic process executes

```python
torch.set_num_threads(1)
```

This restricts every CNN to a single CPU thread.

Using one thread per process prevents contention between the three concurrently executing diagnostic units and ensures that each process exclusively utilizes its assigned CPU core.

---

## 13. Timing

Execution time is measured for the complete deployment pipeline.

The measured time includes

- image routing
- queue communication
- CNN inference
- multiprocessing overhead
- process synchronization

Milestone timestamps are also recorded during execution to evaluate the progression of the pipeline as additional images are processed.

---

## Execution Flow

The overall execution sequence is

1. Configure the deployment mode.
2. Load the required CNN models.
3. Create multiprocessing queues.
4. Launch the three diagnostic processes.
5. Optionally launch the image classifier.
6. Route incoming images.
7. Perform diagnosis concurrently.
8. Collect the predictions.
9. Record execution time.
10. Terminate all processes after the queues become empty.

---

## Notes

The deployment implementation is independent of the evaluated redundancy architecture.

Switching between the Baseline, DMR, TMR, or Hybrid configurations only requires replacing the deployed TorchScript models, while the multiprocessing framework remains unchanged.
