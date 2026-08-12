# Training Notebooks

This directory contains the training notebooks used to train the CNN models evaluated in the proposed CDSS framework.

The provided notebooks cover the following architectures:

- ResNet18 (RN18)
- MobileNetV2 (MNV2)
- RegNet-Y-800MF (RGY)

## Directory Structure

### ResNet18

Separate training notebooks are provided for each dataset:

- `RN18_colon_training.ipynb` — Colon cancer
- `RN18_MRI_training.ipynb` — Brain tumor MRI
- `RN18_Xray_training.ipynb` — Pneumonia chest X-ray

### MobileNetV2

Separate training notebooks are provided for each dataset:

- `MNV2_colon_training.ipynb` — Colon cancer
- `MNV2_MRI_training.ipynb` — Brain tumor MRI
- `MNV2_Xray_training.ipynb` — Pneumonia chest X-ray

### RegNet-Y-800MF

A single configurable notebook is provided:

- `RGY_General_training.ipynb`

The `RGY_General_training.ipynb` notebook contains the training configurations for all three datasets. The dataset-specific paths, preprocessing transformations, and augmentation settings are provided within the notebook. 

To train the model for a specific dataset, the corresponding configuration is enabled while the configurations for the other datasets remain commented out. This avoids duplicating the same training code across three separate notebooks while retaining the dataset-specific training settings.

The notebook therefore supports:

- **Colon cancer**
- **Brain tumor MRI**
- **Pneumonia chest X-ray**

For each dataset, the corresponding dataset path, preprocessing transformations, and augmentation configuration should be uncommented before execution.

## Training Configuration

The notebooks contain the complete training procedures used for the reported models, including:

- Dataset loading
- Training/validation data preparation
- Image preprocessing
- Data augmentation
- Model initialization
- Training configuration
- Validation
- Model evaluation
- Model checkpoint saving

The dataset-specific preprocessing and augmentation settings are explicitly defined in the corresponding notebooks.
