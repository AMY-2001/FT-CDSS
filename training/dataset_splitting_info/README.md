This folder contains the dataset split manifests used in the experiments
reported in the manuscript.

The manifests provide the image filenames/identifiers assigned to the
training, validation, and test partitions for the three evaluated datasets:

- Colon cancer (LC25000 colon subset)
- Brain tumor MRI
- Pneumonia chest X-ray

For each dataset, both TXT and CSV formats are provided. The manifests
correspond to the actual partitions used in the reported experiments.

The CSV files contain the image information in a structured format,
including the corresponding dataset partition and class where available.
The TXT files provide the image identifiers/paths for each partition.

OVERLAP CHECK
-------------

The image identifiers in the manifests can be compared between the training,
validation, and test files to verify whether identical image files occur
across partitions.

An overlap check should report zero common image identifiers between:

    Training ∩ Validation
    Training ∩ Test
    Validation ∩ Test

This check verifies separation at the identical-file/identifier level.

For datasets that do not provide patient-level or original-image identifiers,
the manifests cannot establish patient-level independence or determine
whether transformed/augmented images originated from the same underlying
source image.

use this code snippet for the checkup:

```
from pathlib import Path
import pandas as pd

Info_dir = Path("folder_consist_csv_and_text")

train = pd.read_csv(info_dir / "train_manifest.csv")
val   = pd.read_csv(info_dir / "validation_manifest.csv")
test  = pd.read_csv(info_dir / "test_manifest.csv")

train_ids = set(train["relative_path"])
val_ids   = set(val["relative_path"])
test_ids  = set(test["relative_path"])

train_val = train_ids & val_ids
train_test = train_ids & test_ids
val_test = val_ids & test_ids

all_three = train_ids & val_ids & test_ids

print("===========================================")
print("data splitting check reportt")
print("===========================================")

print(f"Training images   : {len(train_ids)}")
print(f"Validation images : {len(val_ids)}")
print(f"Testing images    : {len(test_ids)}")

print("\nPairwise overlap:")
print(f"Train ∩ Validation : {len(train_val)}")
print(f"Train ∩ Test       : {len(train_test)}")
print(f"Validation ∩ Test  : {len(val_test)}")

print(f"\nPresent in ALL three splits: {len(all_three)}")


if len(train_val) == 0 and len(train_test) == 0 and len(val_test) == 0:
    print("\n No identical image identifiers occur across splits.")
else:
    print("\n Overlapping image identifiers were detected.")

    if train_val:
        print("\nTrain/Validation overlap:")
        print(list(train_val)[:20])

    if train_test:
        print("\nTrain/Test overlap:")
        print(list(train_test)[:20])

    if val_test:
        print("\nValidation/Test overlap:")
        print(list(val_test)[:20])
```
