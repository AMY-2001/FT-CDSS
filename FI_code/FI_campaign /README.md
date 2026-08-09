# Fault-Injection Campaign Data

This directory contains the fault-injection campaign data used to characterize the effect of injected faults on CNN classification accuracy for the three medical-imaging datasets used in the study:

- `colon_campaign.txt` — Colon histopathology dataset (LC25000)
- `MRI_campaign.txt` — Brain MRI dataset
- `Xray_campaign.txt` — Chest X-ray pneumonia dataset

The campaign data were used to examine the sensitivity of the deployed ResNet-18 models to different fault types, fault-injection rates, and bit positions.

---

## Purpose of the Campaign

The fault-injection campaign was performed to empirically characterize the effect of transient bit faults on CNN inference accuracy.

The evaluated fault types include:

- **SEU** — Single-Event Upset
- **DNU** — Double-Node Upset
- **TNU** — Triple-Node Upset
- **QNU** — Quadruple-Node Upset

The campaign evaluates the classification accuracy obtained under different fault-injection conditions.

The campaign results were also used to identify fault conditions that produce a meaningful degradation in classification accuracy and to select representative fault-injection configurations for the subsequent fault-tolerance experiments.

---

## Fault-Injection Metrics

Each campaign records classification accuracy (%) under the corresponding fault-injection condition.

The reported values represent the accuracy of the CNN after applying the specified fault condition.

The campaign dimensions may include:

- Fault type
- Fault-injection percentage
- Target bit group / bit position
- Injection mode

For the colon campaign, the evaluated injection modes include:

- **SA-0** — stuck-at-0
- **SA-1** — stuck-at-1
- **Upset** — bit-flip condition

The colon campaign therefore provides a broader characterization of the effect of different fault models.

---

## Colon Dataset

`colon_campaign.txt`

The colon campaign evaluates SEU, DNU, TNU, and QNU faults over multiple injection rates and bit groups.

For each fault type, the campaign records accuracy for:

- 20% injection
- 40% injection
- 60% injection
- 80% injection

and for the corresponding bit groups and injection modes.

The campaign was used to examine how increasing fault-injection intensity and targeting different bit positions affect classification accuracy.

The results show that faults affecting higher-order/significant bits can produce substantially larger accuracy degradation than faults affecting less significant bits.

---

## Brain MRI Dataset

`mri_campaign.txt`

The MRI campaign contains the empirical fault-injection results obtained using the ResNet-18 MRI classifier.

The campaign focuses on **bit-flip (upset) faults**, with particular attention to higher-order bits because the preliminary campaign results showed that these bit positions produced more significant accuracy degradation.

The evaluated injection rates include:

- 10%
- 20%
- 25%
- 30%
- 40%
- 50%

The campaign considers several high-order bit groups for SEU, DNU, TNU, and QNU faults.

This campaign was used to identify representative and more severe fault conditions for the subsequent fault-tolerance experiments.

---

## Chest X-ray Dataset

`xray_campaign.txt`

The X-ray campaign contains the empirical fault-injection results obtained using the ResNet-18 pneumonia classifier.

The evaluated fault types are:

- SEU
- DNU
- TNU
- QNU

The campaign evaluates injection rates of:

- 10%
- 20%
- 30%
- 40% and 45%
- 50%
- 60%

The results characterize the degradation in classification accuracy as the fault-injection rate increases and provide evidence for the sensitivity of the model to different fault types.


