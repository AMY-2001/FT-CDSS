
SAMPLE-LEVEL PREDICTION LOGS

This folder contains the sample-level prediction logs generated during
the evaluation of the redundant fault-tolerant architectures.

The CSV files provide an image-level record of the predictions and
ground-truth labels used to calculate the reported evaluation results.
They are provided to support reproducibility and allow the reported
DMR, TMR, and Hybrid results to be independently inspected.

The logs include the sample identifier (image path), ground-truth
label, individual model predictions, and the final redundant
architecture decision where applicable.


1. DMR SAMPLE-LEVEL PREDICTIONS

The DMR CSV records the predictions produced by the two CNN replicas
and the resulting DMR decision for each evaluated test image.

Main columns:

- index:
  Sequential index of the evaluated test sample.

- relative_path:
  Relative path of the image within the test dataset. This serves as
  the sample identifier.

- true_label:
  Ground-truth class label assigned to the image.

- model_1_prediction:
  Prediction produced by the first CNN replica.

- model_2_prediction:
  Prediction produced by the second CNN replica.

- dmr_agreement:
  Indicates whether the two CNN replicas produced the same prediction.
  A value of 1 indicates agreement and 0 indicates disagreement.

- dmr_disagreement:
  Indicates whether the two CNN replicas produced different predictions.
  A value of 1 indicates disagreement and 0 indicates agreement.

- dmr_prediction:
  Final DMR prediction.

These records allow the DMR agreement/disagreement counts and the
reported DMR accuracy to be independently verified.


2. TMR SAMPLE-LEVEL PREDICTIONS


The TMR CSV records the predictions produced by the three CNN replicas
and the resulting majority-vote decision for each evaluated test image.

Main columns:

- index:
  Sequential index of the evaluated test sample.

- relative_path:
  Relative path of the image within the test dataset. This serves as
  the sample identifier.

- true_label:
  Ground-truth class label assigned to the image.

- model_1_prediction:
  Prediction produced by the first CNN replica.

- model_2_prediction:
  Prediction produced by the second CNN replica.

- model_3_prediction:
  Prediction produced by the third CNN replica.

- tmr_prediction:
  Final TMR prediction obtained using majority voting.

The TMR decision follows the majority-voting rule:

    TMR = (M1 AND M2) OR (M1 AND M3) OR (M2 AND M3)

These records allow the final TMR predictions and reported accuracy to
be independently verified at the sample level.


3. HYBRID SAMPLE-LEVEL PREDICTIONS


The Hybrid CSV records the predictions used by the proposed Hybrid
redundancy architecture.

The Hybrid architecture first evaluates two CNN replicas:

    M1 and M2

If the two predictions agree, the DMR path is used and the common
prediction becomes the final decision.

If the two predictions disagree, the third CNN replica is activated
and a TMR majority vote is performed.

Main columns:

- index:
  Sequential index of the evaluated test sample.

- relative_path:
  Relative path of the image within the test dataset. This serves as
  the sample identifier.

- true_label:
  Ground-truth class label assigned to the image.

- model_1_prediction:
  Prediction produced by the first CNN replica.

- model_2_prediction:
  Prediction produced by the second CNN replica.

- model_3_prediction:
  Prediction produced by the third CNN replica when the TMR path is
  required.

- hybrid_prediction:
  Final prediction produced by the Hybrid redundancy architecture.

The execution path can be determined from the first two model
predictions:

    M1 = M2  -> DMR path
    M1 != M2 -> TMR path

Thus, the sample-level records allow the number of DMR and TMR path
uses, as well as the final Hybrid accuracy, to be independently
verified.


4. DATASET AND SAMPLE IDENTIFICATION

The CSV files correspond to the test samples used in the reported
fault-tolerance experiments.

The "relative_path" column is used as the sample identifier. It
identifies the original image within its corresponding test-dataset
directory.

The test samples are evaluated using the preprocessing transformations
described in the corresponding training and evaluation notebooks.

__________________________________________________________________________
These CSV files are provided as evaluation logs rather than as model
inputs. They allow individual predictions to be inspected without
requiring the reviewer to reproduce the complete inference campaign.

The corresponding model construction, fault-injection, and redundancy
implementation notebooks are provided in the repository.

The DMR, TMR, and Hybrid predictions can be compared directly with the
"true_label" column to reproduce the reported sample-level accuracy.

For DMR, the agreement and disagreement columns can also be used to
verify the reported agreement statistics.

For Hybrid redundancy, comparing "model_1_prediction" and
"model_2_prediction" identifies whether the DMR or TMR path was used.

