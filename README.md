# Vibration-Based Bearing Fault Diagnosis

This project classifies inner-race, ball and outer-race bearing faults from CWRU
vibration recordings. It combines signal processing with supervised learning to
support condition monitoring and maintenance planning.

## Dataset

The experiment uses 36 MATLAB recordings from the
[CWRU 12 kHz drive-end dataset](https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data).
Each recording contains a drive-end accelerometer signal sampled at 12,000 Hz.

| Variable | Values used |
| --- | --- |
| Fault type | Inner race, ball, outer race |
| Fault diameter | 0.007, 0.014, 0.021 inches |
| Motor load | 0, 1, 2, 3 hp |
| Outer-race position | @6, as labeled in the source table |
| Signal channel | Drive-end accelerometer (`DE_time`) |

The faults were introduced in a laboratory test rig. Healthy bearings and unknown
fault types are outside this experiment. Details of the apparatus are available
[here](https://engineering.case.edu/bearingdatacenter/apparatus-and-procedures).

`download_data.py` downloads the recordings from CWRU and creates a manifest with
source URLs, channel names, sample counts and SHA-256 hashes. Raw recordings remain
in `data/` and are downloaded separately. Dataset usage conditions apply independently
of any license chosen for the project code.

## Run the experiment

Use Python 3.11 or 3.12. From this project folder:

```bash
python -m pip install -r requirements.txt
python download_data.py
python main.py
```

Installation can be skipped if the dependencies are already available.
[START_HERE.md](START_HERE.md) explains the VS Code steps using the JKU environment.
`requirements-tested.txt` records the package versions used for the included results.

## Signal processing and classification

Each recording is divided into non-overlapping windows of 2,048 samples, or about
0.171 seconds. The incomplete final window is discarded. After removing the mean,
the following features are computed:

- RMS, standard deviation and peak-to-peak amplitude.
- Skewness, Pearson kurtosis, crest factor and shape factor.
- Spectral centroid and relative Welch PSD power in five frequency bands.

This produces 13 features per window. RMS and standard deviation are redundant
after mean removal; both are retained in this baseline.

Three models are compared: a majority-class baseline, an RBF SVM with standardized
features, and a Random Forest. Model selection uses validation macro F1. The SVM
scaler is fitted only on the fitting data during selection.

## Evaluation design

| Stage | Motor load | Recordings | Windows |
| --- | --- | ---: | ---: |
| Training | 0 and 1 hp | 18 | 1,062 |
| Validation | 2 hp | 9 | 531 |
| Test | 3 hp | 9 | 532 |

All windows from a recording stay in the same initial split. Recording identifiers,
load, fault size and window indices are excluded from the model inputs. The selected
model is refitted on training and validation recordings before final test evaluation.

## Results

The RBF SVM achieved validation macro F1 of **1.0000**, compared with **0.9906** for
the Random Forest. After refitting, the selected SVM classified all **532 test
windows** correctly: test accuracy, balanced accuracy and macro F1 were **1.0000**.

[RESULTS.md](RESULTS.md) contains the comparison and interpretation. The confusion
matrix and example vibration signals are saved under `results/`.

## Scope and limitations

The result describes a fixed load holdout on nine test recordings. Windows within
a recording are correlated, and recording-disjoint splits do not establish that
the physical bearing specimens are independent. The model has not been evaluated
on new machines, naturally developing faults, healthy bearings or unknown defects.
It diagnoses an existing fault type; remaining useful life and failure time are
separate prediction tasks.

Useful extensions include independent-bearing evaluation, a verified healthy class,
noise sensitivity tests, severity holdouts and feature ablation. Model changes
should be selected on development data rather than the final test recordings.

## Project files

| File | Purpose |
| --- | --- |
| `download_data.py` | Download and validate the CWRU subset |
| `main.py` | Extract features, fit models and evaluate predictions |
| `requirements.txt` | Dependency ranges |
| `requirements-tested.txt` | Versions used for the included run |
| `results/data_manifest.csv` | Recording provenance, hashes and split assignments |
| `results/window_features.csv` | Features and metadata for every window |
| `results/metrics.json` | Model comparison, final metrics and configuration |
| `results/test_predictions.csv` | True and predicted test labels |
| `results/per_recording_results.csv` | Accuracy and majority prediction by recording |
| `results/classification_report.csv` | Per-class precision, recall and F1 |
| `results/confusion_matrix.csv` | Test confusion counts |
| `results/confusion_matrix.png` | Confusion-matrix figure |
| `results/vibration_examples.png` | Example training signals and spectra |

