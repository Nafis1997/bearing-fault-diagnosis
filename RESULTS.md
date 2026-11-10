# Executed results — CWRU bearing fault diagnosis

## Data and evaluation

36 official CWRU drive-end vibration recordings, three known fault classes, three
fault sizes and four motor loads. Initial training: 18 recordings / 1062 windows
(loads 0/1 hp). Validation: 9 recordings / 531 windows (2 hp). Final test: 9 recordings /
532 windows (3 hp). The selected model is refitted on training+validation before test.
No random mixing of windows from one recording across splits.

## Validation model comparison

Model | Accuracy | Balanced accuracy | Macro F1
--- | ---: | ---: | ---:
majority_baseline | 0.3333 | 0.3333 | 0.1667
svm_rbf | 1.0000 | 1.0000 | 1.0000
random_forest | 0.9906 | 0.9906 | 0.9906

## Held-out 3 hp test

Selected model: standardized RBF SVM. Accuracy = 1.0000, balanced accuracy = 1.0000,
macro F1 = 1.0000. All 532 windows were correctly classified in this execution.
This is a result on nine recordings, not 532 independent bearing specimens.
See classification_report.csv and per_recording_results.csv for class/record breakdowns.

## Interpretation

The features distinguish these controlled laboratory faults very well under this
fixed load holdout. Perfect benchmark classification does not establish perfect
industrial performance. The subset excludes healthy bearings and unknown defects;
recording-disjoint evaluation does not guarantee a physical-bearing-disjoint split.
Failure time is outside the target definition. Further model selection requires separate development data.

## Verification

The source MAT files passed channel/hash checks. Every recording belongs to one
initial split only. All classes occur in every split. Feature sanity checks passed
for finite outputs, invariance to DC offset, and spectral localization of a 1500 Hz sine.
Figures were generated from actual extracted data. See metrics.json for package versions.
