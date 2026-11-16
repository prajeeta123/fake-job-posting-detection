# Saved Model Verification Report

## Verification outcome

- Model file: `models\fake_job_detector.joblib`
- Load successful: yes
- Saved object: `CalibratedClassifierCV`
- Complete pipeline: yes; calibrated wrapper contains a fitted preprocessing-and-classification pipeline
- Classifier: `LinearSVC`
- Classifier parameters: `{'C': 1.0, 'class_weight': 'balanced', 'max_iter': 50000, 'tol': 0.01, 'random_state': 42}`
- Calibration: `sigmoid (Platt scaling)` using 3-fold CV on the training partition
- Test split: stratified 80/20, `random_state=42`, reconstructed to match the hybrid workflow
- Test rows: 3,529; exact text and target row alignment checks passed
- Test rows used for training/calibration/feature selection: no

## Preprocessing pipeline

- `preprocessor`: `ColumnTransformer`
- `classifier`: `LinearSVC`

The full transformers are part of the serialized model, so application inputs go through the same fitted text, categorical, and numerical processing used by the classifier.

## Held-out test metrics

- Fraud precision: **0.8662**
- Fraud recall: **0.7907**
- Fraud F1-score: **0.8267**
- ROC-AUC: **0.9884**
- PR-AUC (average precision): **0.8981**

### Confusion matrix

Rows are actual classes; columns are predicted classes.

| Actual class | Predicted legitimate | Predicted fraudulent |
|---|---:|---:|
| Legitimate | 3336 | 21 |
| Fraudulent | 36 | 136 |

### Classification report

```text
precision    recall  f1-score   support

Legitimate (f)       0.99      0.99      0.99      3357
Fraudulent (t)       0.87      0.79      0.83       172

      accuracy                           0.98      3529
     macro avg       0.93      0.89      0.91      3529
  weighted avg       0.98      0.98      0.98      3529
```

### Five examples from each actual class

| Title | Actual label | Predicted label | Fraud probability |
|---|---|---|---:|
| Sales Representative | Legitimate (f) | Legitimate (f) | 0.95% |
| File Clerk | Legitimate (f) | Legitimate (f) | 0.22% |
| Director, Information Security  | Legitimate (f) | Legitimate (f) | 2.76% |
| English Teacher Overseas | Legitimate (f) | Legitimate (f) | 0.05% |
| Sales Manager Netigate Sverige | Legitimate (f) | Legitimate (f) | 0.02% |
| Urgent Requirement : Senior Exchange & Active Directory Systems Administrator for Qatar. | Fraudulent (t) | Fraudulent (t) | 67.74% |
| LEAN Program Manager, Carol Stream, IL | Fraudulent (t) | Legitimate (f) | 5.31% |
| Customer Service Representative | Fraudulent (t) | Fraudulent (t) | 98.40% |
| Junior Project Manager | Fraudulent (t) | Legitimate (f) | 0.77% |
| Finance Assistant | Fraudulent (t) | Fraudulent (t) | 82.47% |

## Deployment readiness

The artifact loaded, includes the complete preprocessing pipeline, and returns calibrated fraud probabilities. It is technically ready for a local application smoke test, provided the app constructs the same input schema and feature definitions. Predictions are risk estimates, not proof that a posting is fraudulent; retain safety guidance and human verification.

The test set was used only for this final evaluation. Do not use these results to tune a threshold or select features/models; doing so would turn it into validation data. The fraud class is much smaller than the legitimate class, so accuracy alone would hide errors. Review precision and recall together and validate the app's input feature construction against the training schema.
