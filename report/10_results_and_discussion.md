# 9. Results and Discussion

## 9.1 Hybrid model comparison

The following results are from `report/hybrid_model_experiments.csv` on the held-out test partition (3,529 rows; 172 fraud cases). Model selection was performed earlier on training-only validation, not on this table.

| Hybrid candidate | Accuracy | Fraud precision | Fraud recall | Fraud F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| XGBoost (weighted) | 0.9853 | 0.8488 | 0.8488 | **0.8488** | **0.9914** | **0.9131** |
| Linear SVM (balanced; selected on validation) | 0.9819 | 0.8068 | 0.8256 | 0.8161 | 0.9856 | 0.8958 |
| Random Forest (balanced) | 0.9833 | **0.9065** | 0.7326 | 0.8103 | 0.9909 | 0.8984 |
| Logistic Regression (balanced) | 0.9680 | 0.6166 | **0.9070** | 0.7341 | 0.9864 | 0.8266 |

The validation PR-AUC of the selected Linear SVM was 0.9273 (validation fraud F1 0.8487), compared with 0.9217 validation PR-AUC for XGBoost. The predeclared validation selection rule therefore selected Linear SVM. XGBoost performed better on several held-out metrics, including fraud F1. This test-versus-validation ranking difference is important: using the test results to replace the selected model would turn the test set into model-selection data. It also shows that the saved model is not the top scorer on every reported test metric.

## 9.1.1 Comparison with the text-only baseline

The project’s text-only LinearSVC baseline uses the same held-out split and reports fraud precision 0.9353, recall 0.7558, F1 0.8360, ROC-AUC 0.9765, and PR-AUC 0.8879. The hybrid Linear SVM reports precision 0.8068, recall 0.8256, F1 0.8161, ROC-AUC 0.9856, and PR-AUC 0.8958. In this experiment, adding metadata and engineered signals coincided with increased recall and AUC values but reduced precision and F1. The hybrid XGBoost candidate's fraud F1 of 0.8488 exceeds the text-only LinearSVC's 0.8360, but this comparison also changes the classifier, so it does not isolate a feature-only effect. No ablation result is available to attribute the differences to any individual feature group.

The balanced Logistic Regression candidate had the highest fraud recall in this test comparison (0.9070) but lower fraud precision (0.6166), illustrating the cost of catching more fraud cases. Random Forest had the highest fraud precision (0.9065) but lower recall (0.7326). The SVM occupies a middle point under the default candidate evaluation threshold.

## 9.2 Saved calibrated model verification

The saved calibrated Linear SVM was verified separately on the same 3,529-row test split, using the classifier's default prediction threshold for this verification report. The reported fraud precision was 0.8662, recall 0.7907, F1 0.8267, ROC-AUC 0.9884, and PR-AUC 0.8981. Its confusion matrix was 3,336 true negatives, 21 false positives, 36 false negatives, and 136 true positives.

These figures are not the metrics for the validation-selected 0.40 threshold. The saved threshold was optimized only on training validation; the held-out test report was not rerun at that threshold for the reported verification values. Therefore, the verification figures above must not be presented as evaluation of app behavior at threshold 0.40.

## 9.3 Threshold analysis

Threshold selection used 2,824 training-only validation examples and did not use any of the final test rows. Selected points from the threshold table are:

| Threshold | Fraud precision | Fraud recall | Fraud F1 | False-positive rate | False-negative rate |
|---:|---:|---:|---:|---:|---:|
| 0.15 (high-recall alternative) | 0.698 | 0.927 | 0.796 | 0.020 | 0.073 |
| 0.40 (selected by validation F1) | 0.881 | 0.810 | **0.844** | 0.006 | 0.190 |
| 0.50 | 0.909 | 0.730 | 0.810 | 0.004 | 0.270 |
| 0.90 | 0.986 | 0.518 | 0.679 | 0.000 (rounded) | 0.482 |

The selected F1 threshold is 0.40. A lower threshold of 0.15 meets the specified 90% recall goal on validation (observed recall 0.927) but produces lower precision (0.698). A lower threshold can catch more fraudulent postings at the cost of more legitimate postings being flagged. Higher thresholds favor precision and fewer false alarms but miss a larger share of fraud.

## 9.4 Error analysis

The held-out saved-model analysis recorded 21 false positives among 3,357 legitimate postings and 36 false negatives among 172 fraudulent postings, under the default verification prediction rule. Examples of false positives included ordinary office, payroll, data-entry, and administrative roles; examples of false negatives included technical and administrative roles. This supports careful interpretation: familiar job titles or individual indicator patterns are not sufficient to determine whether a posting is genuine.

## 9.5 Explainability

The SHAP analysis explained 162 sampled test postings using contributions averaged across the three calibrated Linear SVM components. It describes the underlying SVM decision margin, not calibrated probability points. The largest global mean absolute contributions were the engineered word-count and character-count features. Among the handcrafted scam indicators, work-from-home wording had the largest mean absolute contribution in this sample, followed by urgent-language and no-experience indicators. In this fitted model, the signed mean for work-from-home was slightly negative on the margin; this is an association in this sample, not a general rule about remote work.

Global importance ranks do not imply causal importance. Categorical and text features also contributed. Local examples show that features can push predictions in both directions for the same posting, and the project report presents three examples from each available true-positive, true-negative, false-positive, and false-negative group. The explanations do not prove fraud or legitimacy.

## 9.6 Overall discussion

The project demonstrates the feasibility of combining text with structured and engineered features, calibrating a locally usable pipeline, and selecting a threshold without test leakage. Its results also expose important limitations: class imbalance, false alarms and missed fraud, metric trade-offs, and model-selection uncertainty. The results are specific to the dataset and split and do not establish superiority over literature results from different experiments.
