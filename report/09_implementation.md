# 8. Implementation

## 8.1 Data processing

The inspection workflow verified the `fraudulent` target and raw schema. Preprocessing removed exact duplicates and normalized blank fields. The text construction function concatenates title, company profile, description, requirements, and benefits, then lowercases, strips HTML, replaces URL/email strings with tokens, and normalizes whitespace. Numeric signals are computed from the original text or cleaned combined text as appropriate.

## 8.2 Hybrid pipeline

The hybrid `ColumnTransformer` processes text, categorical metadata, and numerical indicators with separate train-fitted transformations. It is wrapped with a classifier in a scikit-learn `Pipeline` so the same learned transformations are applied during validation, testing, calibration, and app inference. `in_balanced_dataset` is not used as a model feature.

Four classifier families were tested: balanced Logistic Regression, balanced Linear SVM, balanced Random Forest, and weighted XGBoost. The hybrid model selection criterion is training-validation PR-AUC (average precision), with fraud F1 as a tie-breaker. This selected the Linear SVM pipeline. A calibrated version wraps the selected pipeline with three-fold sigmoid calibration fit only on training data.

## 8.3 Application

`app.py` loads `models/fake_job_detector.joblib` and the selected threshold from `models/model_threshold.json`. The user supplies text and job metadata. The app builds the combined-text feature and deterministic engineered features; learned transforms are performed by the saved pipeline. Fraud probability is read from the class-1 probability column and compared with the configured threshold. The interface displays **Likely legitimate** or **Potentially fraudulent**, a probability, warning indicators, recommendations, and a disclaimer. All inference is local.

## 8.4 Reproducibility checks

The project verifies raw/processed row alignment before evaluation. The input test script exercises representative postings through the Streamlit testing interface, checking probability/threshold consistency and warning text. The tests do not alter or refit the model.
