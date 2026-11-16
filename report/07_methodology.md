# 6. Methodology

## 6.1 Preprocessing and features

The raw dataset is deduplicated and checked before it is split. Text is combined and deterministically cleaned. The hybrid model uses:

- TF-IDF text features with up to 10,000 features, unigrams and bigrams, `min_df=2`, and sublinear term frequency;
- one-hot encoded categorical features with rare-category grouping (`min_frequency=10`) and unknown-category handling;
- median-imputed and standardized numerical features, including counts and scam-indicator flags.

All learned transformations are part of a scikit-learn `ColumnTransformer` inside a complete `Pipeline`; they are fitted on training data only. The target is mapped as `f -> 0` and `t -> 1`.

## 6.2 Split and model comparison

Data is split once into a stratified 80% training partition and 20% held-out test partition with `random_state=42`. The training partition is further split for candidate model selection. Logistic Regression, balanced Linear SVM, balanced Random Forest, and weighted XGBoost are compared. Candidate selection uses training-validation PR-AUC (average precision), with fraud F1 as a tie-breaker. The held-out test is reserved for final evaluation and does not select the model.

## 6.3 Calibration and threshold selection

The selected Linear SVM pipeline is calibrated using three-fold sigmoid (Platt) calibration on training data. Threshold optimization recreates the outer split and sets aside a training-only validation subset. Calibrated fraud probabilities are evaluated at thresholds 0.10 to 0.90 in increments of 0.05. Precision, recall, F1, false-positive rate, and false-negative rate are computed for fraud as the positive class.

The threshold maximizing validation fraud F1 is selected, breaking ties in favor of a higher threshold. A separate safety-oriented threshold is the highest tested threshold achieving at least 90% validation fraud recall; if none reached that target, the workflow would report the best observed recall threshold. These selections do not use final test examples.

## 6.4 Evaluation and explainability

The saved calibrated model is evaluated on the original untouched stratified test partition with row-by-row alignment checks against the raw and processed data. The explainability notebook also uses the held-out test set, only to describe predictions and selected examples. SHAP values are calculated on at most 200 examples with a small training-only background sample. For the calibrated Linear SVM ensemble, component contributions are aligned and averaged on the underlying SVM decision-margin scale; they are not calibrated probability-point changes.

## 6.5 Reproducibility and interpretation

The saved artifacts comprise the fitted pipeline, metadata, and selected threshold. The Streamlit application constructs the same text and engineered input fields before passing them to that pipeline. Predictions are risk estimates learned from labeled examples. Neither a feature indicator nor a model output establishes that a posting is fraudulent.
