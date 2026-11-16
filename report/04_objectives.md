# 3. Objectives

1. Inspect the dataset schema, target labels, class distribution, missing values, and duplicates before modeling.
2. Preserve an untouched stratified test partition and fit all learned preprocessing on training data only.
3. Construct a hybrid representation combining TF-IDF text, job metadata, and deterministic scam-indicator/count features.
4. Compare Logistic Regression, Linear SVM, Random Forest, and XGBoost on a consistent experiment split, emphasizing fraud-class metrics and PR-AUC under class imbalance.
5. Save a complete preprocessing-and-classification pipeline and calibrate its probabilities using training data only.
6. Select a fraud decision threshold on training-only validation data, including an F1-oriented choice and a higher-recall alternative.
7. Evaluate the saved pipeline on the held-out test set without using it for selection.
8. Explain global and selected local predictions with a bounded SHAP sample while avoiding causal or definitive fraud claims.
9. Provide a local Streamlit interface with posting inputs, risk score, threshold-based label, detected indicators, and safety recommendations.
10. Document limitations and identify further validation and monitoring work.
