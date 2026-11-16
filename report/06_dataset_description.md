# 5. Dataset Description

## 5.1 Source and schema

The source file is `data/raw/DataSet.csv`. The inspection notebook found 17,880 rows and 18 columns. The schema contains five text fields (`title`, `company_profile`, `description`, `requirements`, and `benefits`); job metadata (`location`, `department`, `salary_range`, `employment_type`, `required_experience`, `required_education`, `industry`, and `function`); three binary metadata fields (`telecommuting`, `has_company_logo`, and `has_questions`); the target (`fraudulent`); and `in_balanced_dataset`.

The target values are `f` for legitimate and `t` for fraudulent. The raw inspection found 17,014 legitimate and 866 fraudulent examples (95.16% and 4.84%). Preprocessing removed 235 exact duplicate rows, producing 17,645 records: 16,787 legitimate and 858 fraudulent (95.14% and 4.86%). The column `in_balanced_dataset` is excluded as a predictor because its provenance may reflect sampling or label construction.

## 5.2 Missing values and quality

Several optional fields are frequently missing. In the deduplicated data, salary range is missing in 14,813 rows; department in 11,368; required education in 8,028; benefits in 7,103; and required experience in 6,976. Blank and whitespace-only strings are treated as missing. Optional missingness does not trigger row removal. Categorical inputs are imputed with a constant missing token; numerical features use training-fitted median imputation.

## 5.3 Text processing and engineered features

The five text fields are concatenated, lowercased, stripped of HTML tags, have URLs replaced by `URL` and email addresses by `EMAIL`, and have whitespace normalized. URL and email counts are calculated from original text. The engineered numerical features include description and requirements lengths, word and character counts, URL/email/phone counts, exclamation count, missing company-profile indicator, urgent-language, fee-request, easy-money, no-experience, work-from-home, cryptocurrency-payment, and WhatsApp/Telegram indicators.

These features are heuristic signals, not ground truth. For example, remote work and no-experience wording also occur in legitimate advertisements. Phone-like strings can be false matches, and simple phrase rules may miss variants or misunderstand context.

## 5.4 Split and class imbalance

The modeling workflow uses a stratified 80/20 split with `random_state=42`. The held-out test set contains 3,529 postings, including 172 fraudulent and 3,357 legitimate examples. The training partition contains 14,116 rows. An inner stratified split within training is used for model selection; a validation split within training is used independently for threshold selection. No test examples are used to fit, calibrate, select models, or select thresholds.

The class imbalance means accuracy alone is insufficient. Fraud precision, fraud recall, F1, confusion matrices, ROC-AUC, and PR-AUC are considered.
