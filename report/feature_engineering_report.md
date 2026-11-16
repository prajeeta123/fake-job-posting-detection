# Feature Engineering Report

## Dataset and duplicates

- Source: `data/raw/DataSet.csv`
- Rows before deduplication: 17,880
- Exact duplicate rows removed: 235
- Rows after deduplication: 17,645
- Target: `fraudulent`; `f` = legitimate, `t` = fraudulent.
- Class distribution after deduplication:

| class | count |
| --- | --- |
| Legitimate (f) | 16787 |
| Fraudulent (t) | 858 |

## Missing-value handling

Whitespace-only strings were changed to missing values. Text is filled with empty strings only for joining; the missing-profile indicator preserves missing company-profile information. Categorical predictors use a constant missing token; engineered numeric columns use the median learned from training rows. Optional-metadata missingness does not cause rows to be dropped.

| column | missing_count |
| --- | --- |
| salary_range | 14813 |
| department | 11368 |
| required_education | 8028 |
| benefits | 7103 |
| required_experience | 6976 |
| function | 6378 |
| industry | 4849 |
| employment_type | 3435 |
| company_profile | 3287 |
| requirements | 2650 |
| location | 343 |
| title | 0 |
| description | 0 |
| telecommuting | 0 |
| has_company_logo | 0 |
| has_questions | 0 |
| fraudulent | 0 |
| in_balanced_dataset | 0 |

## Text cleanup

Available text fields: `title`, `company_profile`, `description`, `requirements`, `benefits`. Fields are concatenated; text is lowercased, HTML tags removed, URLs replaced with `URL`, email addresses with `EMAIL`, and whitespace normalized. URL/email counts use original text.

## Engineered features and why they may help

| feature | meaning_and_possible_use |
| --- | --- |
| description_length | Description character count; gives context about how much information is supplied. |
| requirements_length | Requirements character count; short or vague requirements may merit review. |
| word_count | Word-like tokens in cleaned combined text; measures text volume. |
| character_count | Characters in cleaned combined text; another measure of information volume. |
| url_count | URLs in original text; helps surface external links for review. |
| email_count | Email addresses in original text; captures direct contact details. |
| phone_count | Phone-like digit sequences; captures direct contact, but can include false matches. |
| exclamation_count | Exclamation marks; may capture highly promotional or urgent tone. |
| missing_company_profile | One when company_profile is blank; indicates limited employer context. |
| contains_urgent_language | Urgent/ASAP/immediately/hiring-now phrases; may indicate pressure to act quickly. |
| contains_fee_request | Common applicant fee or deposit phrases; payment requests merit scrutiny. |
| contains_easy_money | Easy-money/quick-cash/guaranteed-income wording; captures unrealistic earning claims. |
| contains_no_experience | No-experience-required wording; occurs in legitimate entry-level jobs too. |
| contains_work_from_home | Work-from-home or remote wording; context, not a fraud verdict. |
| contains_cryptocurrency_payment | Crypto mentioned near payment/salary; captures potentially unusual payment method. |
| contains_whatsapp_or_telegram | WhatsApp/Telegram mention; helps review requests to move off-platform. |

## Categorical encoding and numeric scaling

Categorical predictors present and used: `location`, `department`, `salary_range`, `employment_type`, `required_experience`, `required_education`, `industry`, `function`, `telecommuting`, `has_company_logo`, `has_questions`.
Categoricals use `SimpleImputer(strategy='constant')` and `OneHotEncoder(handle_unknown='ignore', min_frequency=10)`. Rare categories are grouped to keep the sparse table practical.
Engineered numeric predictors use median imputation and `StandardScaler`. Imputers, encoder, and scaler are fitted on training rows only.
`in_balanced_dataset` is excluded as a predictor because its provenance may reflect sampling or label construction; the target is retained only as the label.

## Stratified split and leakage prevention

Rows were split 80/20 with stratification and random state 42 before fitting learned transformations. The train-fitted imputers, encoder, and scaler transform both partitions. No TF-IDF, SMOTE, or model training was done.

| split | f | t |
| --- | --- | --- |
| test | 95.13 | 4.87 |
| train | 95.14 | 4.86 |

## Feature distributions by class

The table reports mean engineered feature values for each class; for binary indicators, the mean is the positive rate. These summaries are descriptive, not evidence of causation. The plot compares indicator rates.

| feature | Legitimate (f) | Fraudulent (t) |
| --- | --- | --- |
| description_length | 1439.3646 | 1438.8252 |
| requirements_length | 744.0926 | 514.3473 |
| word_count | 406.0094 | 300.2063 |
| character_count | 2700.9891 | 2085.6142 |
| url_count | 0.0492 | 0.1737 |
| email_count | 0.0001 | 0.0 |
| phone_count | 0.048 | 0.1166 |
| exclamation_count | 0.7248 | 0.62 |
| missing_company_profile | 0.1612 | 0.6772 |
| contains_urgent_language | 0.0596 | 0.0944 |
| contains_fee_request | 0.0005 | 0.0012 |
| contains_easy_money | 0.0001 | 0.0035 |
| contains_no_experience | 0.0034 | 0.021 |
| contains_work_from_home | 0.0214 | 0.1037 |
| contains_cryptocurrency_payment | 0.0005 | 0.0 |
| contains_whatsapp_or_telegram | 0.0033 | 0.0 |

Plot: `report/feature_distributions_by_class.png`.

## Output

Processed CSV: `data/processed/processed_job_postings.csv` (17,645 rows and 515 columns). It includes cleaned combined text, raw engineered features, scaled numeric features, uniquely named one-hot categorical features, the target, and a train/test marker.
One-hot column names map to the original feature/category in `data/processed/onehot_feature_mapping.csv`.

## Limitations

Regex features can have false positives and negatives. Phone-like numbers may match non-phone digit sequences. No-experience and remote-work phrases also occur in legitimate postings; no single indicator establishes fraud. The target is imbalanced, so later model evaluation should emphasize fraud precision, recall, F1, ROC-AUC, PR-AUC, and a confusion matrix rather than accuracy alone.