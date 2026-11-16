"""Fit a probability-calibrated copy of the selected hybrid pipeline for the app."""

import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split


PROJECT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_PATH = PROJECT_DIR / "data" / "processed" / "processed_job_postings.csv"
RAW_PATH = PROJECT_DIR / "data" / "raw" / "DataSet.csv"
BASE_MODEL_PATH = PROJECT_DIR / "models" / "best_hybrid_model.joblib"
APP_MODEL_PATH = PROJECT_DIR / "models" / "fake_job_detector.joblib"
METADATA_PATH = PROJECT_DIR / "models" / "fake_job_detector_metadata.json"

if not PROCESSED_PATH.is_file():
    raise FileNotFoundError(f"Processed dataset not found: {PROCESSED_PATH}")
if not RAW_PATH.is_file():
    raise FileNotFoundError(f"Raw dataset not found: {RAW_PATH}")
if not BASE_MODEL_PATH.is_file():
    raise FileNotFoundError(f"Selected hybrid pipeline not found: {BASE_MODEL_PATH}")


numeric_features = [
    "raw__description_length",
    "raw__requirements_length",
    "raw__word_count",
    "raw__character_count",
    "raw__url_count",
    "raw__email_count",
    "raw__phone_count",
    "raw__exclamation_count",
    "raw__missing_company_profile",
    "raw__contains_urgent_language",
    "raw__contains_fee_request",
    "raw__contains_easy_money",
    "raw__contains_no_experience",
    "raw__contains_work_from_home",
    "raw__contains_cryptocurrency_payment",
    "raw__contains_whatsapp_or_telegram",
]
categorical_candidates = [
    "location",
    "department",
    "salary_range",
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function",
    "telecommuting",
    "has_company_logo",
    "has_questions",
]

# Keep only processed columns used by the selected hybrid pipeline.
processed = pd.read_csv(
    PROCESSED_PATH,
    usecols=["combined_text", "fraudulent", *numeric_features],
    low_memory=False,
)
raw = pd.read_csv(RAW_PATH, low_memory=False).drop_duplicates().copy()
if len(raw) != len(processed):
    raise ValueError(
        "Cannot align source metadata with the processed dataset: "
        f"{len(raw)} deduplicated raw rows vs {len(processed)} processed rows."
    )

raw_columns = raw.columns.tolist()
categorical_features = [
    column for column in categorical_candidates if column in raw_columns
]
text_columns = [
    column
    for column in ["title", "company_profile", "description", "requirements", "benefits"]
    if column in raw_columns
]
if not categorical_features or not text_columns:
    raise ValueError("The inspected metadata/text columns are missing from the raw CSV.")

raw = raw[[*categorical_features, "fraudulent", *text_columns]].copy()
for column in raw.select_dtypes(include=["object", "string"]).columns:
    raw[column] = raw[column].replace(r"^\s*$", pd.NA, regex=True)

raw_text = raw[text_columns].fillna("").astype(str).agg(" ".join, axis=1)


def clean_text(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\b(?:https?://|www\.)\S+", " URL ", text, flags=re.IGNORECASE)
    text = re.sub(
        r"\b[\w.+%-]+@[\w.-]+\.[a-z]{2,}\b",
        " EMAIL ",
        text,
        flags=re.IGNORECASE,
    )
    return re.sub(r"\s+", " ", text).strip()


rebuilt_text = raw_text.map(clean_text).to_numpy()
processed_text = processed["combined_text"].fillna("").astype(str).to_numpy()
if not np.array_equal(rebuilt_text, processed_text):
    raise ValueError("Raw text and processed combined_text are not aligned row by row.")

raw_labels = raw["fraudulent"].astype(str).str.strip().str.lower().to_numpy()
processed_labels = processed["fraudulent"].astype(str).str.strip().str.lower().to_numpy()
if not np.array_equal(raw_labels, processed_labels):
    raise ValueError("Raw and processed fraud labels are not aligned row by row.")
if set(processed_labels) != {"f", "t"}:
    raise ValueError(f"Unexpected fraud label values: {sorted(set(processed_labels))}")

X = processed[["combined_text", *numeric_features]].copy()
for column in categorical_features:
    X[column] = raw[column].reset_index(drop=True)
X = X[["combined_text", *categorical_features, *numeric_features]]
y = pd.Series(processed_labels, index=X.index).map({"f": 0, "t": 1}).astype("int8")

# Re-create the same outer training partition from notebook 04. Test rows never calibrate the model.
X_train, _, y_train, _ = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42,
)

base_pipeline = joblib.load(BASE_MODEL_PATH)
# Use a practical tolerance and a high iteration cap for each calibration fold.
classifier = base_pipeline.named_steps["classifier"]
if classifier.__class__.__name__ == "LinearSVC":
    base_pipeline.set_params(classifier__max_iter=50_000, classifier__tol=1e-2)
calibrated_pipeline = CalibratedClassifierCV(
    estimator=base_pipeline,
    method="sigmoid",
    cv=3,
)
calibrated_pipeline.fit(X_train, y_train)
joblib.dump(calibrated_pipeline, APP_MODEL_PATH)

metadata = {
    "source_pipeline": str(BASE_MODEL_PATH.relative_to(PROJECT_DIR)),
    "calibrated_pipeline": str(APP_MODEL_PATH.relative_to(PROJECT_DIR)),
    "calibration": "sigmoid (Platt scaling)",
    "calibration_cv": 3,
    "training_rows": len(X_train),
    "held_out_test_rows_used": 0,
    "positive_label": "t",
    "label_mapping": {"f": 0, "t": 1},
    "categorical_features": categorical_features,
    "numeric_features": numeric_features,
    "text_feature": "combined_text",
}
METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
print(f"Saved calibrated app pipeline: {APP_MODEL_PATH}")
print(f"Calibration used {len(X_train):,} training rows; held-out test rows were not used.")
