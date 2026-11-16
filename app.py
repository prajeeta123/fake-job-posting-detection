import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "fake_job_detector.joblib"
THRESHOLD_PATH = PROJECT_DIR / "models" / "model_threshold.json"

TEXT_FIELDS = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits",
]
CATEGORICAL_FIELDS = [
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
NUMERIC_FEATURES = [
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

URL_PATTERN = r"\b(?:https?://|www\.)\S+"
EMAIL_PATTERN = r"\b[\w.+%-]+@[\w.-]+\.[a-z]{2,}\b"
PHONE_PATTERN = r"(?<!\w)(?:\+?\d[\d().\s-]{5,}\d)(?!\w)"
SCAM_PATTERNS = {
    "contains_urgent_language": r"\b(?:urgent(?:ly)?|asap|immediately|hiring now|start immediately)\b",
    "contains_fee_request": r"\b(?:application|processing|registration|upfront|training)\s+fees?\b|\bpay\s+(?:an?\s+)?(?:application\s+)?fee\b|\bdeposit required\b",
    "contains_easy_money": r"\b(?:easy money|quick cash|earn money fast|get rich(?: quick)?|guaranteed income)\b",
    "contains_no_experience": r"\bno (?:previous )?experience (?:required|needed|necessary)\b",
    "contains_work_from_home": r"\bwork[- ]from[- ]home\b|\bremote work\b|\bwork remotely\b",
    "contains_cryptocurrency_payment": r"\b(?:bitcoin|crypto(?:currency)?)\b.{0,40}\b(?:pay(?:ment)?|paid|salary|wage|compensation)\b|\b(?:pay(?:ment)?|paid|salary|wage|compensation)\b.{0,40}\b(?:bitcoin|crypto(?:currency)?)\b",
    "contains_whatsapp_or_telegram": r"\bwhats\s?app\b|\btelegram\b",
}

st.set_page_config(page_title="Fake Job Posting Detector", layout="wide")


@st.cache_resource
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run "
            "`python src/fit_app_calibrator.py` from the project directory first."
        )
    model = joblib.load(MODEL_PATH)
    if not hasattr(model, "predict_proba"):
        raise TypeError(
            "The saved pipeline does not provide calibrated probabilities. "
            "Run `python src/fit_app_calibrator.py` to prepare the app model."
        )
    classes = np.asarray(getattr(model, "classes_", []))
    if not np.any(classes == 1):
        raise TypeError("The saved pipeline does not expose fraud class 1.")
    return model


@st.cache_resource
def load_threshold() -> float:
    if not THRESHOLD_PATH.is_file():
        raise FileNotFoundError(
            f"Threshold configuration not found at {THRESHOLD_PATH}. "
            "Run `notebooks/08_threshold_optimization.ipynb` first."
        )
    with THRESHOLD_PATH.open(encoding="utf-8") as file:
        threshold_config = json.load(file)
    if not isinstance(threshold_config, dict):
        raise ValueError(f"Threshold configuration in {THRESHOLD_PATH} must be a JSON object.")
    threshold = threshold_config.get("selected_threshold")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError(
            f"`selected_threshold` must be a number in {THRESHOLD_PATH}."
        )
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            f"`selected_threshold` must be between 0 and 1 in {THRESHOLD_PATH}."
        )
    return float(threshold)


def clean_text(text: str) -> str:
    """Match the deterministic text cleanup used to create the training text."""
    cleaned = text.lower()
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = re.sub(URL_PATTERN, " URL ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(EMAIL_PATTERN, " EMAIL ", cleaned, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", cleaned).strip()


def make_model_row(fields: dict[str, str | None]) -> tuple[pd.DataFrame, dict[str, int]]:
    raw_text = " ".join(fields[name] or "" for name in TEXT_FIELDS)
    combined_text = clean_text(raw_text)
    description = fields["description"] or ""
    requirements = fields["requirements"] or ""
    company_profile = fields["company_profile"] or ""

    engineered = {
        "raw__description_length": len(description),
        "raw__requirements_length": len(requirements),
        "raw__word_count": len(re.findall(r"\b\w+\b", combined_text)),
        "raw__character_count": len(combined_text),
        "raw__url_count": len(re.findall(URL_PATTERN, raw_text, flags=re.IGNORECASE)),
        "raw__email_count": len(re.findall(EMAIL_PATTERN, raw_text, flags=re.IGNORECASE)),
        "raw__phone_count": len(re.findall(PHONE_PATTERN, raw_text)),
        "raw__exclamation_count": raw_text.count("!"),
        "raw__missing_company_profile": int(not company_profile.strip()),
    }
    for feature_name, pattern in SCAM_PATTERNS.items():
        engineered[f"raw__{feature_name}"] = int(
            bool(re.search(pattern, combined_text, flags=re.IGNORECASE))
        )

    model_input = {
        "combined_text": combined_text,
        "location": fields["location"],
        "department": fields["department"],
        "salary_range": fields["salary"],
        "employment_type": fields["employment_type"],
        "required_experience": fields["required_experience"],
        "required_education": fields["required_education"],
        "industry": fields["industry"],
        "function": fields["function"],
        "telecommuting": fields["telecommuting"],
        "has_company_logo": fields["has_company_logo"],
        "has_questions": fields["has_questions"],
        **engineered,
    }
    return pd.DataFrame([model_input]), engineered


def show_warning_indicators(features: dict[str, int]) -> None:
    warnings = []
    labels = {
        "raw__missing_company_profile": "Company profile was not provided.",
        "raw__contains_urgent_language": "Urgent or immediate-action wording was found.",
        "raw__contains_fee_request": "Language requesting application, processing, or upfront fees was found.",
        "raw__contains_easy_money": "Easy-money or guaranteed-income wording was found.",
        "raw__contains_no_experience": "The posting says no experience is required; this can also be legitimate.",
        "raw__contains_work_from_home": "Work-from-home or remote-work wording was found; this is not suspicious by itself.",
        "raw__contains_cryptocurrency_payment": "Cryptocurrency was mentioned near payment or salary terms.",
        "raw__contains_whatsapp_or_telegram": "WhatsApp or Telegram was mentioned as a contact channel.",
    }
    for feature_name, message in labels.items():
        if features[feature_name]:
            warnings.append(message)
    if features["raw__url_count"]:
        warnings.append(f"{features['raw__url_count']} URL(s) were found; verify each domain independently.")
    if features["raw__email_count"]:
        warnings.append(f"{features['raw__email_count']} email address(es) were found.")
    if features["raw__phone_count"]:
        warnings.append(f"{features['raw__phone_count']} phone-like number sequence(s) were found.")
    if features["raw__exclamation_count"] >= 3:
        warnings.append(
            f"Many exclamation marks were found ({features['raw__exclamation_count']})."
        )

    st.subheader("Posting indicators")
    if warnings:
        for warning in warnings:
            st.warning(warning)
    else:
        st.success(
            "No configured warning indicators were detected. This does not prove the posting is safe."
        )


st.title("Fake Job Posting Detector")
st.markdown(
    "Review a job posting with a locally run machine-learning model. "
    "Your information stays on this machine and is not sent to an external AI service."
)
st.info(
    "The result is a model-based risk estimate, not a final decision. "
    "A low risk score does not guarantee that a posting is legitimate."
)

try:
    model = load_model()
    decision_threshold = load_threshold()
except (FileNotFoundError, TypeError, ValueError) as error:
    st.error(str(error))
    st.stop()

with st.form("job_posting_form"):
    st.subheader("Job posting")
    title = st.text_input("Job title", placeholder="For example, Customer Support Specialist")
    company_profile = st.text_area("Company profile", height=100)
    description = st.text_area("Job description", height=180)
    left, right = st.columns(2)
    with left:
        requirements = st.text_area("Requirements", height=130)
    with right:
        benefits = st.text_area("Benefits", height=130)

    st.subheader("Job details")
    left, right = st.columns(2)
    with left:
        location = st.text_input("Location")
        salary = st.text_input("Salary", placeholder="For example, 60000-75000 USD")
        employment_type = st.text_input("Employment type", placeholder="For example, Full-time")
        required_experience = st.text_input("Required experience")
        required_education = st.text_input("Required education")
    with right:
        industry = st.text_input("Industry")
        function = st.text_input("Function")
        st.caption("Optional model details")
        department = st.text_input("Department")
        telecommuting_choice = st.selectbox(
            "Remote / telecommuting", ["Not specified", "Yes", "No"]
        )
        has_company_logo_choice = st.selectbox(
            "Company logo shown in posting", ["Not specified", "Yes", "No"]
        )
        has_questions_choice = st.selectbox(
            "Application questions included", ["Not specified", "Yes", "No"]
        )

    submitted = st.form_submit_button("Check posting")

if submitted:
    if not any(
        value and value.strip()
        for value in (
            title,
            company_profile,
            description,
            requirements,
            benefits,
            location,
            salary,
            employment_type,
            required_experience,
            required_education,
            industry,
            function,
        )
    ):
        st.warning("Enter at least one job-posting detail before requesting an assessment.")
        st.stop()

    flag_values = {
        "Not specified": None,
        "Yes": "t",
        "No": "f",
    }
    fields = {
        "title": title,
        "company_profile": company_profile,
        "description": description,
        "requirements": requirements,
        "benefits": benefits,
        "location": location or None,
        "salary": salary or None,
        "employment_type": employment_type or None,
        "required_experience": required_experience or None,
        "required_education": required_education or None,
        "industry": industry or None,
        "function": function or None,
        "department": department or None,
        "telecommuting": flag_values[telecommuting_choice],
        "has_company_logo": flag_values[has_company_logo_choice],
        "has_questions": flag_values[has_questions_choice],
    }
    model_input, engineered_features = make_model_row(fields)
    probabilities = np.asarray(model.predict_proba(model_input), dtype=float)
    classes = np.asarray(model.classes_)
    fraud_class_indices = np.flatnonzero(classes == 1)
    if probabilities.ndim != 2 or probabilities.shape[0] != 1:
        raise ValueError(f"Expected one row of class probabilities; received shape {probabilities.shape}.")
    if len(fraud_class_indices) != 1 or probabilities.shape[1] != len(classes):
        raise ValueError("The saved model returned probabilities that do not match its classes.")
    fraud_probability = float(probabilities[0, fraud_class_indices[0]])
    if not np.isfinite(fraud_probability) or not 0.0 <= fraud_probability <= 1.0:
        raise ValueError(f"The model returned an invalid fraud probability: {fraud_probability}.")
    prediction = int(fraud_probability >= decision_threshold)

    st.divider()
    st.subheader("Assessment")
    risk_column, threshold_column = st.columns(2)
    with risk_column:
        st.metric("Estimated fraud risk", f"{fraud_probability:.1%}")
    with threshold_column:
        st.metric("Selected classification threshold", f"{decision_threshold:.0%}")
    st.progress(float(np.clip(fraud_probability, 0.0, 1.0)))
    if prediction == 1:
        st.error("Potentially fraudulent")
    else:
        st.success("Likely legitimate")

    show_warning_indicators(engineered_features)

    st.subheader("Safety recommendations")
    st.markdown(
        "- Verify the employer and job opening using the company’s official website or a trusted contact.\n"
        "- Do not pay application, processing, training, or equipment fees to obtain a job.\n"
        "- Avoid sharing bank details, identity documents, or passwords before verifying the employer and need.\n"
        "- Check website and email domains carefully; do not rely on links supplied in a suspicious message.\n"
        "- Keep communication on official hiring channels and be cautious about pressure to move to messaging apps.\n"
        "- Treat this score as one signal only; a low score cannot guarantee that a posting is genuine."
    )

st.divider()
st.caption(
    "Disclaimer: This academic tool provides a model-based risk estimate, not a final decision or "
    "verification of an employer. It may miss fraudulent postings or flag legitimate ones. "
    "Always verify the employer independently."
)
