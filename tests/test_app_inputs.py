"""Exercise the Streamlit prediction flow with representative job postings."""

import json
from pathlib import Path

from streamlit.testing.v1 import AppTest


PROJECT_DIR = Path(__file__).resolve().parents[1]
APP_PATH = PROJECT_DIR / "app.py"
THRESHOLD_PATH = PROJECT_DIR / "models" / "model_threshold.json"

TEST_CASES = [
    {
        "name": "Realistic legitimate software role",
        "fields": {
            "Job title": "Software Engineer",
            "Location": "US, CA, San Francisco",
            "Salary": "90000-120000 USD",
            "Employment type": "Full-time",
            "Required experience": "Mid-Senior level",
            "Required education": "Bachelor's Degree",
            "Industry": "Computer Software",
            "Function": "Engineering",
            "Department": "Engineering",
        },
        "text": {
            "Company profile": "Established software company building business applications.",
            "Job description": (
                "We are hiring a software engineer to develop and maintain web "
                "applications, write clean code, review changes, and support releases."
            ),
            "Requirements": (
                "Bachelor's degree in Computer Science or equivalent experience. "
                "Experience with JavaScript, Python, or Java is preferred."
            ),
            "Benefits": "Health insurance, paid leave, and professional development.",
        },
        "expected_indicators": [],
        "reasonable": lambda label, probability, threshold, indicators: (
            label == "Likely legitimate" and probability < threshold
        ),
    },
    {
        "name": "Fee request and urgent-language posting",
        "fields": {
            "Job title": "Urgent Remote Data Entry Assistant",
            "Location": "",
            "Salary": "Earn money quickly",
            "Employment type": "Part-time",
            "Required experience": "Entry level",
            "Required education": "",
            "Industry": "",
            "Function": "",
            "Department": "",
        },
        "text": {
            "Company profile": "Online hiring agency.",
            "Job description": (
                "Urgent hiring now! Pay a registration fee before you can start. "
                "Work from home and earn money quickly."
            ),
            "Requirements": "No experience required.",
            "Benefits": "Immediate start and guaranteed income.",
        },
        "expected_indicators": [
            "Language requesting application, processing, or upfront fees",
            "Urgent or immediate-action wording",
        ],
        "reasonable": lambda label, probability, threshold, indicators: (
            label == "Potentially fraudulent" and probability >= threshold
        ),
    },
    {
        "name": "Missing company profile",
        "fields": {
            "Job title": "Office Administrator",
            "Location": "US, IL, Chicago",
            "Salary": "",
            "Employment type": "Full-time",
            "Required experience": "Associate",
            "Required education": "",
            "Industry": "Accounting",
            "Function": "Administrative",
            "Department": "Operations",
        },
        "text": {
            "Company profile": "",
            "Job description": (
                "The office administrator will schedule meetings, maintain records, "
                "and provide support to the operations team."
            ),
            "Requirements": "Strong communication and organizational skills.",
            "Benefits": "Paid leave and health coverage.",
        },
        "expected_indicators": ["Company profile was not provided."],
        "reasonable": lambda label, probability, threshold, indicators: (
            "Company profile was not provided." in indicators
        ),
    },
    {
        "name": "Posting containing many URLs",
        "fields": {
            "Job title": "Marketing Coordinator",
            "Location": "",
            "Salary": "",
            "Employment type": "Full-time",
            "Required experience": "Entry level",
            "Required education": "",
            "Industry": "Marketing and Advertising",
            "Function": "Marketing",
            "Department": "Marketing",
        },
        "text": {
            "Company profile": "Digital marketing team.",
            "Job description": (
                "Coordinate campaign materials and review these resources: "
                "https://example.com/jobs https://example.org/about "
                "https://example.net/careers https://example.edu/team"
            ),
            "Requirements": "Clear writing and organization.",
            "Benefits": "Health insurance and paid leave.",
        },
        "expected_indicators": ["4 URL(s) were found"],
        "reasonable": lambda label, probability, threshold, indicators: (
            any("4 URL(s) were found" in indicator for indicator in indicators)
        ),
    },
    {
        "name": "Unrealistic salary and guaranteed-income claims",
        "fields": {
            "Job title": "Work From Home Data Entry Clerk",
            "Location": "",
            "Salary": "$10,000 per week",
            "Employment type": "Part-time",
            "Required experience": "Entry level",
            "Required education": "",
            "Industry": "",
            "Function": "",
            "Department": "",
        },
        "text": {
            "Company profile": "",
            "Job description": (
                "Earn guaranteed income of $10,000 every week doing simple tasks "
                "from home. Apply immediately."
            ),
            "Requirements": "No experience required.",
            "Benefits": "Easy money and immediate start.",
        },
        "expected_indicators": [
            "Easy-money or guaranteed-income wording",
            "Urgent or immediate-action wording",
        ],
        "reasonable": lambda label, probability, threshold, indicators: (
            label == "Potentially fraudulent" and probability >= threshold
        ),
    },
]


def run_prediction(case: dict, threshold: float) -> dict:
    app = AppTest.from_file(str(APP_PATH), default_timeout=120).run()
    if app.exception:
        raise AssertionError(f"{case['name']}: app failed to load: {app.exception[0].message}")

    fields = case["fields"]
    text = case["text"]
    for widget in app.text_input:
        if widget.label in fields:
            widget.set_value(fields[widget.label])
    for widget in app.text_area:
        if widget.label in text:
            widget.set_value(text[widget.label])

    app.button[0].click().run()
    if app.exception:
        raise AssertionError(f"{case['name']}: prediction raised: {app.exception[0].message}")

    fraud_metrics = [metric for metric in app.metric if metric.label == "Estimated fraud risk"]
    if len(fraud_metrics) != 1:
        raise AssertionError(f"{case['name']}: expected one displayed fraud-risk metric.")
    probability = float(fraud_metrics[0].value.rstrip("%")) / 100

    labels = [element.value for element in app.error]
    labels.extend(element.value for element in app.success)
    predicted_labels = [
        label
        for label in labels
        if label in {"Likely legitimate", "Potentially fraudulent"}
    ]
    if len(predicted_labels) != 1:
        raise AssertionError(f"{case['name']}: expected one prediction label, got {predicted_labels}.")
    predicted_label = predicted_labels[0]
    indicators = [element.value for element in app.warning]

    if not 0 <= probability <= 1:
        raise AssertionError(f"{case['name']}: invalid probability {probability}.")
    if (predicted_label == "Potentially fraudulent") != (probability >= threshold):
        raise AssertionError(
            f"{case['name']}: label does not match probability and threshold {threshold:.2f}."
        )

    return {
        "label": predicted_label,
        "probability": probability,
        "indicators": indicators,
    }


def main() -> None:
    with THRESHOLD_PATH.open(encoding="utf-8") as file:
        threshold_config = json.load(file)
    threshold = threshold_config["selected_threshold"]

    print(f"Using saved app threshold: {threshold:.0%}")
    print("Predictions are generated locally by the saved pipeline; no model is modified.\n")
    failures = []
    for case in TEST_CASES:
        result = run_prediction(case, threshold)
        missing_indicators = [
            expected
            for expected in case["expected_indicators"]
            if not any(expected in detected for detected in result["indicators"])
        ]
        reasonable = bool(
            not missing_indicators
            and case["reasonable"](
                result["label"],
                result["probability"],
                threshold,
                result["indicators"],
            )
        )
        print(f"Case: {case['name']}")
        print(f"  Predicted label: {result['label']}")
        print(f"  Fraud probability: {result['probability']:.1%}")
        print(f"  Detected warning indicators: {result['indicators'] or ['None']}")
        print(f"  Prediction reasonable: {'Yes' if reasonable else 'No'}")
        if missing_indicators:
            print(f"  Expected indicators not detected: {missing_indicators}")
        print()
        if not reasonable:
            failures.append(case["name"])

    if failures:
        raise AssertionError(f"Unreasonable results or missing indicators: {failures}")
    print(f"All {len(TEST_CASES)} input scenarios passed.")


def test_streamlit_prediction_scenarios() -> None:
    main()


if __name__ == "__main__":
    main()
