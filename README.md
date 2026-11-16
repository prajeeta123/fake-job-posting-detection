# Fake Job Posting Detection System

A hybrid machine learning and NLP-based system that detects potentially fraudulent online job postings using TF-IDF text features, job metadata, scam-indicator features, and Streamlit.

## 1. Project overview

This academic project combines posting text, structured job details, and manually engineered indicators to estimate whether an online job posting resembles fraudulent examples in its training data. The Streamlit application runs the saved model locally.

## 2. Problem statement

Online recruitment scams can imitate legitimate job advertisements. Reviewing every posting manually is difficult, while simple keyword rules can miss new wording or flag harmless phrases. This project explores a hybrid model that combines text and structured signals to support an initial review.

## 3. Objectives

- Inspect and preprocess job-posting data.
- Compare text-only baseline classifiers with models using text, metadata, and engineered indicators.
- Evaluate results with metrics suited to an imbalanced binary classification problem.
- Provide a local interface that presents a risk estimate and detected indicators.
- Use model explanations to explore learned feature associations.

## 4. Dataset description

The project includes `data/raw/DataSet.zip`, containing the dataset of 17,880 postings and 18 columns. It contains five text fields (`title`, `company_profile`, `description`, `requirements`, and `benefits`), job metadata, three binary posting fields, the `fraudulent` target (`t` for fraudulent and `f` for legitimate), and `in_balanced_dataset`. The latter is excluded from model features. The bundled copy replaces one email-shaped value in a job title with `redacted@example.invalid`; preprocessing still converts that value to the same `EMAIL` token.

The inspection found 17,014 legitimate and 866 fraudulent raw rows. Removing 235 exact duplicate rows leaves 17,645 rows: 16,787 legitimate and 858 fraudulent. Optional fields have missing values; the fitted pipeline handles them. The source CSV is bundled in a compressed ZIP; the 83.9 MB processed CSV is stored with Git LFS. The dataset source is listed as CC0 on [Kaggle](https://www.kaggle.com/datasets/amruthjithrajvr/recruitment-scam). See [the dataset report](report/06_dataset_description.md) and [the cited dataset paper](https://doi.org/10.3390/fi9010006) for context and attribution.

## 5. Technologies used

- Python, pandas, NumPy, SciPy
- scikit-learn and XGBoost
- TF-IDF, one-hot encoded metadata, and scaled engineered features
- Matplotlib, Seaborn, and SHAP
- Joblib, Jupyter, IPython, and Streamlit

## 6. Project workflow

1. `notebooks/01_dataset_inspection.ipynb` inspects the dataset and label distribution.
2. `notebooks/02_preprocessing_and_features.ipynb` cleans text and creates engineered features.
3. `notebooks/03_baseline_models.ipynb` compares text-only classifiers.
4. `notebooks/04_hybrid_model.ipynb` compares hybrid classifiers and selects a model using training-only validation.
5. `notebooks/06_model_verification.ipynb` evaluates the saved model on the held-out test partition.
6. `notebooks/07_error_analysis.ipynb` reviews errors; `notebooks/08_threshold_optimization.ipynb` selects a threshold on training-only validation; and `notebooks/09_explainability.ipynb` explores SHAP explanations.
7. `src/fit_app_calibrator.py` fits the calibrated app pipeline using training data only.
8. `app.py` accepts a posting, creates its input features, and displays the model-based estimate.

The held-out test partition is reserved for final evaluation. It is not used for model selection, calibration, or threshold selection.

## 7. System architecture

```text
Posting text + job metadata
            |
            v
Text cleanup + scam-indicator feature creation
            |
            v
Saved pipeline: TF-IDF + categorical preprocessing + numeric preprocessing
            |
            v
Calibrated classifier -> estimated fraud risk -> selected threshold -> Streamlit result
```

The app loads `models/fake_job_detector.joblib` and `models/model_threshold.json`. Text normalization and deterministic feature creation happen in the app; fitted transformations and classification happen in the saved pipeline.

## 8. Folder structure

```text
fake-job-detector/
├── app.py
├── data/
│   ├── raw/
│   │   └── DataSet.zip      # Sanitized source dataset archive
│   └── processed/
│       └── processed_job_postings.csv  # Stored with Git LFS
├── models/                  # Saved models and JSON metadata/configuration
├── notebooks/               # Inspection, preprocessing, modeling, and analysis
├── report/                  # Project chapters, metrics, charts, and analysis
├── src/                     # Model calibration script
├── tests/                   # Streamlit input scenarios
├── .gitattributes           # Git LFS rule for the processed dataset
├── .gitignore
├── README.md
└── requirements.txt
```

## 9. Installation instructions

Create a GitHub repository named `fake-job-posting-detection`, then replace `<repository-url>` below with its HTTPS URL.

### Windows Command Prompt

```powershell
git lfs install
git clone <repository-url>
cd fake-job-posting-detection
git lfs pull
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

In Windows PowerShell, activate the environment with `.\venv\Scripts\Activate.ps1`.

### Linux/macOS

```bash
git lfs install
git clone <repository-url>
cd fake-job-posting-detection
git lfs pull
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Install [Git LFS](https://git-lfs.com/) first if the `git lfs` command is unavailable. `git lfs pull` downloads the processed CSV tracked outside ordinary Git history.

The saved model was verified in the current Python 3.14 environment. Use a recent Python version supported by the listed packages. Loading scikit-learn models across incompatible scikit-learn versions is not guaranteed; if deserialization fails, recreate the environment with a compatible scikit-learn version or regenerate the model from the notebooks.

## 10. How to run the Streamlit application

From the repository root, with the virtual environment active:

```bash
streamlit run app.py
```

The app needs `models/fake_job_detector.joblib` and `models/model_threshold.json`. Enter at least one posting detail and select **Check posting**. The app displays an estimated fraud risk, the threshold-based label, configured warning indicators, and safety suggestions.

## 11. How to reproduce the model

The source dataset archive is included at `data/raw/DataSet.zip`; the processed CSV is downloaded by Git LFS. In a fresh clone, extract the archive into `data/raw` (PowerShell: `Expand-Archive -LiteralPath data/raw/DataSet.zip -DestinationPath data/raw`; Linux/macOS: `unzip data/raw/DataSet.zip -d data/raw`). Then run the notebooks in workflow order. The preprocessing notebook can regenerate the processed data. The saved hybrid classifier and calibration script also need the processed CSV and source CSV to align posting metadata.

Run `python src/fit_app_calibrator.py` after the selected hybrid model and both datasets are available. Generate or validate the threshold configuration with `notebooks/08_threshold_optimization.ipynb`. Keep the held-out test data out of training, calibration, and threshold selection.

## 12. Model evaluation metrics

The project reports accuracy, fraud precision, fraud recall, fraud F1, ROC-AUC, PR-AUC, and confusion matrices. PR-AUC and class-specific precision/recall are useful alongside accuracy because fraudulent examples are a small share of this dataset.

## 13. Results summary

On the held-out test split (3,529 postings, including 172 fraudulent examples), the validation-selected hybrid Linear SVM candidate reported 98.19% accuracy, 80.68% fraud precision, 82.56% fraud recall, 81.61% fraud F1, 98.56% ROC-AUC, and 89.58% PR-AUC. The saved calibrated model's separate default-prediction verification reported 86.62% fraud precision, 79.07% fraud recall, 82.67% fraud F1, 98.84% ROC-AUC, and 89.81% PR-AUC.

The app threshold of 0.40 was selected using training-only validation, where fraud F1 was 0.844. The reported saved-model test metrics use its default prediction rule, not the app's 0.40 threshold; they should not be read as test results for the app threshold. Full details are in [the results report](report/10_results_and_discussion.md).

## 14. Explainability

The SHAP notebook explains sampled model outputs and summarizes feature contributions. For the calibrated Linear SVM, the reported SHAP values describe contributions to the underlying SVM decision margin, not percentage points of calibrated probability. Feature associations are not causes and do not establish that an individual posting is fraudulent or legitimate. See [the explainability report](report/explainability_report.md).

## 15. Limitations

- The data is imbalanced and may not represent current postings, all locations, industries, languages, or hiring practices.
- False positives and false negatives are possible.
- Handwritten indicators are pattern-based and can miss variants or flag harmless context. Remote work and entry-level language are not inherently suspicious.
- A model explanation describes learned associations, not proof or causality.
- Dataset metrics describe the documented split and are not a guarantee of performance on new postings.

## 16. Future scope

- Evaluate on newer, independently sourced postings and monitor performance over time.
- Assess performance across relevant industries, locations, and groups.
- Review indicator patterns with human feedback and expand language coverage.
- Document model versions, privacy-conscious feedback, and retraining procedures.

## 17. Disclaimer

**The prediction is a model-based risk estimate and not a final decision.** It does not verify an employer or prove that a posting is fraudulent or legitimate. Independently confirm job openings through trusted employer channels. Do not pay fees to obtain a job, and protect personal, identity, and financial information.

## 18. References

- Vidros, S., Kolias, C., Kambourakis, G., & Akoglu, L. (2017). “Automatic detection of online recruitment frauds: Characteristics, methods, and a public dataset.” *Future Internet*, 9(1), 6. [https://doi.org/10.3390/fi9010006](https://doi.org/10.3390/fi9010006).
- Additional literature and project-specific sources are listed in [report/13_references.md](report/13_references.md).
- Project metrics and dataset details are documented in the files under `report/`.

## 19. Author information

- **Author:** [Prajeeta Pathare]
- **Institution / department:** [MITAOE]
- **Course / supervisor:** [Mrs.Gauri Thithe]
  

