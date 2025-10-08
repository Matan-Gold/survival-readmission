# Survival Analysis of 30-Day Hospital Readmission (MIMIC-IV Demo)

This repository implements a **time-to-event (survival) modeling** workflow to predict **30-day readmission after hospital discharge**. We prototype on the **MIMIC-IV Clinical Database Demo** (same schema as full MIMIC-IV), then scale to the full dataset once access is granted.

## Goals

- Frame readmission as **time-to-event** with proper **right-censoring**.
- Compare an **interpretable baseline** (Cox Proportional Hazards) with **nonlinear survival boosting** (XGBoost Survival: Cox and AFT objectives).
- Report **C-index**, **time-dependent AUC**, **Integrated Brier Score (IBS)**, and **calibration**.
- Provide **interpretability**: Cox hazard ratios, SHAP values for boosted model, Kaplan–Meier curves by risk strata.
- Ensure **reproducibility** with a clean project structure, tests, and CI.

## Project Structure

```
survival-readmission/
│
├── app/                      # Core application code
│   ├── data_loader.py        # Load MIMIC-IV data
│   ├── feature_engineering.py # Preprocessing & cohort definition
│   ├── evaluation.py         # Metrics (C-index, td-AUC, Brier)
│   ├── interpret.py          # SHAP, hazard ratios, survival curves
│   └── utils.py              # Shared utilities
│
├── models/                   # Saved trained models
├── archived_experiments/     # Old runs and checkpoints
├── data/                     # Datasets (tracked externally)
│   ├── raw/                  # Raw MIMIC-IV data (not committed)
│   ├── processed/            # Preprocessed survival dataset
│   └── sample/               # Tiny demo data for CI/smoke tests
│
├── results/                  # Outputs
│   ├── predictions/          # Survival risk scores
│   ├── figures/              # Kaplan-Meier, SHAP plots
│   └── metrics/              # Performance metrics
│
├── notebooks/                # Jupyter notebooks for exploration
│   ├── 01_EDA.ipynb
│   ├── 02_CohortDefinition.ipynb
│   ├── 03_Modeling.ipynb
│   ├── 04_Explainability.ipynb
│   └── 99_Report.ipynb
│
├── tests/                    # Unit tests
│
├── preprocess.py             # Data preprocessing script
├── train.py                  # Model training script
├── predict.py                # Prediction script
├── result.py                 # Results analysis script
└── tasks.py                  # Task automation (invoke)
```

## Workflow

### 1. **Preprocess** (`preprocess.py`)
- Load MIMIC-IV Demo (`hosp/admissions.csv.gz`, `hosp/patients.csv.gz`, …).
- Define the cohort: index discharge → next admission; **event=1** if readmitted ≤30 days; **censor=0** otherwise.
- Produce a processed dataset with `time_to_event` (days) and `event` (0/1), plus features:
  - Demographics: age, sex
  - Clinical: length of stay, comorbidity signals
  - Labs/vitals: selected values if available
- Output: `data/processed/cohort.csv`

### 2. **Train** (`train.py`)
- Fit **CoxPH** (with elastic-net regularization) and **XGBoost Survival** (Cox and AFT objectives).
- Persist trained models under `models/`.
- Example:
  ```bash
  python train.py --model coxph --data data/processed/cohort.csv
  python train.py --model xgboost --objective cox
  python train.py --model xgboost --objective aft
  ```

### 3. **Predict** (`predict.py`)
- Generate risk scores and/or survival functions for a validation/test split.
- Save outputs under `results/predictions/`.
- Example:
  ```bash
  python predict.py --model models/coxph_model.pkl --data data/processed/test.csv
  ```

### 4. **Analyze Results** (`result.py`)
- Compute **C-index**, **td-AUC**, **IBS**, **calibration**.
- Create **KM curves** (low vs high risk) and **SHAP** plots.
- Save figures to `results/figures/` and summary tables to `results/metrics/`.
- Example:
  ```bash
  python result.py --predictions results/predictions/test_pred.csv
  ```

## Data Handling

### MIMIC-IV Demo Access

#### Option 1: Automatic Download (Recommended)
The MIMIC-IV Demo dataset is publicly available and can be downloaded automatically:

```bash
python -m app.download_demo --dest data/raw/mimic-iv-demo
```

This will download and verify:
- `hosp/admissions.csv.gz`
- `hosp/patients.csv.gz`
- `hosp/diagnoses_icd.csv.gz` (for comorbidity scoring)
- `hosp/d_labitems.csv.gz` (lab item definitions)

#### Option 2: Manual Download
1. Visit [PhysioNet MIMIC-IV Demo](https://physionet.org/content/mimic-iv-demo/)
2. Download required tables manually
3. Place files in `data/raw/mimic-iv-demo/`

### Storage
- Keep the **MIMIC-IV Demo** data **outside the repo** or manage via **DVC**.
- Use an environment variable (e.g., `MIMIC_DEMO_DIR=/path/to/mimic-iv-demo`) in loader code.
- Place only **tiny samples** under `data/sample/` for CI.

### Environment Variable Setup
```bash
# Linux/Mac
export MIMIC_DEMO_DIR=/path/to/mimic-iv-demo

# Windows
set MIMIC_DEMO_DIR=C:\path\to\mimic-iv-demo
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd survival-readmission
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Reproducibility (Quick Start)

```bash
# Set up environment
git clone <repo_url>
cd survival-readmission
pip install -r requirements.txt

# Download MIMIC-IV Demo data
python -m app.download_demo --dest data/raw/mimic-iv-demo

# Run pipeline
python preprocess.py
python train.py --model coxph
python train.py --model xgboost --objective cox
python predict.py --model models/coxph_model.pkl
python result.py --predictions results/predictions/test_pred.csv
```

### Using Make/Invoke
```bash
make install
make preprocess
make train
make predict
make results

# Or using invoke
invoke preprocess
invoke train --model coxph
invoke predict --model coxph
invoke test
```

## Metrics & Evaluation

### Expected Performance (Placeholder)

| Model | C-index | td-AUC@30d | IBS | Notes |
|-------|---------|------------|-----|-------|
| CoxPH (elastic-net) | 0.68 | 0.66 | 0.18 | Interpretable baseline |
| XGB Survival (Cox) | 0.72 | 0.70 | 0.16 | Nonlinear hazard model |
| XGB Survival (AFT) | 0.74 | 0.72 | 0.15 | Direct time-to-event model |

### Evaluation Metrics

- **Harrell's C-index**: Concordance between predicted risk and actual survival times
- **Time-dependent AUC**: Classification performance at specific time points (e.g., 30 days)
- **Integrated Brier Score (IBS)**: Overall prediction accuracy across time
- **Calibration**: Agreement between predicted and observed event probabilities

### Interpretability

- **Cox Hazard Ratios**: Effect size and direction for each feature
- **SHAP Values**: Feature importance for XGBoost models
- **Kaplan-Meier Curves**: Survival curves stratified by risk groups (low/medium/high)
- **Feature Importance**: Ranking of most predictive features

## Cohort Definition

### Inclusion Criteria
- Adult patients (age ≥ 18) discharged from hospital
- Alive at discharge
- At least one prior admission in the database

### Index Event
- Hospital discharge date

### Outcome (Event)
- **Event = 1**: Readmitted within 30 days of discharge
- **Event = 0**: Not readmitted within 30 days (censored)

### Censoring Rules
- Administrative censoring at 30 days
- Death before readmission (treat as competing risk or censor)
- End of database follow-up

### Exclusions
- In-hospital deaths
- Transfers to other acute care facilities
- Discharges against medical advice (optional)

## Testing

Run unit tests:
```bash
pytest tests/ -v
```

Run with coverage:
```bash
pytest tests/ -v --cov=app --cov-report=html
```

Run smoke tests (CI):
```bash
pytest tests/test_end_to_end.py -v
```

## Limitations

- **Demo dataset is small**: The full MIMIC-IV is needed for robust estimates and clinical validation.
- **Incomplete follow-up**: Readmissions outside the source health system are not visible (treated as censoring).
- **Not a clinical decision tool**: Predictive models are for research purposes only and should not be used for clinical decisions without proper validation.
- **Selection bias**: Cohort definition may introduce selection bias.
- **Feature availability**: Limited to features available in MIMIC-IV Demo.
- **Temporal validation**: Models should be validated on future time periods for deployment.

## Licensing & Compliance

### Data License
- Follow the [PhysioNet Data Use Agreement](https://physionet.org/content/mimic-iv-demo/) and do not attempt re-identification.
- MIMIC-IV data is for research purposes only.
- Cite the MIMIC-IV publication in any resulting work.

### Code License
MIT License - see [LICENSE](LICENSE) file for details.

## References

### MIMIC-IV
- Johnson, A., Bulgarelli, L., Pollard, T., Horng, S., Celi, L. A., & Mark, R. (2023). MIMIC-IV (version 2.2). PhysioNet. https://doi.org/10.13026/6mm1-ek67
- Goldberger, A., et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation, 101(23), e215-e220.

### Survival Analysis Methods
- Cox, D. R. (1972). Regression models and life-tables. Journal of the Royal Statistical Society: Series B (Methodological), 34(2), 187-202.
- Harrell Jr, F. E., et al. (1982). Evaluating the yield of medical tests. JAMA, 247(18), 2543-2546.
- Graf, E., et al. (1999). Assessment and comparison of prognostic classification schemes for survival data. Statistics in Medicine, 18(17‐18), 2529-2545.

### XGBoost Survival
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. KDD '16.
- XGBoost Survival: https://xgboost.readthedocs.io/en/stable/tutorials/aft_survival_analysis.html

### Interpretability
- Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. NeurIPS.

## Contributing

Please follow the project conventions outlined in `.cursorrules`. Key points:

- Write docstrings for all public functions
- Use type hints where appropriate
- Write tests for new functionality
- Keep functions small and single-purpose
- Respect censoring in survival analysis
- Use clear commit messages and descriptive branch names
- Document any changes to cohort definitions

## Contact

For questions or issues, please open a GitHub issue or contact the maintainers.

---

**Note**: This is a research project using the MIMIC-IV Demo dataset. Results should be validated on the full MIMIC-IV dataset before drawing clinical conclusions.
