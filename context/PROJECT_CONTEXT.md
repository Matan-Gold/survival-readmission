# Project Context: Survival Analysis for 30-Day Hospital Readmission

## Overview

This project implements a survival analysis (time-to-event) modeling pipeline to predict 30-day hospital readmission using the MIMIC-IV Clinical Database Demo. The goal is to develop interpretable and accurate models that properly handle censoring and provide actionable insights for clinical decision support research.

## Problem Framing

### Clinical Question
**What is the risk and timing of hospital readmission within 30 days of discharge?**

### Why Survival Analysis?
- **Time-to-event**: We care about *when* readmission occurs, not just *if* it occurs
- **Right-censoring**: Not all patients are readmitted within the observation window
- **Proper handling**: Unlike classification, survival models correctly handle censored observations
- **Risk stratification**: Can identify high-risk patients for targeted interventions

## Cohort Definition

### Index Event
- **Hospital discharge** from an acute care stay

### Inclusion Criteria
1. Adult patients (age ≥ 18 years)
2. Alive at discharge
3. At least one prior admission in database (for baseline features)
4. Complete data for key features (age, sex, length of stay)

### Outcome Definition
- **Event (event=1)**: Hospital readmission within 30 days of discharge
- **Censored (event=0)**: No readmission within 30 days

### Censoring Rules
1. **Administrative censoring**: 30-day follow-up window
2. **Death before readmission**: Can be treated as:
   - Censoring (assuming readmission is the only outcome of interest)
   - Competing risk (if death is considered informative)
3. **End of database**: Patients with <30 days follow-up in database
4. **Loss to follow-up**: Readmissions outside the source health system

### Exclusions
1. In-hospital deaths (no opportunity for readmission)
2. Transfers to other acute care facilities (not true discharge)
3. Discharges against medical advice (optional, depending on research question)
4. Planned readmissions (e.g., staged procedures) - optional

## Feature Engineering

### Demographics
- Age at discharge
- Sex/gender
- Race/ethnicity (if available and appropriate)

### Clinical Features
- **Length of stay** (index admission)
- **Admission type** (emergency, urgent, elective)
- **Discharge location** (home, skilled nursing, etc.)
- **Number of prior admissions** (lookback: 6-12 months)

### Comorbidities
- **Elixhauser Comorbidity Index** or **Charlson Comorbidity Index**
- Individual comorbidity flags (heart failure, COPD, diabetes, etc.)

### Lab Values (if available)
- Last available values before discharge
- Creatinine, hemoglobin, white blood cell count, etc.
- Abnormal flag (outside normal range)

### Vital Signs (if available)
- Last recorded before discharge
- Heart rate, blood pressure, temperature, respiratory rate

### Medications (future work)
- Number of discharge medications
- High-risk medication flags

## Modeling Strategy

### Baseline: Cox Proportional Hazards (CoxPH)
**Why?**
- Gold standard for survival analysis
- Interpretable hazard ratios
- Handles censoring naturally
- Well-understood assumptions

**Implementation:**
- Elastic-net regularization (L1+L2) to handle multicollinearity
- Penalizer tuning via cross-validation
- Check proportional hazards assumption

**Outputs:**
- Hazard ratios with 95% CI
- Partial hazard (risk score) per patient
- Baseline survival function

### Advanced: XGBoost Survival

#### XGBoost Cox Objective
**Why?**
- Captures nonlinear relationships and interactions
- Handles missing values naturally
- Often better predictive performance than CoxPH

**Implementation:**
- `objective='survival:cox'`
- Hyperparameter tuning (max_depth, learning_rate, n_estimators)
- Cross-validation for early stopping

**Outputs:**
- Risk scores
- Feature importance
- SHAP values for interpretability

#### XGBoost AFT (Accelerated Failure Time)
**Why?**
- Directly models survival time (not just hazard)
- Flexible parametric approach
- Can generate survival time predictions

**Implementation:**
- `objective='survival:aft'`
- AFT distribution (normal, logistic, extreme)
- Hyperparameter tuning

**Outputs:**
- Predicted survival times
- Survival functions
- Feature importance

## Evaluation Metrics

### Discrimination
1. **Harrell's C-index** (concordance index)
   - Probability that model ranks pairs correctly
   - Range: 0.5 (random) to 1.0 (perfect)
   - Target: >0.65 for useful model

2. **Time-dependent AUC**
   - ROC curve at specific time point (e.g., 30 days)
   - Accounts for censoring
   - Evaluates classification performance at fixed horizon

### Calibration
1. **Integrated Brier Score (IBS)**
   - Mean squared error between predicted and observed survival
   - Lower is better
   - Accounts for both discrimination and calibration

2. **Calibration plots**
   - Predicted vs observed event probabilities
   - Stratify by risk deciles
   - Assess over/under-prediction

### Clinical Utility
1. **Net benefit curves** (decision curve analysis)
2. **Risk stratification**: Low/medium/high risk groups
3. **Kaplan-Meier curves** by risk strata

## Interpretability

### Cox Model
- **Hazard ratios**: exp(β) interpretation
  - HR > 1: increased risk
  - HR < 1: decreased risk
- **Confidence intervals**: statistical significance
- **Feature ranking**: by |log(HR)|

### XGBoost Models
- **SHAP values**: 
  - Summary plots (global importance)
  - Waterfall plots (individual predictions)
  - Dependence plots (feature interactions)
- **Feature importance**: gain, cover, frequency
- **Partial dependence plots**: marginal effects

### Survival Curves
- **Kaplan-Meier curves**: non-parametric survival estimates
  - Stratified by risk groups (tertiles or quartiles)
  - Log-rank test for differences
- **Individual survival functions**: patient-specific predictions

## Data Management

### MIMIC-IV Demo
- **Source**: https://physionet.org/content/mimic-iv-demo/
- **Size**: ~100 patients (small sample of full MIMIC-IV)
- **Purpose**: Development and prototyping
- **Tables needed**:
  - `hosp/admissions.csv.gz`
  - `hosp/patients.csv.gz`
  - `hosp/diagnoses_icd.csv.gz`
  - `hosp/labevents.csv.gz` (optional)

### Data Storage
- **Environment variable**: `MIMIC_DEMO_DIR`
- **Not committed**: Keep data outside repo
- **DVC option**: For larger datasets and version control

### Sample Data for CI
- **Location**: `data/sample/`
- **Size**: 10-20 synthetic/anonymized examples
- **Purpose**: Smoke tests, CI/CD validation

## Reproducibility

### Version Control
- Git for code
- DVC for data (optional but recommended)
- Model versioning in `models/` directory

### Environment
- `requirements.txt` with pinned versions
- Python 3.8+ recommended
- Key dependencies:
  - `lifelines` (Cox PH)
  - `xgboost` (with survival support)
  - `scikit-survival` (evaluation metrics)
  - `shap` (interpretability)

### Random Seeds
- Set seeds for train/test splits
- Set seeds for model training (where applicable)
- Document in notebooks and scripts

### Documentation
- Cohort definition in `notebooks/02_CohortDefinition.ipynb`
- Model results in `notebooks/99_Report.ipynb`
- README.md with quick start guide

## Limitations & Future Work

### Current Limitations
1. **Demo dataset size**: Too small for robust estimates
2. **Feature availability**: Limited to what's in MIMIC-IV
3. **External validity**: Single health system data
4. **Competing risks**: Death not modeled as competing risk
5. **Time-varying covariates**: Not currently handled

### Future Enhancements
1. **Full MIMIC-IV**: Scale to complete dataset
2. **Deep learning**: Survival neural networks (DeepSurv, DeepHit)
3. **Time-varying features**: Longitudinal measurements
4. **Competing risks**: Fine-Gray models
5. **External validation**: Test on other EHR databases
6. **Clinical deployment**: Real-time risk scoring system

## Regulatory & Ethical Considerations

### Data Privacy
- Follow PhysioNet Data Use Agreement
- No re-identification attempts
- HIPAA-compliant handling

### Research Ethics
- Models are for research only
- Not intended for clinical decision-making without validation
- Bias assessment (by demographic groups)
- Fairness metrics (equalized odds, calibration by subgroup)

### Limitations Disclosure
- Clear communication of model limitations
- Uncertainty quantification
- Clinical context required for interpretation

## Key References

### Survival Analysis
- Harrell FE. *Regression Modeling Strategies* (2015)
- Therneau TM, Grambsch PM. *Modeling Survival Data* (2000)

### Machine Learning for Survival
- Katzman et al. "DeepSurv" (2018)
- Lee et al. "DeepHit" (2018)

### Clinical Readmission
- Kansagara et al. "Risk Prediction Models for Hospital Readmission" (2011)
- LACE Index (Length of stay, Acuity, Comorbidities, ED visits)

### MIMIC-IV
- Johnson et al. "MIMIC-IV" (2023)
- PhysioNet: https://physionet.org/

---

**Last Updated**: 2025-10-08
**Project Status**: Initial setup and development
**Next Steps**: Implement data preprocessing and cohort definition

