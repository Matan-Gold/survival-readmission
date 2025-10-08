# Notebooks Guide

This directory contains Jupyter notebooks for the survival analysis project.

## Notebook Overview

### 01_EDA.ipynb ✅ READY
**Exploratory Data Analysis**

A comprehensive exploration of the MIMIC-IV Demo dataset including:

1. **Setup & Data Loading**
   - Loads admissions, patients, diagnoses_icd, and d_labitems
   - Sets up visualization styling

2. **Admissions Analysis**
   - Data structure and statistics
   - Admission types and locations
   - Length of stay distributions
   - Discharge locations

3. **Patient Demographics**
   - Gender distribution
   - Age at admission
   - Demographic visualizations

4. **Readmission Analysis** (Key for Survival Modeling!)
   - Computes time to next admission
   - Flags 30-day readmissions
   - Readmission rates by patient characteristics
   - Time-to-event distributions

5. **Diagnosis Data**
   - ICD code distributions
   - Most common diagnoses
   - Diagnoses per admission

6. **Data Quality**
   - Missing value assessment
   - Data completeness checks

7. **Key Insights & Next Steps**
   - Summary of findings
   - Recommendations for cohort definition
   - Next steps for survival analysis

**Output:** Saves processed data to `data/processed/admissions_with_readmit.csv`

---

### 02_CohortDefinition.ipynb (TODO)
**Cohort Definition for Survival Analysis**

Will define:
- Inclusion/exclusion criteria
- Index event (discharge)
- Outcome (30-day readmission)
- Censoring rules
- Final cohort for modeling

---

### 03_Modeling.ipynb (TODO)
**Survival Model Training**

Will implement:
- Cox Proportional Hazards
- XGBoost Survival (Cox objective)
- XGBoost Survival (AFT objective)
- Model comparison

---

### 04_Explainability.ipynb (TODO)
**Model Interpretation**

Will generate:
- SHAP values
- Hazard ratios
- Feature importance
- Kaplan-Meier curves by risk strata

---

### 99_Report.ipynb (TODO)
**Final Report**

Will include:
- Executive summary
- Key findings
- Model performance
- Clinical implications
- Limitations

---

## Usage

### Running the EDA Notebook

1. **Ensure data is downloaded:**
   ```bash
   python -m app.download_demo --dest data/raw/mimic-iv-demo
   ```

2. **Launch Jupyter:**
   ```bash
   jupyter notebook
   ```

3. **Open `01_EDA.ipynb` and run all cells**

4. **Review the outputs:**
   - Summary statistics
   - Visualizations
   - Readmission patterns
   - Processed data saved to `data/processed/`

### Expected Runtime
- EDA notebook: ~1-2 minutes (demo dataset)

### Dependencies
All required packages are in `requirements.txt`:
- pandas, numpy, matplotlib, seaborn
- pathlib (standard library)

---

## Notes

- **Demo Dataset:** Small sample for prototyping
- **Full MIMIC-IV:** Will require more computational resources
- **Notebooks are for exploration:** Core logic should be in `app/` modules
- **Save intermediate outputs:** Each notebook saves processed data for the next

---

**Last Updated:** 2025-10-08

