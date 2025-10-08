# Technical Notes & Implementation Details

## Data Pipeline

### Preprocessing Steps

1. **Load Raw Data**
   ```python
   # From app/data_loader.py
   admissions = load_admissions()  # hosp/admissions.csv.gz
   patients = load_patients()      # hosp/patients.csv.gz
   ```

2. **Cohort Selection**
   ```python
   # From app/feature_engineering.py
   # - Filter to adults (age >= 18)
   # - Exclude in-hospital deaths
   # - Require complete discharge dates
   cohort = define_cohort(admissions, patients)
   ```

3. **Define Index Event & Outcome**
   ```python
   # For each index discharge:
   # - Look forward for next admission
   # - If readmit <= 30 days: event=1, time=days_to_readmit
   # - If no readmit or >30 days: event=0, time=30
   ```

4. **Feature Engineering**
   ```python
   # Compute features at index discharge:
   # - Demographics: age, sex
   # - LOS, admission_type, discharge_location
   # - Prior admissions (count in past 6/12 months)
   # - Comorbidities from ICD codes
   # - Labs/vitals (last value before discharge)
   ```

5. **Train/Test Split**
   ```python
   # Temporal split recommended:
   # - Train: earlier admissions
   # - Test: later admissions
   # Or random split with stratification by event
   ```

## Model Training

### Cox Proportional Hazards

```python
from lifelines import CoxPHFitter

# Initialize with elastic-net penalty
cph = CoxPHFitter(penalizer=0.1, l1_ratio=0.5)

# Fit model
cph.fit(
    df=train_data,
    duration_col='time_to_event',
    event_col='event',
    show_progress=True
)

# Get predictions
risk_scores = cph.predict_partial_hazard(test_data)
survival_functions = cph.predict_survival_function(test_data)
```

### XGBoost Survival (Cox)

```python
import xgboost as xgb

# Prepare data
dtrain = xgb.DMatrix(X_train)
dtrain.set_float_info('label', y_train['time_to_event'])
dtrain.set_uint_info('label_lower_bound', y_train['event'])

# Parameters
params = {
    'objective': 'survival:cox',
    'eval_metric': 'cox-nloglik',
    'max_depth': 3,
    'learning_rate': 0.1,
    'min_child_weight': 1,
}

# Train
model = xgb.train(
    params,
    dtrain,
    num_boost_round=100,
    evals=[(dval, 'validation')],
    early_stopping_rounds=10
)

# Predict
risk_scores = model.predict(dtest)
```

### XGBoost Survival (AFT)

```python
params = {
    'objective': 'survival:aft',
    'eval_metric': 'aft-nloglik',
    'aft_loss_distribution': 'normal',  # or 'logistic', 'extreme'
    'aft_loss_distribution_scale': 1.0,
}

# For AFT, set label and label_lower_bound differently
dtrain.set_float_info('label', y_train['time_to_event'])
# For censored: set label_lower_bound to 0
# For events: set label_lower_bound to time_to_event
lower_bound = np.where(y_train['event'] == 1, 
                       y_train['time_to_event'], 
                       0)
dtrain.set_float_info('label_lower_bound', lower_bound)
```

## Evaluation Implementation

### C-index

```python
from sksurv.metrics import concordance_index_censored

# Compute C-index
c_index = concordance_index_censored(
    event_indicator=y_test['event'].astype(bool),
    event_time=y_test['time_to_event'],
    estimate=risk_scores
)[0]
```

### Time-Dependent AUC

```python
from sksurv.metrics import cumulative_dynamic_auc

# Compute td-AUC at 30 days
times = np.array([30.0])
auc, mean_auc = cumulative_dynamic_auc(
    y_train=y_train_structured,
    y_test=y_test_structured,
    estimate=risk_scores,
    times=times
)
```

### Integrated Brier Score

```python
from sksurv.metrics import integrated_brier_score

# Compute IBS
times = np.linspace(0, 30, 31)
preds = np.column_stack([
    survival_function(t) for t in times
])

ibs = integrated_brier_score(
    survival_train=y_train_structured,
    survival_test=y_test_structured,
    estimate=preds,
    times=times
)
```

### Calibration

```python
# Stratify by risk deciles
risk_deciles = pd.qcut(risk_scores, q=10, labels=False)

# For each decile:
# - Compute predicted probability at 30 days
# - Compute observed Kaplan-Meier probability at 30 days
# - Plot predicted vs observed
```

## Interpretability

### SHAP for XGBoost

```python
import shap

# Create explainer
explainer = shap.TreeExplainer(xgb_model)

# Compute SHAP values
shap_values = explainer.shap_values(X_test)

# Summary plot (global importance)
shap.summary_plot(shap_values, X_test, feature_names=feature_names)

# Waterfall plot (individual prediction)
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=X_test.iloc[0],
        feature_names=feature_names
    )
)
```

### Kaplan-Meier by Risk Strata

```python
from lifelines import KaplanMeierFitter

# Stratify into risk groups
risk_tertiles = pd.qcut(risk_scores, q=3, labels=['Low', 'Medium', 'High'])

# Plot KM curves
kmf = KaplanMeierFitter()
fig, ax = plt.subplots(figsize=(10, 6))

for risk_group in ['Low', 'Medium', 'High']:
    mask = risk_tertiles == risk_group
    kmf.fit(
        durations=y_test['time_to_event'][mask],
        event_observed=y_test['event'][mask],
        label=risk_group
    )
    kmf.plot_survival_function(ax=ax)

plt.xlabel('Days since discharge')
plt.ylabel('Probability of no readmission')
plt.title('Kaplan-Meier Curves by Risk Strata')
```

## Data Structures

### Cohort DataFrame Schema

```python
cohort_df = pd.DataFrame({
    'subject_id': int,           # Patient ID
    'hadm_id': int,              # Admission ID (index admission)
    'time_to_event': float,      # Days to readmission or censoring (0-30)
    'event': int,                # 1=readmitted, 0=censored
    
    # Demographics
    'age': float,                # Age at discharge
    'gender': str,               # M/F
    
    # Clinical
    'los': float,                # Length of stay (days)
    'admission_type': str,       # EMERGENCY, URGENT, ELECTIVE
    'discharge_location': str,   # HOME, SNF, etc.
    
    # Comorbidities
    'n_prior_admissions': int,   # Count in past 6 months
    'elixhauser_score': float,   # Comorbidity index
    'chf': int,                  # Congestive heart failure flag
    'copd': int,                 # COPD flag
    'diabetes': int,             # Diabetes flag
    # ... other comorbidities
    
    # Labs (optional)
    'creatinine': float,         # Last value before discharge
    'hemoglobin': float,
    # ... other labs
})
```

### Structured Array for sksurv

```python
# scikit-survival uses structured arrays
y_structured = np.array(
    [(event, time) for event, time in zip(df['event'], df['time_to_event'])],
    dtype=[('event', bool), ('time', float)]
)
```

## File Formats

### Saved Models

```python
# Cox model (pickle)
import pickle
with open('models/coxph_model.pkl', 'wb') as f:
    pickle.dump(cph, f)

# XGBoost model (JSON)
xgb_model.save_model('models/xgb_cox_model.json')
```

### Predictions Output

```python
# CSV format
predictions_df = pd.DataFrame({
    'subject_id': test_data['subject_id'],
    'hadm_id': test_data['hadm_id'],
    'risk_score': risk_scores,
    'predicted_survival_30d': survival_probs_30d,
    'risk_group': risk_groups,  # Low/Medium/High
    'actual_event': test_data['event'],
    'actual_time': test_data['time_to_event']
})
predictions_df.to_csv('results/predictions/test_predictions.csv', index=False)
```

### Metrics Output

```python
# JSON format
metrics = {
    'model': 'CoxPH',
    'c_index': float(c_index),
    'td_auc_30d': float(auc_30d),
    'ibs': float(ibs),
    'calibration_slope': float(calib_slope),
    'calibration_intercept': float(calib_intercept),
    'n_train': int(len(train)),
    'n_test': int(len(test)),
    'n_events_test': int(test['event'].sum())
}

import json
with open('results/metrics/coxph_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
```

## Performance Considerations

### Memory Management
- Load MIMIC data in chunks if needed (for full dataset)
- Use Dask for large-scale processing
- Save intermediate results to disk

### Computation Time
- Cox model: Fast (<1 min for 10K patients)
- XGBoost: Moderate (5-10 min with early stopping)
- SHAP: Slow for large test sets (use sampling)

### Optimization
- Use `n_jobs=-1` for parallel processing where available
- Enable early stopping for XGBoost
- Cache preprocessed data

## Testing Strategy

### Unit Tests
```python
# tests/test_data_loader.py
def test_load_admissions():
    df = load_admissions(sample=True)
    assert 'subject_id' in df.columns
    assert 'hadm_id' in df.columns
    assert not df.empty

# tests/test_preprocess.py
def test_compute_age():
    age = compute_age(anchor_year=2180, anchor_age=50, event_year=2185)
    assert age == 55

# tests/test_eval.py
def test_c_index_perfect():
    c_index = compute_concordance_index(
        event=[1, 1, 0],
        time=[10, 20, 30],
        risk=[1.0, 0.5, 0.1]
    )
    assert c_index == 1.0
```

### Integration Tests
```python
# tests/test_end_to_end.py
def test_full_pipeline_on_sample():
    # Load sample data
    df = pd.read_csv('data/sample/sample_cohort.csv')
    
    # Train model
    model = train_cox_model(df)
    
    # Make predictions
    preds = model.predict(df)
    
    # Compute metrics
    c_index = compute_c_index(df['event'], df['time'], preds)
    
    assert c_index > 0.5  # Better than random
```

## Debugging Tips

### Check Data Quality
```python
# Missing values
print(df.isnull().sum())

# Event rate
print(f"Event rate: {df['event'].mean():.2%}")

# Time distribution
print(df['time_to_event'].describe())
```

### Validate Cohort
```python
# No negative times
assert (df['time_to_event'] >= 0).all()

# Events should have time > 0
assert (df.loc[df['event'] == 1, 'time_to_event'] > 0).all()

# Censored should be <= 30 days
assert (df.loc[df['event'] == 0, 'time_to_event'] <= 30).all()
```

### Model Diagnostics
```python
# Cox: Check proportional hazards assumption
cph.check_assumptions(train_data, p_value_threshold=0.05)

# XGBoost: Plot learning curve
results = model.evals_result()
plt.plot(results['validation']['cox-nloglik'])
```

---

**Last Updated**: 2025-10-08

