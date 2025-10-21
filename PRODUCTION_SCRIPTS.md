# Production Scripts Documentation

This document describes the production-level scripts for the survival readmission prediction system.

## Overview

The production system consists of several modular scripts that can be run independently or as part of a complete pipeline:

- **`preprocess.py`** - Data preprocessing and feature engineering
- **`train.py`** - Model training dispatcher
- **`train_cox.py`** - Cox Proportional Hazards training
- **`train_xgb_cox.py`** - XGBoost Cox training
- **`train_xgb_aft.py`** - XGBoost AFT training
- **`train_rf.py`** - Random Forest training
- **`predict.py`** - Generate predictions
- **`result.py`** - Analyze results and generate reports
- **`tasks.py`** - Task automation with invoke

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Full Pipeline

```bash
# Using invoke (recommended)
invoke full-pipeline --model cox --wandb-project survival-readmission

# Or manually
python preprocess.py
python train.py --model cox --wandb-project survival-readmission
python predict.py --model models/cox_model.pkl --data data/processed/features.csv --output results/cox_predictions.csv
python result.py --predictions results/cox_predictions.csv --targets data/processed/targets.csv --output results/cox_analysis --model-name Cox
```

## Detailed Usage

### Data Preprocessing

```bash
python preprocess.py --input data/raw/mimic-iv-demo --output data/processed
```

**Arguments:**
- `--input`: Path to input data directory (default: `data/raw/mimic-iv-demo`)
- `--output`: Path to output processed data (default: `data/processed`)
- `--cohort-file`: Name of cohort output file (default: `cohort_30d.csv`)
- `--features-file`: Name of features output file (default: `features.csv`)

**Outputs:**
- `cohort_30d.csv`: Cohort definition with event and time information
- `features.csv`: Feature matrix
- `targets.csv`: Target variables (event, time_to_event)

### Model Training

#### Using the Dispatcher

```bash
# Train Cox PH model
python train.py --model cox --data data/processed --output models/cox_model.pkl

# Train XGBoost Cox model
python train.py --model xgb-cox --data data/processed --eta 0.05 --max-depth 3

# Train XGBoost AFT model
python train.py --model xgb-aft --data data/processed --eta 0.05 --max-depth 3

# Train Random Forest model
python train.py --model rf --data data/processed --n-estimators 500 --max-depth 10
```

#### Direct Training Scripts

```bash
# Cox PH with custom parameters
python train_cox.py --data data/processed --output models/cox_model.pkl --penalizer 0.1 --l1-ratio 0.5

# XGBoost Cox with custom parameters
python train_xgb_cox.py --data data/processed --output models/xgb_cox_model.pkl --eta 0.05 --max-depth 3 --num-boost-round 300

# XGBoost AFT with custom parameters
python train_xgb_aft.py --data data/processed --output models/xgb_aft_model.pkl --eta 0.05 --max-depth 3 --num-boost-round 300

# Random Forest with custom parameters
python train_rf.py --data data/processed --output models/rf_model.pkl --n-estimators 500 --max-depth 10
```

### Generating Predictions

```bash
python predict.py --model models/cox_model.pkl --data data/processed/features.csv --output results/predictions.csv
```

**Arguments:**
- `--model`: Path to trained model
- `--data`: Path to input features
- `--output`: Path to save predictions
- `--time-horizons`: Comma-separated time horizons (default: `1,7,14,21,30`)

**Outputs:**
- Risk scores
- Predicted 30-day risk
- Survival probabilities for each time horizon
- Risk categories (Low, Medium, High)

### Analyzing Results

```bash
python result.py --predictions results/predictions.csv --targets data/processed/targets.csv --output results/analysis --model-name Cox
```

**Arguments:**
- `--predictions`: Path to predictions file
- `--targets`: Path to targets file
- `--output`: Path to save analysis results
- `--model-name`: Name of the model for reporting

**Outputs:**
- Performance metrics (C-index, time-dependent AUC, calibration)
- Visualizations (risk distribution, calibration plot, risk categories)
- Detailed results summary

## Weights & Biases Integration

All training scripts automatically log to Weights & Biases for experiment tracking:

```bash
# Initialize W&B (first time only)
wandb login

# Train with W&B logging
python train.py --model cox --wandb-project survival-readmission --wandb-run cox-experiment-1
```

**Logged Metrics:**
- C-index
- Time-dependent AUC
- Integrated Brier Score
- Training parameters
- Model performance over time

## Task Automation

Use `invoke` for convenient task automation:

```bash
# Install invoke if not already installed
pip install invoke

# Available tasks
invoke --list

# Run individual tasks
invoke preprocess
invoke train --model cox --wandb-project survival-readmission
invoke train-all --wandb-project survival-readmission
invoke predict --model cox
invoke result --model cox

# Run full pipeline
invoke full-pipeline --model cox --wandb-project survival-readmission

# Clean up
invoke clean
```

## Model-Specific Parameters

### Cox Proportional Hazards
- `--penalizer`: L2 penalizer (default: 0.1)
- `--l1-ratio`: L1 ratio for elastic net (default: 0.1)

### XGBoost Models
- `--eta`: Learning rate (default: 0.05)
- `--max-depth`: Maximum tree depth (default: 3)
- `--subsample`: Subsample ratio (default: 0.8)
- `--colsample-bytree`: Column sampling ratio (default: 0.8)
- `--num-boost-round`: Number of boosting rounds (default: 300)

### Random Forest
- `--n-estimators`: Number of trees (default: 300)
- `--max-depth`: Maximum tree depth (default: 10)
- `--min-samples-split`: Minimum samples to split (default: 2)
- `--min-samples-leaf`: Minimum samples per leaf (default: 1)
- `--max-features`: Number of features to consider (default: "sqrt")

## Output Structure

```
models/
├── cox_model.pkl
├── xgb_cox_model.pkl
├── xgb_aft_model.pkl
└── rf_model.pkl

results/
├── predictions/
│   ├── cox_predictions.csv
│   ├── xgb_cox_predictions.csv
│   ├── xgb_aft_predictions.csv
│   └── rf_predictions.csv
└── analysis/
    ├── cox_analysis/
    ├── xgb_cox_analysis/
    ├── xgb_aft_analysis/
    └── rf_analysis/
```

## Error Handling

All scripts include comprehensive error handling:
- Input validation
- Model loading verification
- Data consistency checks
- Graceful failure with informative error messages

## Performance Considerations

- **Memory**: XGBoost models may require more memory for large datasets
- **CPU**: Random Forest uses all available cores by default
- **Storage**: Model files can be large (especially XGBoost)
- **Time**: Training time varies by model complexity and data size

## Troubleshooting

### Common Issues

1. **Missing dependencies**: Install all requirements with `pip install -r requirements.txt`
2. **Data not found**: Ensure data is in the correct directory structure
3. **Model loading errors**: Check that model files exist and are not corrupted
4. **W&B errors**: Ensure you're logged in with `wandb login`

### Debug Mode

Run scripts with verbose output:
```bash
python train.py --model cox --data data/processed --wandb-project survival-readmission --wandb-run debug-run
```

## Next Steps

1. **Hyperparameter tuning**: Use W&B sweeps for automated hyperparameter optimization
2. **Model ensemble**: Combine multiple models for improved performance
3. **Cross-validation**: Implement k-fold cross-validation for robust evaluation
4. **Feature selection**: Add automated feature selection capabilities
5. **Model deployment**: Create API endpoints for real-time predictions
