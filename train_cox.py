"""
Training script for Cox Proportional Hazards model.

This script trains a Cox PH model with configurable parameters and logs to Weights & Biases.

Usage:
    python train_cox.py --data data/processed --output models/cox_model.pkl
    python train_cox.py --data data/processed --output models/cox_model.pkl --penalizer 0.1 --l1_ratio 0.5
"""

import argparse
import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from lifelines import CoxPHFitter
from app.evaluation import compute_concordance_index, compute_td_auc, compute_brier_score
from app.utils import setup_logging

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Weights & Biases not available. Install with: pip install wandb")


def main():
    """Main Cox PH training pipeline."""
    parser = argparse.ArgumentParser(description="Train Cox Proportional Hazards model")
    parser.add_argument("--data", type=str, default="data/processed", 
                        help="Path to processed data directory")
    parser.add_argument("--output", type=str, default="models/cox_model.pkl", 
                        help="Path to save trained model")
    parser.add_argument("--penalizer", type=float, default=0.1, 
                        help="L2 penalizer for Cox PH")
    parser.add_argument("--l1_ratio", type=float, default=0.1, 
                        help="L1 ratio for elastic net regularization")
    parser.add_argument("--test-size", type=float, default=0.3, 
                        help="Test set size")
    parser.add_argument("--random-state", type=int, default=42, 
                        help="Random state for reproducibility")
    parser.add_argument("--wandb-project", type=str, default="survival-readmission", 
                        help="Weights & Biases project name")
    parser.add_argument("--wandb-run", type=str, default="cox-ph", 
                        help="Weights & Biases run name")
    args = parser.parse_args()
    
    setup_logging()
    
    print("Starting Cox PH model training...")
    print(f"Data directory: {args.data}")
    print(f"Output path: {args.output}")
    print(f"Penalizer: {args.penalizer}, L1 ratio: {args.l1_ratio}")
    
    # Initialize Weights & Biases
    if WANDB_AVAILABLE:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run,
            config={
                "model_type": "cox_ph",
                "penalizer": args.penalizer,
                "l1_ratio": args.l1_ratio,
                "test_size": args.test_size,
                "random_state": args.random_state
            }
        )
    
    try:
        # Load data
        data_dir = Path(args.data)
        X = pd.read_csv(data_dir / "features.csv")
        y = pd.read_csv(data_dir / "targets.csv")
        
        print(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Event rate: {y['event'].mean():.3f}")
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.random_state, 
            stratify=y['event']
        )
        
        print(f"Train set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        # Prepare data for Cox PH
        cox_train = X_train.copy()
        cox_train['time_to_event'] = y_train['time_to_event'].values
        cox_train['event'] = y_train['event'].values
        
        cox_test = X_test.copy()
        cox_test['time_to_event'] = y_test['time_to_event'].values
        cox_test['event'] = y_test['event'].values
        
        # Train Cox PH model
        print("Training Cox PH model...")
        cph = CoxPHFitter(penalizer=args.penalizer, l1_ratio=args.l1_ratio)
        cph.fit(cox_train, duration_col='time_to_event', event_col='event', show_progress=False)
        
        # Generate predictions
        risk_scores = cph.predict_partial_hazard(cox_test)
        
        # Compute metrics
        c_index = compute_concordance_index(y_test['event'], y_test['time_to_event'], risk_scores)
        
        # Time-dependent AUC
        time_horizons = [1, 7, 14, 21, 30]
        y_train_tuple = (y_train['event'].values, y_train['time_to_event'].values)
        y_test_tuple = (y_test['event'].values, y_test['time_to_event'].values)
        
        td_auc, mean_auc = compute_td_auc(y_train_tuple, y_test_tuple, risk_scores, time_horizons)
        
        # Integrated Brier Score
        survival_probs = cph.predict_survival_function(cox_test, times=time_horizons)
        survival_probs_matrix = survival_probs.T.values
        ibs = compute_brier_score(y_train_tuple, y_test_tuple, survival_probs_matrix, time_horizons)
        
        # Print results
        print(f"\nModel Performance:")
        print(f"C-index: {c_index:.3f}")
        print(f"Mean time-dependent AUC: {mean_auc:.3f}")
        print(f"Integrated Brier Score: {ibs:.3f}")
        
        # Log to Weights & Biases
        if WANDB_AVAILABLE:
            wandb.log({
                "c_index": c_index,
                "mean_td_auc": mean_auc,
                "ibs": ibs,
                "train_size": len(X_train),
                "test_size": len(X_test),
                "event_rate": y['event'].mean()
            })
            
            # Log time-dependent AUC
            for t, auc in zip(time_horizons, td_auc):
                wandb.log({f"td_auc_day_{t}": auc})
        
        # Save model and results
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': cph,
            'c_index': c_index,
            'mean_td_auc': mean_auc,
            'ibs': ibs,
            'td_auc': dict(zip(time_horizons, td_auc)),
            'X_test': X_test,
            'y_test': y_test,
            'risk_scores': risk_scores,
            'survival_probs': survival_probs_matrix
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to: {output_path}")
        
        if WANDB_AVAILABLE:
            wandb.finish()
        
        print("Cox PH training complete!")
        
    except Exception as e:
        print(f"Error during training: {e}")
        if WANDB_AVAILABLE:
            wandb.finish(exit_code=1)
        raise


if __name__ == "__main__":
    main()
