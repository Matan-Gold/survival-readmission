"""
Training script for XGBoost Survival model with AFT objective.

This script trains an XGBoost model with survival:aft objective and logs to Weights & Biases.

Usage:
    python train_xgb_aft.py --data data/processed --output models/xgb_aft_model.pkl
    python train_xgb_aft.py --data data/processed --output models/xgb_aft_model.pkl --eta 0.05 --max-depth 3
"""

import argparse
import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import xgboost as xgb
from app.evaluation import compute_concordance_index, compute_td_auc, compute_brier_score
from app.utils import setup_logging

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Weights & Biases not available. Install with: pip install wandb")


def main():
    """Main XGBoost AFT training pipeline."""
    parser = argparse.ArgumentParser(description="Train XGBoost Survival model with AFT objective")
    parser.add_argument("--data", type=str, default="data/processed", 
                        help="Path to processed data directory")
    parser.add_argument("--output", type=str, default="models/xgb_aft_model.pkl", 
                        help="Path to save trained model")
    parser.add_argument("--eta", type=float, default=0.05, 
                        help="Learning rate")
    parser.add_argument("--max-depth", type=int, default=3, 
                        help="Maximum tree depth")
    parser.add_argument("--subsample", type=float, default=0.8, 
                        help="Subsample ratio")
    parser.add_argument("--colsample-bytree", type=float, default=0.8, 
                        help="Column sampling ratio")
    parser.add_argument("--num-boost-round", type=int, default=300, 
                        help="Number of boosting rounds")
    parser.add_argument("--test-size", type=float, default=0.3, 
                        help="Test set size")
    parser.add_argument("--random-state", type=int, default=42, 
                        help="Random state for reproducibility")
    parser.add_argument("--wandb-project", type=str, default="survival-readmission", 
                        help="Weights & Biases project name")
    parser.add_argument("--wandb-run", type=str, default="xgb-aft", 
                        help="Weights & Biases run name")
    args = parser.parse_args()
    
    setup_logging()
    
    print("Starting XGBoost AFT model training...")
    print(f"Data directory: {args.data}")
    print(f"Output path: {args.output}")
    print(f"Parameters: eta={args.eta}, max_depth={args.max_depth}, num_rounds={args.num_boost_round}")
    
    # Initialize Weights & Biases
    if WANDB_AVAILABLE:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run,
            config={
                "model_type": "xgboost_aft",
                "eta": args.eta,
                "max_depth": args.max_depth,
                "subsample": args.subsample,
                "colsample_bytree": args.colsample_bytree,
                "num_boost_round": args.num_boost_round,
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
        
        # Prepare data for XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train['time_to_event'].values)
        dtest = xgb.DMatrix(X_test, label=y_test['time_to_event'].values)
        
        # XGBoost parameters
        params = {
            'objective': 'survival:aft',
            'eval_metric': 'aft-nloglik',
            'eta': args.eta,
            'max_depth': args.max_depth,
            'subsample': args.subsample,
            'colsample_bytree': args.colsample_bytree,
            'seed': args.random_state,
        }
        
        # Train XGBoost model
        print("Training XGBoost AFT model...")
        model = xgb.train(params, dtrain, num_boost_round=args.num_boost_round, verbose_eval=False)
        
        # Generate predictions
        risk_scores = model.predict(dtest)
        
        # Compute metrics
        c_index = compute_concordance_index(y_test['event'], y_test['time_to_event'], risk_scores)
        
        # Time-dependent AUC
        time_horizons = [1, 7, 14, 21, 30]
        y_train_tuple = (y_train['event'].values, y_train['time_to_event'].values)
        y_test_tuple = (y_test['event'].values, y_test['time_to_event'].values)
        
        td_auc, mean_auc = compute_td_auc(y_train_tuple, y_test_tuple, risk_scores, time_horizons)
        
        # Approximate survival probabilities for IBS
        survival_probs = np.exp(-risk_scores.reshape(-1, 1) * np.array(time_horizons).reshape(1, -1))
        ibs = compute_brier_score(y_train_tuple, y_test_tuple, survival_probs, time_horizons)
        
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
            'model': model,
            'c_index': c_index,
            'mean_td_auc': mean_auc,
            'ibs': ibs,
            'td_auc': dict(zip(time_horizons, td_auc)),
            'X_test': X_test,
            'y_test': y_test,
            'risk_scores': risk_scores,
            'survival_probs': survival_probs,
            'params': params
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to: {output_path}")
        
        if WANDB_AVAILABLE:
            wandb.finish()
        
        print("XGBoost AFT training complete!")
        
    except Exception as e:
        print(f"Error during training: {e}")
        if WANDB_AVAILABLE:
            wandb.finish(exit_code=1)
        raise


if __name__ == "__main__":
    main()
