"""
Training script for Random Forest model (classification proxy).

This script trains a Random Forest model for 30-day readmission prediction and logs to Weights & Biases.

Usage:
    python train_rf.py --data data/processed --output models/rf_model.pkl
    python train_rf.py --data data/processed --output models/rf_model.pkl --n-estimators 500 --max-depth 10
"""

import argparse
import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
from app.evaluation import compute_concordance_index, compute_td_auc, compute_brier_score
from app.utils import setup_logging

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Weights & Biases not available. Install with: pip install wandb")


def main():
    """Main Random Forest training pipeline."""
    parser = argparse.ArgumentParser(description="Train Random Forest model")
    parser.add_argument("--data", type=str, default="data/processed", 
                        help="Path to processed data directory")
    parser.add_argument("--output", type=str, default="models/rf_model.pkl", 
                        help="Path to save trained model")
    parser.add_argument("--n-estimators", type=int, default=300, 
                        help="Number of trees in the forest")
    parser.add_argument("--max-depth", type=int, default=10, 
                        help="Maximum depth of the tree")
    parser.add_argument("--min-samples-split", type=int, default=2, 
                        help="Minimum number of samples required to split an internal node")
    parser.add_argument("--min-samples-leaf", type=int, default=1, 
                        help="Minimum number of samples required to be at a leaf node")
    parser.add_argument("--max-features", type=str, default="sqrt", 
                        help="Number of features to consider when looking for the best split")
    parser.add_argument("--test-size", type=float, default=0.3, 
                        help="Test set size")
    parser.add_argument("--random-state", type=int, default=42, 
                        help="Random state for reproducibility")
    parser.add_argument("--wandb-project", type=str, default="survival-readmission", 
                        help="Weights & Biases project name")
    parser.add_argument("--wandb-run", type=str, default="random-forest", 
                        help="Weights & Biases run name")
    args = parser.parse_args()
    
    setup_logging()
    
    print("Starting Random Forest model training...")
    print(f"Data directory: {args.data}")
    print(f"Output path: {args.output}")
    print(f"Parameters: n_estimators={args.n_estimators}, max_depth={args.max_depth}")
    
    # Initialize Weights & Biases
    if WANDB_AVAILABLE:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run,
            config={
                "model_type": "random_forest",
                "n_estimators": args.n_estimators,
                "max_depth": args.max_depth,
                "min_samples_split": args.min_samples_split,
                "min_samples_leaf": args.min_samples_leaf,
                "max_features": args.max_features,
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
        
        # Train Random Forest model
        print("Training Random Forest model...")
        rf = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf,
            max_features=args.max_features,
            random_state=args.random_state,
            n_jobs=-1
        )
        rf.fit(X_train, y_train['event'])
        
        # Generate predictions
        y_pred_proba = rf.predict_proba(X_test)[:, 1]
        y_pred = rf.predict(X_test)
        
        # Compute metrics
        auc = roc_auc_score(y_test['event'], y_pred_proba)
        c_index = compute_concordance_index(y_test['event'], y_test['time_to_event'], y_pred_proba)
        
        # Time-dependent AUC
        time_horizons = [1, 7, 14, 21, 30]
        y_train_tuple = (y_train['event'].values, y_train['time_to_event'].values)
        y_test_tuple = (y_test['event'].values, y_test['time_to_event'].values)
        
        td_auc, mean_auc = compute_td_auc(y_train_tuple, y_test_tuple, y_pred_proba, time_horizons)
        
        # Approximate survival probabilities for IBS
        survival_probs = 1 - y_pred_proba.reshape(-1, 1) * np.array(time_horizons).reshape(1, -1) / 30
        survival_probs = np.clip(survival_probs, 0, 1)
        ibs = compute_brier_score(y_train_tuple, y_test_tuple, survival_probs, time_horizons)
        
        # Feature importance
        feature_importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
        
        # Print results
        print(f"\nModel Performance:")
        print(f"AUC: {auc:.3f}")
        print(f"C-index: {c_index:.3f}")
        print(f"Mean time-dependent AUC: {mean_auc:.3f}")
        print(f"Integrated Brier Score: {ibs:.3f}")
        
        print(f"\nTop 10 Most Important Features:")
        for i, (feature, importance) in enumerate(feature_importance.head(10).items()):
            print(f"{i+1:2d}. {feature}: {importance:.3f}")
        
        # Log to Weights & Biases
        if WANDB_AVAILABLE:
            wandb.log({
                "auc": auc,
                "c_index": c_index,
                "mean_td_auc": mean_auc,
                "ibs": ibs,
                "train_size": len(X_train),
                "test_size": len(X_test),
                "event_rate": y['event'].mean()
            })
            
            # Log time-dependent AUC
            for t, auc_t in zip(time_horizons, td_auc):
                wandb.log({f"td_auc_day_{t}": auc_t})
            
            # Log feature importance
            for i, (feature, importance) in enumerate(feature_importance.head(20).items()):
                wandb.log({f"feature_importance_{i+1}": importance})
        
        # Save model and results
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': rf,
            'auc': auc,
            'c_index': c_index,
            'mean_td_auc': mean_auc,
            'ibs': ibs,
            'td_auc': dict(zip(time_horizons, td_auc)),
            'X_test': X_test,
            'y_test': y_test,
            'y_pred_proba': y_pred_proba,
            'y_pred': y_pred,
            'survival_probs': survival_probs,
            'feature_importance': feature_importance,
            'params': {
                'n_estimators': args.n_estimators,
                'max_depth': args.max_depth,
                'min_samples_split': args.min_samples_split,
                'min_samples_leaf': args.min_samples_leaf,
                'max_features': args.max_features,
                'random_state': args.random_state
            }
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to: {output_path}")
        
        if WANDB_AVAILABLE:
            wandb.finish()
        
        print("Random Forest training complete!")
        
    except Exception as e:
        print(f"Error during training: {e}")
        if WANDB_AVAILABLE:
            wandb.finish(exit_code=1)
        raise


if __name__ == "__main__":
    main()
