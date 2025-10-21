"""
Entry script for training survival models.

This script dispatches to the appropriate training script based on model type.

Usage:
    python train.py --model cox --data data/processed --output models/cox_model.pkl
    python train.py --model xgb-cox --data data/processed --output models/xgb_cox_model.pkl
    python train.py --model xgb-aft --data data/processed --output models/xgb_aft_model.pkl
    python train.py --model rf --data data/processed --output models/rf_model.pkl
"""

import argparse
import subprocess
import sys
from app.utils import setup_logging


def main():
    """Main training dispatcher."""
    parser = argparse.ArgumentParser(description="Train survival models")
    parser.add_argument("--model", type=str, required=True, 
                        choices=["cox", "xgb-cox", "xgb-aft", "rf"],
                        help="Model type to train")
    parser.add_argument("--data", type=str, default="data/processed", 
                        help="Path to preprocessed data")
    parser.add_argument("--output", type=str, 
                        help="Path to save trained model")
    parser.add_argument("--wandb-project", type=str, default="survival-readmission", 
                        help="Weights & Biases project name")
    parser.add_argument("--wandb-run", type=str, 
                        help="Weights & Biases run name")
    
    # Model-specific arguments
    parser.add_argument("--penalizer", type=float, default=0.1, 
                        help="L2 penalizer for Cox PH")
    parser.add_argument("--l1-ratio", type=float, default=0.1, 
                        help="L1 ratio for Cox PH")
    parser.add_argument("--eta", type=float, default=0.05, 
                        help="Learning rate for XGBoost")
    parser.add_argument("--max-depth", type=int, default=3, 
                        help="Maximum tree depth for XGBoost")
    parser.add_argument("--num-boost-round", type=int, default=300, 
                        help="Number of boosting rounds for XGBoost")
    parser.add_argument("--n-estimators", type=int, default=300, 
                        help="Number of trees for Random Forest")
    parser.add_argument("--test-size", type=float, default=0.3, 
                        help="Test set size")
    parser.add_argument("--random-state", type=int, default=42, 
                        help="Random state for reproducibility")
    
    args = parser.parse_args()
    
    setup_logging()
    
    print(f"Starting model training: {args.model}...")
    
    # Set default output path if not provided
    if args.output is None:
        model_outputs = {
            "cox": "models/cox_model.pkl",
            "xgb-cox": "models/xgb_cox_model.pkl", 
            "xgb-aft": "models/xgb_aft_model.pkl",
            "rf": "models/rf_model.pkl"
        }
        args.output = model_outputs[args.model]
    
    # Set default wandb run name if not provided
    if args.wandb_run is None:
        args.wandb_run = args.model
    
    # Build command for the appropriate training script
    if args.model == "cox":
        cmd = [
            sys.executable, "train_cox.py",
            "--data", args.data,
            "--output", args.output,
            "--penalizer", str(args.penalizer),
            "--l1-ratio", str(args.l1_ratio),
            "--test-size", str(args.test_size),
            "--random-state", str(args.random_state),
            "--wandb-project", args.wandb_project,
            "--wandb-run", args.wandb_run
        ]
    elif args.model == "xgb-cox":
        cmd = [
            sys.executable, "train_xgb_cox.py",
            "--data", args.data,
            "--output", args.output,
            "--eta", str(args.eta),
            "--max-depth", str(args.max_depth),
            "--num-boost-round", str(args.num_boost_round),
            "--test-size", str(args.test_size),
            "--random-state", str(args.random_state),
            "--wandb-project", args.wandb_project,
            "--wandb-run", args.wandb_run
        ]
    elif args.model == "xgb-aft":
        cmd = [
            sys.executable, "train_xgb_aft.py",
            "--data", args.data,
            "--output", args.output,
            "--eta", str(args.eta),
            "--max-depth", str(args.max_depth),
            "--num-boost-round", str(args.num_boost_round),
            "--test-size", str(args.test_size),
            "--random-state", str(args.random_state),
            "--wandb-project", args.wandb_project,
            "--wandb-run", args.wandb_run
        ]
    elif args.model == "rf":
        cmd = [
            sys.executable, "train_rf.py",
            "--data", args.data,
            "--output", args.output,
            "--n-estimators", str(args.n_estimators),
            "--max-depth", str(args.max_depth),
            "--test-size", str(args.test_size),
            "--random-state", str(args.random_state),
            "--wandb-project", args.wandb_project,
            "--wandb-run", args.wandb_run
        ]
    
    # Execute the training script
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"Training complete for {args.model}!")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Training failed for {args.model}: {e}")
        return e.returncode
    except Exception as e:
        print(f"Error during training: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

