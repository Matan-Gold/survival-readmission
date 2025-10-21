"""
Entry script for generating predictions.

This script loads a trained model and generates survival risk scores
and hazard estimates for new data.

Usage:
    python predict.py --model models/cox_model.pkl --data data/processed/features.csv --output results/predictions.csv
"""

import argparse
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from app.utils import setup_logging


def main():
    """Main prediction pipeline."""
    parser = argparse.ArgumentParser(description="Generate survival predictions")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--data", type=str, required=True, help="Path to input data")
    parser.add_argument("--output", type=str, help="Path to save predictions")
    parser.add_argument("--time-horizons", type=str, default="1,7,14,21,30", 
                        help="Comma-separated time horizons for survival probabilities")
    args = parser.parse_args()
    
    setup_logging()
    
    print("Generating predictions...")
    print(f"Model: {args.model}")
    print(f"Data: {args.data}")
    
    try:
        # Load model
        with open(args.model, 'rb') as f:
            model_data = pickle.load(f)
        
        model = model_data['model']
        print(f"Model loaded: {type(model).__name__}")
        
        # Load data
        X = pd.read_csv(args.data)
        print(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Generate predictions based on model type
        if hasattr(model, 'predict_partial_hazard'):  # Cox PH model
            print("Generating Cox PH predictions...")
            risk_scores = model.predict_partial_hazard(X)
            
            # Generate survival probabilities
            time_horizons = [int(t) for t in args.time_horizons.split(',')]
            survival_probs = model.predict_survival_function(X, times=time_horizons)
            survival_probs_matrix = survival_probs.T.values
            
        elif hasattr(model, 'predict'):  # XGBoost or Random Forest
            print("Generating tree-based model predictions...")
            if hasattr(model, 'DMatrix'):  # XGBoost
                import xgboost as xgb
                dtest = xgb.DMatrix(X)
                risk_scores = model.predict(dtest)
            else:  # Random Forest
                risk_scores = model.predict_proba(X)[:, 1]
            
            # Approximate survival probabilities
            time_horizons = [int(t) for t in args.time_horizons.split(',')]
            if hasattr(model, 'DMatrix'):  # XGBoost
                survival_probs = np.exp(-risk_scores.reshape(-1, 1) * np.array(time_horizons).reshape(1, -1))
            else:  # Random Forest
                survival_probs = 1 - risk_scores.reshape(-1, 1) * np.array(time_horizons).reshape(1, -1) / 30
                survival_probs = np.clip(survival_probs, 0, 1)
            
            survival_probs_matrix = survival_probs
        
        # Create predictions dataframe
        predictions = pd.DataFrame({
            'risk_score': risk_scores,
            'predicted_30d_risk': 1 - survival_probs_matrix[:, -1] if survival_probs_matrix.shape[1] > 0 else 1 - survival_probs[:, -1]
        })
        
        # Add time-specific survival probabilities
        for i, t in enumerate(time_horizons):
            if survival_probs_matrix.shape[1] > i:
                predictions[f'survival_prob_{t}d'] = survival_probs_matrix[:, i]
            else:
                predictions[f'survival_prob_{t}d'] = survival_probs[:, i]
        
        # Add risk categories
        predictions['risk_category'] = pd.cut(
            predictions['risk_score'], 
            bins=3, 
            labels=['Low', 'Medium', 'High']
        )
        
        # Save predictions
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            predictions.to_csv(output_path, index=False)
            print(f"Predictions saved to: {output_path}")
        else:
            print("Predictions generated (not saved)")
        
        # Print summary
        print(f"\nPrediction Summary:")
        print(f"Mean risk score: {predictions['risk_score'].mean():.3f}")
        print(f"Mean 30-day risk: {predictions['predicted_30d_risk'].mean():.3f}")
        print(f"Risk category distribution:")
        print(predictions['risk_category'].value_counts())
        
        print("Predictions complete!")
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise


if __name__ == "__main__":
    main()

