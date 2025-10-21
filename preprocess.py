"""
Entry script for data preprocessing.

This script loads raw data, defines the cohort, and engineers features
for survival analysis.

Usage:
    python preprocess.py --input data/raw/mimic-iv-demo --output data/processed
"""

import argparse
import os
import pandas as pd
from pathlib import Path
from app.feature_engineering import define_cohort, engineer_features
from app.utils import setup_logging, load_config


def main():
    """Main preprocessing pipeline."""
    parser = argparse.ArgumentParser(description="Preprocess MIMIC-IV data for survival analysis")
    parser.add_argument("--input", type=str, default="data/raw/mimic-iv-demo", 
                        help="Path to input data directory")
    parser.add_argument("--output", type=str, default="data/processed", 
                        help="Path to output processed data")
    parser.add_argument("--cohort-file", type=str, default="cohort_30d.csv",
                        help="Name of cohort output file")
    parser.add_argument("--features-file", type=str, default="features.csv",
                        help="Name of features output file")
    args = parser.parse_args()
    
    setup_logging()
    
    print("Starting data preprocessing...")
    print(f"Input directory: {args.input}")
    print(f"Output directory: {args.output}")
    
    # Set environment variable for MIMIC data location
    os.environ['MIMIC_DEMO_DIR'] = args.input
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Define cohort
        print("Defining cohort...")
        cohort = define_cohort()
        print(f"Cohort shape: {cohort.shape}")
        print(f"Event rate: {cohort['event'].mean():.3f}")
        
        # Save cohort
        cohort_path = output_dir / args.cohort_file
        cohort.to_csv(cohort_path, index=False)
        print(f"Cohort saved to: {cohort_path}")
        
        # Engineer features
        print("Engineering features...")
        X, y = engineer_features(cohort)
        print(f"Feature matrix shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        
        # Save features and targets
        features_path = output_dir / args.features_file
        targets_path = output_dir / "targets.csv"
        
        X.to_csv(features_path, index=False)
        y.to_csv(targets_path, index=False)
        
        print(f"Features saved to: {features_path}")
        print(f"Targets saved to: {targets_path}")
        
        # Print feature summary
        print("\nFeature Summary:")
        print(f"Total features: {X.shape[1]}")
        print(f"Lab features: {sum(1 for col in X.columns if col.startswith('lab_'))}")
        print(f"Missing values: {X.isnull().sum().sum()}")
        
        print("\nPreprocessing complete!")
        
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        raise


if __name__ == "__main__":
    main()

