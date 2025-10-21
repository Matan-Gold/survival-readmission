"""
Entry script for analyzing results and generating reports.

This script analyzes model predictions, computes evaluation metrics,
and generates visualization plots.

Usage:
    python result.py --predictions results/predictions.csv --targets data/processed/targets.csv --output results/analysis
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from app.evaluation import compute_concordance_index, compute_td_auc, compute_brier_score, compute_calibration
from app.utils import setup_logging


def main():
    """Main results analysis pipeline."""
    parser = argparse.ArgumentParser(description="Analyze results and generate reports")
    parser.add_argument("--predictions", type=str, required=True, 
                        help="Path to predictions file")
    parser.add_argument("--targets", type=str, required=True,
                        help="Path to targets file")
    parser.add_argument("--output", type=str, default="results/analysis",
                        help="Path to save results")
    parser.add_argument("--model-name", type=str, default="Model",
                        help="Name of the model for reporting")
    args = parser.parse_args()
    
    setup_logging()
    
    print("Analyzing results...")
    print(f"Predictions: {args.predictions}")
    print(f"Targets: {args.targets}")
    print(f"Output: {args.output}")
    
    try:
        # Load data
        predictions = pd.read_csv(args.predictions)
        targets = pd.read_csv(args.targets)
        
        print(f"Predictions loaded: {predictions.shape}")
        print(f"Targets loaded: {targets.shape}")
        
        # Ensure same number of samples
        min_samples = min(len(predictions), len(targets))
        predictions = predictions.iloc[:min_samples]
        targets = targets.iloc[:min_samples]
        
        # Compute metrics
        print("Computing evaluation metrics...")
        
        # C-index
        c_index = compute_concordance_index(targets['event'], targets['time_to_event'], predictions['risk_score'])
        
        # Time-dependent AUC
        time_horizons = [1, 7, 14, 21, 30]
        y_train_tuple = (targets['event'].values, targets['time_to_event'].values)
        y_test_tuple = (targets['event'].values, targets['time_to_event'].values)
        
        td_auc, mean_auc = compute_td_auc(y_train_tuple, y_test_tuple, predictions['risk_score'], time_horizons)
        
        # Calibration
        calibration = compute_calibration(predictions['predicted_30d_risk'], targets['event'], n_bins=10)
        
        # Print results
        print(f"\n{args.model_name} Performance:")
        print(f"C-index: {c_index:.3f}")
        print(f"Mean time-dependent AUC: {mean_auc:.3f}")
        print(f"Event rate: {targets['event'].mean():.3f}")
        print(f"Mean predicted risk: {predictions['predicted_30d_risk'].mean():.3f}")
        
        # Create output directory
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate plots
        print("Generating visualizations...")
        
        # Set style
        sns.set_style('whitegrid')
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        
        # 1. Risk score distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(predictions['risk_score'], bins=30, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Risk Score')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{args.model_name} - Risk Score Distribution')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'risk_score_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Calibration plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Extract calibration data
        pred_rates = []
        obs_rates = []
        bin_counts = []
        
        for bin_data in calibration['bins']:
            if bin_data['n'] > 0:
                pred_rates.append(bin_data['pred'])
                obs_rates.append(bin_data['obs'])
                bin_counts.append(bin_data['n'])
        
        if pred_rates:
            ax.plot(pred_rates, obs_rates, 'bo-', linewidth=2, markersize=6, label='Model')
            ax.plot([0, 1], [0, 1], 'r--', alpha=0.7, label='Perfect Calibration')
            
            # Add sample sizes
            for i, (pred, obs, count) in enumerate(zip(pred_rates, obs_rates, bin_counts)):
                ax.annotate(f'n={count}', (pred, obs), xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax.set_xlabel('Predicted Probability')
        ax.set_ylabel('Observed Rate')
        ax.set_title(f'{args.model_name} - Calibration Plot')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'calibration_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Risk category analysis
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Risk category distribution
        risk_counts = predictions['risk_category'].value_counts()
        ax1.bar(risk_counts.index, risk_counts.values, alpha=0.7, color=['green', 'orange', 'red'])
        ax1.set_title('Risk Category Distribution')
        ax1.set_ylabel('Number of Patients')
        ax1.grid(True, alpha=0.3)
        
        # Event rate by risk category
        event_rates = []
        categories = []
        for category in ['Low', 'Medium', 'High']:
            mask = predictions['risk_category'] == category
            if mask.sum() > 0:
                event_rate = targets.loc[mask, 'event'].mean()
                event_rates.append(event_rate)
                categories.append(category)
        
        ax2.bar(categories, event_rates, alpha=0.7, color=['green', 'orange', 'red'])
        ax2.set_title('Event Rate by Risk Category')
        ax2.set_ylabel('Event Rate')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'risk_category_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Time-dependent AUC
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(time_horizons, td_auc, 'o-', linewidth=2, markersize=6)
        ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Random')
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('AUC')
        ax.set_title(f'{args.model_name} - Time-Dependent AUC')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'time_dependent_auc.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save detailed results
        results_summary = {
            'model_name': args.model_name,
            'c_index': c_index,
            'mean_td_auc': mean_auc,
            'td_auc_by_time': dict(zip(time_horizons, td_auc)),
            'event_rate': targets['event'].mean(),
            'mean_predicted_risk': predictions['predicted_30d_risk'].mean(),
            'calibration_bins': calibration['bins']
        }
        
        # Save results to CSV
        results_df = pd.DataFrame([results_summary])
        results_df.to_csv(output_dir / 'results_summary.csv', index=False)
        
        # Save detailed metrics
        metrics_df = pd.DataFrame({
            'metric': ['C-index', 'Mean TD-AUC', 'Event Rate', 'Mean Predicted Risk'],
            'value': [c_index, mean_auc, targets['event'].mean(), predictions['predicted_30d_risk'].mean()]
        })
        metrics_df.to_csv(output_dir / 'metrics.csv', index=False)
        
        print(f"Results saved to: {output_dir}")
        print("Results analysis complete!")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        raise


if __name__ == "__main__":
    main()

