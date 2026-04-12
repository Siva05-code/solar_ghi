"""
Comprehensive Model Comparison & Analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 5)


def compare_all_models(results_dict):
    """
    Compare performance of all models
    
    Args:
        results_dict: Dictionary containing results from all models
    """
    print("\n" + "="*80)
    print("MODEL COMPARISON & ANALYSIS")
    print("="*80)
    
    # Create comparison dataframe
    comparison_data = []
    for model_name, results in results_dict.items():
        metrics = results['metrics']
        comparison_data.append({
            'Model': model_name,
            'MSE': metrics['mse'],
            'RMSE': metrics['rmse'],
            'MAE': metrics['mae'],
            'R²': metrics['r2'],
            'MAPE': metrics.get('mape', 0)
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.set_index('Model').sort_values('RMSE')
    
    print("\n" + comparison_df.to_string())
    print("\n" + "="*80)
    
    # Save to CSV
    comparison_df.to_csv('/Users/sivakarthick/s2/results/model_comparison.csv')
    print(f"✓ Comparison saved: /Users/sivakarthick/s2/results/model_comparison.csv")
    
    return comparison_df


def plot_metric_comparison(comparison_df):
    """
    Create visualization comparing metrics across models
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # RMSE
    axes[0, 0].barh(comparison_df.index, comparison_df['RMSE'], color='steelblue')
    axes[0, 0].set_title('RMSE Comparison (Lower is Better)', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('RMSE')
    for i, v in enumerate(comparison_df['RMSE']):
        axes[0, 0].text(v + 0.001, i, f'{v:.4f}', va='center')
    
    # MAE
    axes[0, 1].barh(comparison_df.index, comparison_df['MAE'], color='coral')
    axes[0, 1].set_title('MAE Comparison (Lower is Better)', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('MAE')
    for i, v in enumerate(comparison_df['MAE']):
        axes[0, 1].text(v + 0.0005, i, f'{v:.4f}', va='center')
    
    # R²
    axes[1, 0].barh(comparison_df.index, comparison_df['R²'], color='mediumseagreen')
    axes[1, 0].set_title('R² Comparison (Higher is Better)', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('R²')
    for i, v in enumerate(comparison_df['R²']):
        axes[1, 0].text(v + 0.01, i, f'{v:.4f}', va='center')
    
    # MAPE
    axes[1, 1].barh(comparison_df.index, comparison_df['MAPE'], color='mediumpurple')
    axes[1, 1].set_title('MAPE Comparison (Lower is Better)', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('MAPE')
    for i, v in enumerate(comparison_df['MAPE']):
        axes[1, 1].text(v + 0.5, i, f'{v:.2f}%', va='center')
    
    plt.tight_layout()
    plt.savefig('/Users/sivakarthick/s2/results/metric_comparison.png', dpi=100, bbox_inches='tight')
    print(f"✓ Metric comparison plot saved: /Users/sivakarthick/s2/results/metric_comparison.png")
    plt.close()


def plot_predictions_comparison(results_dict, n_samples=500):
    """
    Compare predictions from all models on same test data
    """
    fig, axes = plt.subplots(len(results_dict), 1, figsize=(15, 4*len(results_dict)))
    
    if len(results_dict) == 1:
        axes = [axes]
    
    for idx, (model_name, results) in enumerate(sorted(results_dict.items())):
        y_test = results['y_test'][:n_samples]
        y_pred = results['y_pred'][:n_samples]
        
        axes[idx].plot(y_test, label='Actual', alpha=0.8, linewidth=2, color='black')
        axes[idx].plot(y_pred, label=f'{model_name} Prediction', alpha=0.7, linewidth=1.5)
        axes[idx].fill_between(range(n_samples), y_test, y_pred, alpha=0.2)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        axes[idx].set_title(
            f'{model_name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}',
            fontsize=12, fontweight='bold'
        )
        axes[idx].set_ylabel('Normalized GHI')
        axes[idx].legend(loc='upper right', fontsize=10)
        axes[idx].grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time Sample')
    plt.tight_layout()
    plt.savefig('/Users/sivakarthick/s2/results/predictions_comparison.png', dpi=100, bbox_inches='tight')
    print(f"✓ Predictions comparison plot saved: /Users/sivakarthick/s2/results/predictions_comparison.png")
    plt.close()


def plot_error_distribution(results_dict):
    """
    Compare error distributions across models
    """
    fig, axes = plt.subplots(1, len(results_dict), figsize=(4*len(results_dict), 5))
    
    if len(results_dict) == 1:
        axes = [axes]
    
    for idx, (model_name, results) in enumerate(sorted(results_dict.items())):
        y_test = results['y_test']
        y_pred = results['y_pred']
        errors = np.abs(y_test - y_pred)
        
        axes[idx].hist(errors, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        axes[idx].axvline(errors.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {errors.mean():.4f}')
        axes[idx].axvline(np.median(errors), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(errors):.4f}')
        axes[idx].set_title(f'{model_name} Error Distribution', fontsize=11, fontweight='bold')
        axes[idx].set_xlabel('Absolute Error')
        axes[idx].set_ylabel('Frequency')
        axes[idx].legend(fontsize=9)
        axes[idx].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('/Users/sivakarthick/s2/results/error_distribution.png', dpi=100, bbox_inches='tight')
    print(f"✓ Error distribution plot saved: /Users/sivakarthick/s2/results/error_distribution.png")
    plt.close()


def plot_actual_vs_predicted(results_dict):
    """
    Create scatter plots of actual vs predicted values
    """
    fig, axes = plt.subplots(1, len(results_dict), figsize=(5*len(results_dict), 5))
    
    if len(results_dict) == 1:
        axes = [axes]
    
    for idx, (model_name, results) in enumerate(sorted(results_dict.items())):
        y_test = results['y_test']
        y_pred = results['y_pred']
        
        axes[idx].scatter(y_test, y_pred, alpha=0.5, s=10)
        
        # Perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        axes[idx].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        r2 = r2_score(y_test, y_pred)
        axes[idx].set_title(f'{model_name} (R²: {r2:.4f})', fontsize=11, fontweight='bold')
        axes[idx].set_xlabel('Actual GHI')
        axes[idx].set_ylabel('Predicted GHI')
        axes[idx].legend(fontsize=9)
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/Users/sivakarthick/s2/results/actual_vs_predicted.png', dpi=100, bbox_inches='tight')
    print(f"✓ Actual vs Predicted plot saved: /Users/sivakarthick/s2/results/actual_vs_predicted.png")
    plt.close()


def generate_summary_report(comparison_df, results_dict):
    """
    Generate a comprehensive summary report
    """
    report = []
    report.append("="*80)
    report.append("SOLAR IRRADIANCE (GHI) PREDICTION - MODEL EVALUATION REPORT")
    report.append("="*80)
    report.append(f"\nTotal Models Evaluated: {len(results_dict)}")
    report.append(f"Models: {', '.join(results_dict.keys())}\n")
    
    # Best performers
    best_rmse_idx = comparison_df['RMSE'].idxmin()
    best_r2_idx = comparison_df['R²'].idxmax()
    best_mae_idx = comparison_df['MAE'].idxmin()
    
    report.append("BEST PERFORMERS:")
    report.append(f"  • Best RMSE: {best_rmse_idx} ({comparison_df.loc[best_rmse_idx, 'RMSE']:.6f})")
    report.append(f"  • Best R²:   {best_r2_idx} ({comparison_df.loc[best_r2_idx, 'R²']:.6f})")
    report.append(f"  • Best MAE:  {best_mae_idx} ({comparison_df.loc[best_mae_idx, 'MAE']:.6f})")
    
    report.append("\n" + "="*80)
    report.append("DETAILED METRICS:")
    report.append("="*80)
    report.append(comparison_df.to_string())
    
    report.append("\n" + "="*80)
    report.append("MODEL STRENGTHS & WEAKNESSES:")
    report.append("="*80)
    
    for model_name, results in sorted(results_dict.items()):
        metrics = results['metrics']
        report.append(f"\n{model_name}:")
        report.append(f"  MSE:  {metrics['mse']:.6f}")
        report.append(f"  RMSE: {metrics['rmse']:.6f}")
        report.append(f"  MAE:  {metrics['mae']:.6f}")
        report.append(f"  R²:   {metrics['r2']:.6f}")
        report.append(f"  MAPE: {metrics.get('mape', 'N/A')}")
    
    report.append("\n" + "="*80)
    report.append("RECOMMENDATIONS:")
    report.append("="*80)
    report.append(f"1. Best Overall Model: {best_rmse_idx} (lowest RMSE)")
    report.append(f"2. For explainability: Consider tree-based models (RF, XGBoost)")
    report.append(f"3. For accuracy: Use Transformer model for best spatial-temporal representation")
    report.append(f"4. For interpretability with good performance: Use {best_mae_idx}")
    
    # Write report
    report_text = "\n".join(report)
    print("\n" + report_text)
    
    ensure_dir(RESULTS_DIR)
    with open(EVALUATION_REPORT_FILE, 'w') as f:
        f.write(report_text)
    
    print(f"\n✓ Report saved: {EVALUATION_REPORT_FILE}")


if __name__ == "__main__":
    print("\nNote: Run this after training all models!")
    print("This script requires outputs from:")
    print("  - 02_arima_model.py")
    print("  - 03_tree_models.py")
    print("  - 04_lstm_model.py")
    print("  - 05_transformer_model.py")
