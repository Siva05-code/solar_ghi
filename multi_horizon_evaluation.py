"""
Multi-Horizon Evaluation Framework
Evaluates forecasting models across different time horizons (short-term and long-term)
Follows synopsis requirements for comprehensive model comparison
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import warnings
import pickle
import os

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")


class MultiHorizonEvaluator:
    """Evaluates forecasting models across multiple time horizons"""
    
    def __init__(self, y_test, y_pred_dict, horizons=[1, 6, 12, 24]):
        """
        Args:
            y_test: Ground truth values
            y_pred_dict: Dictionary of {model_name: predictions}
            horizons: List of horizon indices to evaluate (in hours)
        """
        self.y_test = y_test
        self.y_pred_dict = y_pred_dict
        self.horizons = horizons
        self.results = {}
    
    def calculate_metrics(self, y_true, y_pred):
        """Calculate comprehensive metrics"""
        # Flatten if multi-dimensional
        y_true_flat = y_true.flatten() if len(y_true.shape) > 1 else y_true
        y_pred_flat = y_pred.flatten() if len(y_pred.shape) > 1 else y_pred
        
        # Remove any NaN values
        mask = ~(np.isnan(y_true_flat) | np.isnan(y_pred_flat))
        y_true_clean = y_true_flat[mask]
        y_pred_clean = y_pred_flat[mask]
        
        if len(y_true_clean) == 0:
            return {
                'mse': np.nan, 'rmse': np.nan, 'mae': np.nan,
                'r2': np.nan, 'mape': np.nan, 'nrmse': np.nan
            }
        
        mse = mean_squared_error(y_true_clean, y_pred_clean)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true_clean, y_pred_clean)
        r2 = r2_score(y_true_clean, y_pred_clean)
        mape = mean_absolute_percentage_error(y_true_clean, y_pred_clean)
        
        # Normalized RMSE (by mean of y_true)
        nrmse = rmse / np.mean(np.abs(y_true_clean)) if np.mean(np.abs(y_true_clean)) > 0 else np.nan
        
        return {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'mape': float(mape),
            'nrmse': float(nrmse),
            'n_samples': len(y_true_clean)
        }
    
    def evaluate_by_horizon(self):
        """Evaluate each model across multiple horizons"""
        print("\n" + "="*80)
        print("MULTI-HORIZON EVALUATION")
        print("="*80)
        
        # Handle different input shapes
        if len(self.y_test.shape) > 1:
            # Multi-site: average across sites
            y_test_eval = np.mean(self.y_test, axis=1)
        else:
            y_test_eval = self.y_test
        
        for model_name, y_pred in self.y_pred_dict.items():
            print(f"\n[{model_name}] Evaluating across horizons...")
            
            horizon_results = {}
            
            # Determine number of samples we can evaluate
            n_samples = min(len(y_test_eval), len(y_pred.flatten()))
            
            for horizon in self.horizons:
                if horizon > n_samples:
                    print(f"  ⚠ Horizon {horizon}: Not enough samples ({n_samples})")
                    continue
                
                # Extract horizon-specific predictions
                indices = np.arange(0, n_samples - horizon, horizon)
                
                if len(indices) == 0:
                    print(f"  ⚠ Horizon {horizon}: Insufficient sample pairs")
                    continue
                
                y_true_h = y_test_eval[indices]
                y_pred_h = y_pred.flatten()[indices]
                
                metrics = self.calculate_metrics(y_true_h, y_pred_h)
                horizon_results[horizon] = metrics
                
                print(f"  Horizon {horizon}h: RMSE={metrics['rmse']:.6f}, "
                      f"MAE={metrics['mae']:.6f}, R²={metrics['r2']:.6f}")
            
            self.results[model_name] = horizon_results
        
        return self.results
    
    def evaluate_overall(self):
        """Evaluate models on overall test set"""
        print("\n" + "="*80)
        print("OVERALL MODEL EVALUATION (Full Test Set)")
        print("="*80)
        
        overall_results = {}
        
        # Handle different input shapes
        if len(self.y_test.shape) > 1:
            y_test_eval = np.mean(self.y_test, axis=1)
        else:
            y_test_eval = self.y_test
        
        for model_name, y_pred in self.y_pred_dict.items():
            print(f"\n[{model_name}]")
            
            # Ensure y_pred has compatible shape
            if len(y_pred.shape) > 1:
                # Multi-site predictions: average across sites
                y_pred_eval = np.mean(y_pred, axis=1)
            else:
                y_pred_eval = y_pred.flatten()
            
            # Ensure shapes match
            if len(y_test_eval) != len(y_pred_eval):
                print(f"  ⚠ Shape mismatch: y_test_eval {y_test_eval.shape} vs y_pred_eval {y_pred_eval.shape}")
                # Truncate to minimum length
                min_len = min(len(y_test_eval), len(y_pred_eval))
                y_test_eval = y_test_eval[:min_len]
                y_pred_eval = y_pred_eval[:min_len]
                print(f"  ✓ Truncated to {min_len} samples")
            
            metrics = self.calculate_metrics(y_test_eval, y_pred_eval)
            
            print(f"  MSE:   {metrics['mse']:.6f}")
            print(f"  RMSE:  {metrics['rmse']:.6f}")
            print(f"  MAE:   {metrics['mae']:.6f}")
            print(f"  R²:    {metrics['r2']:.6f}")
            print(f"  MAPE:  {metrics['mape']:.6f}%")
            print(f"  NRMSE: {metrics['nrmse']:.6f}")
            
            overall_results[model_name] = metrics
        
        return overall_results
    
    def create_comparison_dataframe(self, overall_results):
        """Create comparison dataframe"""
        df_data = []
        for model_name, metrics in overall_results.items():
            df_data.append({
                'Model': model_name,
                'MSE': metrics['mse'],
                'RMSE': metrics['rmse'],
                'MAE': metrics['mae'],
                'R²': metrics['r2'],
                'MAPE': metrics['mape'],
                'NRMSE': metrics['nrmse']
            })
        
        df = pd.DataFrame(df_data)
        df = df.set_index('Model').round(6)
        return df.sort_values('RMSE')
    
    def plot_horizon_results(self, output_dir='results'):
        """Plot horizon-specific results"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract metrics for each horizon
        horizon_data = {}
        
        for model_name, horizon_results in self.results.items():
            for horizon, metrics in horizon_results.items():
                if horizon not in horizon_data:
                    horizon_data[horizon] = {}
                horizon_data[horizon][model_name] = metrics
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        horizons = sorted(horizon_data.keys())
        rmse_data = {h: [] for h in horizons}
        mae_data = {h: [] for h in horizons}
        r2_data = {h: [] for h in horizons}
        mape_data = {h: [] for h in horizons}
        
        model_names = list(self.results.keys())
        
        for model_name in model_names:
            for h in horizons:
                if h in self.results[model_name]:
                    metrics = self.results[model_name][h]
                    rmse_data[h].append(metrics['rmse'])
                    mae_data[h].append(metrics['mae'])
                    r2_data[h].append(metrics['r2'])
                    mape_data[h].append(metrics['mape'])
        
        # RMSE
        for i, h in enumerate(horizons):
            axes[0, 0].plot([model_names[j] for j in range(len(model_names))], 
                          rmse_data[h], marker='o', label=f'Horizon {h}h', linewidth=2)
        axes[0, 0].set_title('RMSE across Horizons', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('RMSE')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # MAE
        for i, h in enumerate(horizons):
            axes[0, 1].plot([model_names[j] for j in range(len(model_names))], 
                          mae_data[h], marker='s', label=f'Horizon {h}h', linewidth=2)
        axes[0, 1].set_title('MAE across Horizons', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # R²
        for i, h in enumerate(horizons):
            axes[1, 0].plot([model_names[j] for j in range(len(model_names))], 
                          r2_data[h], marker='^', label=f'Horizon {h}h', linewidth=2)
        axes[1, 0].set_title('R² across Horizons', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('R² Score')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # MAPE
        for i, h in enumerate(horizons):
            axes[1, 1].plot([model_names[j] for j in range(len(model_names))], 
                          mape_data[h], marker='d', label=f'Horizon {h}h', linewidth=2)
        axes[1, 1].set_title('MAPE across Horizons', fontsize=12, fontweight='bold')
        axes[1, 1].set_ylabel('MAPE (%)')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'multi_horizon_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Multi-horizon comparison plot saved: {plot_path}")
        plt.close()
    
    def plot_model_comparison(self, overall_results, output_dir='results'):
        """Plot overall model comparison"""
        os.makedirs(output_dir, exist_ok=True)
        
        df = self.create_comparison_dataframe(overall_results)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # RMSE
        axes[0, 0].barh(df.index, df['RMSE'], color='steelblue')
        axes[0, 0].set_title('RMSE Comparison (Lower is Better)', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('RMSE')
        for i, v in enumerate(df['RMSE']):
            axes[0, 0].text(v + 0.001, i, f'{v:.6f}', va='center', fontsize=9)
        
        # MAE
        axes[0, 1].barh(df.index, df['MAE'], color='coral')
        axes[0, 1].set_title('MAE Comparison (Lower is Better)', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('MAE')
        for i, v in enumerate(df['MAE']):
            axes[0, 1].text(v + 0.0001, i, f'{v:.6f}', va='center', fontsize=9)
        
        # R²
        axes[1, 0].barh(df.index, df['R²'], color='mediumseagreen')
        axes[1, 0].set_title('R² Comparison (Higher is Better)', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('R² Score')
        for i, v in enumerate(df['R²']):
            axes[1, 0].text(v + 0.01, i, f'{v:.6f}', va='center', fontsize=9)
        
        # MAPE
        axes[1, 1].barh(df.index, df['MAPE'], color='mediumpurple')
        axes[1, 1].set_title('MAPE Comparison (Lower is Better)', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('MAPE (%)')
        for i, v in enumerate(df['MAPE']):
            axes[1, 1].text(v + 0.1, i, f'{v:.2f}%', va='center', fontsize=9)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'model_comparison_overall.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Model comparison plot saved: {plot_path}")
        plt.close()
        
        # Save comparison table
        csv_path = os.path.join(output_dir, 'model_comparison_metrics.csv')
        df.to_csv(csv_path)
        print(f"✓ Comparison metrics saved: {csv_path}")
        print("\n" + str(df))
        
        return df
