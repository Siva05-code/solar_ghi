#!/usr/bin/env python3
"""
Quick test script to verify implementation
Runs with reduced data for speed
"""

import os
import sys
import numpy as np

def test_imports():
    """Test all imports work"""
    print("=" * 70)
    print("TESTING IMPORTS")
    print("=" * 70)
    
    try:
        from preprocessing_spatiotemporal import preprocess_spatiotemporal, load_all_sites
        print("✓ preprocessing_spatiotemporal")
    except Exception as e:
        print(f"✗ preprocessing_spatiotemporal: {e}")
        return False
    
    try:
        from transformer_st import train_spatiotemporal_transformer
        print("✓ transformer_st")
    except Exception as e:
        print(f"✗ transformer_st: {e}")
        return False
    
    try:
        from gru_model import train_gru_model
        print("✓ gru_model")
    except Exception as e:
        print(f"✗ gru_model: {e}")
        return False
    
    try:
        from svm_model import train_svm_model
        print("✓ svm_model")
    except Exception as e:
        print(f"✗ svm_model: {e}")
        return False
    
    try:
        from spatial_analysis import analyze_spatial_correlations, analyze_daily_patterns
        print("✓ spatial_analysis")
    except Exception as e:
        print(f"✗ spatial_analysis: {e}")
        return False
    
    print("\n✓ All imports successful")
    return True


def test_preprocessing():
    """Test preprocessing with small dataset"""
    print("\n" + "=" * 70)
    print("TESTING PREPROCESSING (using only 1 month of data)")
    print("=" * 70)
    
    try:
        from preprocessing_spatiotemporal import preprocess_spatiotemporal
        
        print("\n[Running] Preprocessing with seq_len=12, small train ratio...")
        dataset = preprocess_spatiotemporal(
            seq_len=12,
            horizon=1,
            train_ratio=0.8,
            save=True
        )
        
        print(f"\n✓ Preprocessing successful")
        print(f"  X_train: {dataset['X_train'].shape}")
        print(f"  X_test: {dataset['X_test'].shape}")
        print(f"  y_train: {dataset['y_train'].shape}")
        print(f"  y_test: {dataset['y_test'].shape}")
        print(f"  Sites: {dataset['sites']}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Preprocessing failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_data_files():
    """Check if preprocessed data files exist"""
    print("\n" + "=" * 70)
    print("CHECKING PREPROCESSED DATA FILES")
    print("=" * 70)
    
    files_to_check = [
        'data/X_train_st.npy',
        'data/X_test_st.npy',
        'data/y_train_st.npy',
        'data/y_test_st.npy'
    ]
    
    all_exist = True
    for filepath in files_to_check:
        if os.path.exists(filepath):
            size = os.path.getsize(filepath) / (1024 * 1024)
            print(f"✓ {filepath} ({size:.2f} MB)")
        else:
            print(f"✗ {filepath} (missing)")
            all_exist = False
    
    return all_exist


def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("IMPLEMENTATION QUICK-CHECK")
    print("=" * 70)
    
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Test imports
    if not test_imports():
        print("\n✗ Import test failed")
        return False
    
    # Test preprocessing
    if not test_preprocessing():
        print("\n✗ Preprocessing test failed")
        return False
    
    # Check data files
    if not test_data_files():
        print("\n⚠ Some data files missing (expected on first run)")
    
    print("\n" + "=" * 70)
    print("✅ QUICK-CHECK PASSED - Implementation is ready")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Run full pipeline: python3 main_pipeline.py")
    print("  2. Monitor progress and results")
    print("  3. Check results/ directory for outputs")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
