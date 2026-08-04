#!/usr/bin/env python3
"""
Verify optimized FLiESANN models against reference outputs.

This script tests optimized models to ensure they produce outputs within
acceptable tolerance of the original model (rtol=5e-4, atol=1e-4).

Usage:
    python -m FLiESANN.verify_optimized_model [model_name]
    python -m FLiESANN.verify_optimized_model slim
    python -m FLiESANN.verify_optimized_model --all
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

from .ECOv002_calval_FLiESANN_inputs import load_ECOv002_calval_FLiESANN_inputs
from .ECOv002_calval_FLiESANN_outputs import load_ECOv002_calval_FLiESANN_outputs
from .process_FLiESANN_table import process_FLiESANN_table


def verify_model(model_filename: str = None) -> bool:
    """
    Verify model outputs against reference dataset.
    
    Args:
        model_filename: Path to model file (None = use default FLiESANN.h5)
        
    Returns:
        True if verification passes, False otherwise
    """
    print("=" * 80)
    if model_filename:
        print(f"VERIFYING MODEL: {Path(model_filename).name}")
    else:
        print("VERIFYING DEFAULT MODEL: FLiESANN.h5")
    print("=" * 80)
    
    # Load input and output tables
    print("\n📁 Loading reference data...")
    input_df = load_ECOv002_calval_FLiESANN_inputs()
    output_df = load_ECOv002_calval_FLiESANN_outputs()
    print(f"   Loaded {len(input_df):,} reference samples")
    
    # Run the model on the input table
    print("\n🔄 Running model inference...")
    model_df = process_FLiESANN_table(input_df, offline_mode=True, model_filename=model_filename)
    print(f"   Generated {len(model_df):,} predictions")
    
    # Columns to compare (model outputs)
    output_columns = [
        "SWin_Wm2",
        "PAR_diffuse_Wm2",
        "PAR_direct_Wm2",
        "NIR_diffuse_Wm2",
        "NIR_direct_Wm2",
        "UV_Wm2"
    ]
    
    print("\n🔍 Comparing outputs...")
    print("   Tolerance: rtol=5e-4 (0.05%), atol=1e-4")
    
    # Compare each output column
    mismatches = []
    for col in output_columns:
        if col not in model_df or col not in output_df:
            mismatches.append((col, 'missing_column', None))
            continue
            
        model_vals = model_df[col].values
        ref_vals = output_df[col].values
        
        # Ensure values are numeric
        try:
            model_vals = pd.to_numeric(model_vals, errors='coerce')
            ref_vals = pd.to_numeric(ref_vals, errors='coerce')
        except:
            mismatches.append((col, 'value_mismatch', {'type': 'string_mismatch'}))
            continue
        
        # Use numpy allclose for comparison
        if not np.allclose(model_vals, ref_vals, rtol=5e-4, atol=1e-4, equal_nan=True):
            diffs = np.abs(model_vals - ref_vals)
            max_diff = np.nanmax(diffs) if not np.all(np.isnan(diffs)) else np.nan
            mean_diff = np.nanmean(diffs)
            idxs = np.where(~np.isclose(model_vals, ref_vals, rtol=5e-4, atol=1e-4, equal_nan=True))[0]
            
            mismatch_info = {
                'count': len(idxs),
                'max_diff': float(max_diff),
                'mean_diff': float(mean_diff),
                'sample_indices': idxs[:5].tolist(),
                'sample_model': model_vals[idxs[:5]].tolist(),
                'sample_ref': ref_vals[idxs[:5]].tolist(),
            }
            mismatches.append((col, 'value_mismatch', mismatch_info))
        else:
            print(f"   ✅ {col:<20s} - PASS")
    
    # Print results
    if mismatches:
        print("\n❌ VERIFICATION FAILED\n")
        print("Details:")
        for col, reason, info in mismatches:
            if reason == 'missing_column':
                print(f"\n  ❌ {col}: Missing column")
            elif reason == 'value_mismatch':
                if info.get('type') == 'string_mismatch':
                    print(f"\n  ❌ {col}: Type mismatch")
                else:
                    print(f"\n  ❌ {col}: Value mismatch")
                    print(f"     Mismatched values: {info['count']:,} / {len(model_vals):,} "
                          f"({info['count']/len(model_vals)*100:.2f}%)")
                    print(f"     Max difference:    {info['max_diff']:.6f}")
                    print(f"     Mean difference:   {info['mean_diff']:.6f}")
                    if len(info['sample_indices']) > 0:
                        print(f"     Sample mismatches (first 5):")
                        for i, (idx, m_val, r_val) in enumerate(zip(
                            info['sample_indices'], 
                            info['sample_model'], 
                            info['sample_ref']
                        )):
                            diff = abs(m_val - r_val)
                            print(f"       [{idx}] model={m_val:.6f}, ref={r_val:.6f}, diff={diff:.6f}")
        return False
    else:
        print("\n✅ VERIFICATION PASSED")
        print("   All outputs match reference values within tolerance")
        return True


def find_optimized_models():
    """Find all optimized model files."""
    model_dir = Path(__file__).parent
    pattern = "FLiESANN_*.h5"
    models = sorted(model_dir.glob(pattern))
    return models


def verify_all_models():
    """Verify all available optimized models."""
    models = find_optimized_models()
    
    if not models:
        print("No optimized models found. Run optimize_model.py first.")
        return
    
    print(f"\nFound {len(models)} optimized model(s)")
    
    results = []
    for model_path in models:
        result = verify_model(str(model_path))
        model_name = model_path.stem.replace('FLiESANN_', '')
        results.append((model_name, result))
        print()
    
    # Summary
    print("=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    print(f"\n{'Model':<20} {'Status'}")
    print("─" * 40)
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name:<20} {status}")
    
    passed_count = sum(1 for _, p in results if p)
    print(f"\nPassed: {passed_count}/{len(results)}")


def main():
    parser = argparse.ArgumentParser(description='Verify optimized FLiESANN models')
    parser.add_argument('model', nargs='?', default=None,
                       help='Model name (slim, minimal, micro, balanced) or path to .h5 file')
    parser.add_argument('--all', action='store_true',
                       help='Verify all optimized models')
    args = parser.parse_args()
    
    if args.all:
        verify_all_models()
    elif args.model:
        # Check if it's a model name or path
        if args.model.endswith('.h5'):
            model_path = args.model
        else:
            model_dir = Path(__file__).parent
            model_path = model_dir / f"FLiESANN_{args.model}.h5"
            
        if not Path(model_path).exists():
            print(f"Error: Model file not found: {model_path}")
            sys.exit(1)
            
        success = verify_model(str(model_path))
        sys.exit(0 if success else 1)
    else:
        # Verify default model
        success = verify_model()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
