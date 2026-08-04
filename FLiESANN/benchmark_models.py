#!/usr/bin/env python3
"""
Benchmark performance of FLiESANN models.

This script measures inference speed of different model variants to compare
performance improvements from optimization.

Usage:
    python -m FLiESANN.benchmark_models [--sample-size N] [--model MODEL]
"""

import os
import sys
import argparse
import time
from pathlib import Path
import warnings

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from .ECOv002_calval_FLiESANN_inputs import load_ECOv002_calval_FLiESANN_inputs
from .process_FLiESANN_table import process_FLiESANN_table


def format_time(seconds):
    """Format time in seconds to human-readable string."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m ({seconds:.2f}s)"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h ({seconds/60:.2f}m)"


def benchmark_model(model_path, input_df, model_name="Model"):
    """
    Benchmark a single model's performance.
    
    Args:
        model_path: Path to model file (None = default)
        input_df: Input DataFrame
        model_name: Name for display
        
    Returns:
        dict with timing metrics
    """
    print(f"\n{'─' * 80}")
    print(f"🔬 Benchmarking: {model_name}")
    print(f"   Model: {Path(model_path).name if model_path else 'FLiESANN.h5 (default)'}")
    
    # Get model file size
    if model_path:
        size_mb = Path(model_path).stat().st_size / (1024 ** 2)
        print(f"   Size:  {size_mb:.2f} MB")
    
    # Warm-up run (to load model and compile TensorFlow graph)
    print(f"   Warming up...")
    warmup_df = input_df.head(10)
    _ = process_FLiESANN_table(warmup_df, offline_mode=True, model_filename=model_path)
    
    # Actual benchmark
    print(f"   Running benchmark on {len(input_df):,} samples...")
    
    start_time = time.time()
    output_df = process_FLiESANN_table(input_df, offline_mode=True, model_filename=model_path)
    end_time = time.time()
    
    elapsed = end_time - start_time
    throughput = len(input_df) / elapsed
    time_per_row = elapsed / len(input_df) * 1000  # in milliseconds
    
    print(f"   ✓ Complete")
    print(f"   Time:        {format_time(elapsed)}")
    print(f"   Throughput:  {throughput:.2f} rows/sec")
    print(f"   Per-row:     {time_per_row:.2f} ms/row")
    
    return {
        'name': model_name,
        'model_file': Path(model_path).name if model_path else 'FLiESANN.h5',
        'size_mb': size_mb if model_path else None,
        'samples': len(input_df),
        'elapsed_sec': elapsed,
        'throughput_rows_per_sec': throughput,
        'time_per_row_ms': time_per_row
    }


def find_all_models():
    """Find all available FLiESANN models."""
    model_dir = Path(__file__).parent
    
    models = []
    
    # Default model
    default_model = model_dir / "FLiESANN.h5"
    if default_model.exists():
        models.append(('original', None))  # None means use default
    
    # Optimized models
    for model_file in sorted(model_dir.glob("FLiESANN_*.h5")):
        name = model_file.stem.replace('FLiESANN_', '')
        models.append((name, str(model_file)))
    
    return models


def main():
    parser = argparse.ArgumentParser(description='Benchmark FLiESANN model performance')
    parser.add_argument('--sample-size', type=int, default=None,
                       help='Number of samples to benchmark (default: all)')
    parser.add_argument('--model', type=str, default=None,
                       help='Specific model to benchmark (default: all models)')
    parser.add_argument('--compare', action='store_true',
                       help='Compare all models')
    args = parser.parse_args()
    
    print("=" * 80)
    print("FLiESANN MODEL PERFORMANCE BENCHMARK")
    print("=" * 80)
    
    # Load input data
    print("\n📁 Loading input data...")
    input_df = load_ECOv002_calval_FLiESANN_inputs()
    print(f"   Loaded {len(input_df):,} samples")
    
    if args.sample_size:
        print(f"   Using subset: {args.sample_size:,} samples")
        input_df = input_df.head(args.sample_size)
    
    # Find models to benchmark
    all_models = find_all_models()
    
    if not all_models:
        print("❌ No models found!")
        return
    
    print(f"\n📊 Found {len(all_models)} model(s):")
    for name, path in all_models:
        print(f"   • {name}")
    
    # Benchmark models
    results = []
    
    if args.model:
        # Benchmark specific model
        model_found = False
        for name, path in all_models:
            if name == args.model:
                result = benchmark_model(path, input_df, name)
                results.append(result)
                model_found = True
                break
        
        if not model_found:
            print(f"\n❌ Model '{args.model}' not found")
            print(f"   Available: {', '.join([n for n, _ in all_models])}")
            return
    else:
        # Benchmark all models
        print("\n" + "=" * 80)
        print("BENCHMARKING ALL MODELS")
        print("=" * 80)
        
        for name, path in all_models:
            result = benchmark_model(path, input_df, name)
            results.append(result)
    
    # Summary comparison
    if len(results) > 1 or args.compare:
        print("\n" + "=" * 80)
        print("PERFORMANCE COMPARISON")
        print("=" * 80)
        
        # Sort by throughput (fastest first)
        results_sorted = sorted(results, key=lambda x: x['throughput_rows_per_sec'], reverse=True)
        
        # Get baseline (original model)
        baseline = next((r for r in results if r['name'] == 'original'), results_sorted[-1])
        baseline_throughput = baseline['throughput_rows_per_sec']
        
        print(f"\n{'Model':<15} {'Size':<10} {'Time':<12} {'Throughput':<18} {'Speedup':<10}")
        print("─" * 80)
        
        for r in results_sorted:
            speedup = r['throughput_rows_per_sec'] / baseline_throughput
            speedup_str = f"{speedup:.2f}x" if speedup != 1.0 else "baseline"
            size_str = f"{r['size_mb']:.1f} MB" if r['size_mb'] else "N/A"
            
            marker = "🏆 " if r == results_sorted[0] and len(results_sorted) > 1 else "   "
            
            print(f"{marker}{r['name']:<12} {size_str:<10} {format_time(r['elapsed_sec']):<12} "
                  f"{r['throughput_rows_per_sec']:>7.1f} rows/s   {speedup_str:<10}")
        
        print("\n📈 Performance Summary:")
        best = results_sorted[0]
        if best['name'] != 'original':
            improvement = (best['throughput_rows_per_sec'] / baseline_throughput - 1) * 100
            time_saved = baseline['elapsed_sec'] - best['elapsed_sec']
            print(f"   Best model: {best['name']}")
            print(f"   Speedup: {best['throughput_rows_per_sec']/baseline_throughput:.2f}x faster")
            print(f"   Improvement: {improvement:.1f}% faster than original")
            print(f"   Time saved: {format_time(time_saved)} on {len(input_df):,} samples")
            
            # Extrapolate to large datasets
            if len(input_df) < 10000:
                print(f"\n   📊 Extrapolated for larger datasets:")
                for size in [10000, 100000, 1000000]:
                    if size <= len(input_df):
                        continue
                    baseline_time = size / baseline_throughput
                    optimized_time = size / best['throughput_rows_per_sec']
                    saved = baseline_time - optimized_time
                    print(f"      {size:,} samples: save {format_time(saved)}")
        else:
            print(f"   Original model is still the fastest")
    
    print("\n" + "=" * 80)
    print("✨ Benchmark complete!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
