#!/usr/bin/env python3
"""
One-command script to optimize FLiESANN model.

This script runs the complete optimization workflow:
1. Trains optimized models via knowledge distillation
2. Verifies they pass the reference tests
3. Benchmarks performance improvements

Usage:
    python optimize_fliesann.py [--test] [--skip-benchmark]
    
Options:
    --test             Use only 1000 samples for quick testing
    --skip-benchmark   Skip the benchmark step (faster)
    --models MODELS    Comma-separated list of models to train (default: all)
"""

import sys
import argparse
import subprocess
from pathlib import Path


def run_command(cmd, description):
    """Run a command and report status."""
    print("\n" + "=" * 80)
    print(f"⚙️  {description}")
    print("=" * 80)
    print(f"Running: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    
    if result.returncode != 0:
        print(f"\n❌ {description} failed with exit code {result.returncode}")
        return False
    
    print(f"\n✅ {description} completed successfully")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Complete FLiESANN model optimization workflow',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full optimization (recommended)
  python optimize_fliesann.py
  
  # Quick test with 1000 samples
  python optimize_fliesann.py --test
  
  # Train only, skip benchmarking
  python optimize_fliesann.py --skip-benchmark
  
  # Train specific models only
  python optimize_fliesann.py --models slim,minimal
        """
    )
    parser.add_argument('--test', action='store_true',
                       help='Use only 1000 samples for quick testing')
    parser.add_argument('--skip-benchmark', action='store_true',
                       help='Skip performance benchmarking')
    parser.add_argument('--skip-verify', action='store_true',
                       help='Skip verification step')
    parser.add_argument('--models', type=str, default=None,
                       help='Comma-separated list of models (slim,minimal,micro,balanced)')
    args = parser.parse_args()
    
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║                   FLiESANN MODEL OPTIMIZATION WORKFLOW                   ║
║                                                                          ║
║  This will train smaller, faster models while maintaining accuracy      ║
║  within verification tolerances (rtol=5e-4, atol=1e-4)                  ║
╚══════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Step 1: Train optimized models
    cmd = [sys.executable, '-m', 'FLiESANN.optimize_model']
    if args.test:
        cmd.append('--test')
    if args.models:
        print(f"Note: Training only specified models: {args.models}")
    
    success = run_command(cmd, "STEP 1: Training Optimized Models")
    if not success:
        print("\n⚠️  Training failed. Cannot continue.")
        return 1
    
    # Step 2: Verify optimized models
    if not args.skip_verify:
        cmd = [sys.executable, '-m', 'FLiESANN.verify_optimized_model', '--all']
        success = run_command(cmd, "STEP 2: Verifying Optimized Models")
        if not success:
            print("\n⚠️  Verification failed for some models.")
            print("   You can still benchmark them, but they may not pass validation.")
            response = input("\n   Continue anyway? [y/N]: ")
            if response.lower() != 'y':
                return 1
    else:
        print("\n⏭️  Skipping verification (--skip-verify)")
    
    # Step 3: Benchmark performance
    if not args.skip_benchmark:
        cmd = [sys.executable, '-m', 'FLiESANN.benchmark_models', '--compare']
        if args.test:
            cmd.extend(['--sample-size', '1000'])
        success = run_command(cmd, "STEP 3: Benchmarking Performance")
        if not success:
            print("\n⚠️  Benchmarking failed, but optimized models were created.")
    else:
        print("\n⏭️  Skipping benchmark (--skip-benchmark)")
    
    # Summary
    print("\n" + "=" * 80)
    print("🎉 OPTIMIZATION COMPLETE!")
    print("=" * 80)
    
    print("\n📋 Next Steps:")
    print("   1. Review benchmark results above to choose best model")
    print("   2. Use optimized model in your code:")
    print("      FLiESANN(..., model_filename='FLiESANN/FLiESANN_slim.h5')")
    print("   3. Or replace default model:")
    print("      cp FLiESANN/FLiESANN_slim.h5 FLiESANN/FLiESANN.h5")
    print("\n📖 See OPTIMIZATION_GUIDE.md for detailed documentation")
    print("\n✨ Done!\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
