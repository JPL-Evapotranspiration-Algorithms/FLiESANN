"""
Performance measurement script for FLiESANN table processing.

This script loads the ECOv002 calibration/validation dataset and measures
the execution time of the process_FLiESANN_table function.
"""

import time
import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path to import FLiESANN
sys.path.insert(0, str(Path(__file__).parent.parent))

from FLiESANN.process_FLiESANN_table import process_FLiESANN_table


def format_time(seconds):
    """Format time in seconds to a human-readable string."""
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f} minutes ({seconds:.2f} seconds)"
    else:
        hours = seconds / 3600
        minutes = (seconds % 3600) / 60
        return f"{hours:.2f} hours ({minutes:.2f} minutes)"


def main():
    # Define paths
    project_root = Path(__file__).parent.parent
    input_csv = project_root / "FLiESANN" / "ECOv002-cal-val-FLiESANN-inputs.csv"
    
    print("=" * 70)
    print("FLiESANN Performance Measurement")
    print("=" * 70)
    print(f"\nInput file: {input_csv}")
    
    # Check if file exists
    if not input_csv.exists():
        print(f"ERROR: Input file not found: {input_csv}")
        sys.exit(1)
    
    # Load the dataset
    print("\nLoading dataset...")
    load_start = time.time()
    input_df = pd.read_csv(input_csv)
    load_time = time.time() - load_start
    
    print(f"  ✓ Loaded {len(input_df)} rows in {load_time:.2f} seconds")
    print(f"  Columns: {len(input_df.columns)}")
    print(f"  Memory usage: {input_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Process the table with timing
    print("\nProcessing FLiESANN table (offline mode)...")
    print("  This may take a while...")
    
    process_start = time.time()
    output_df = process_FLiESANN_table(input_df, offline_mode=True)
    process_time = time.time() - process_start
    
    # Calculate statistics
    total_time = load_time + process_time
    time_per_row = process_time / len(input_df)
    rows_per_second = len(input_df) / process_time
    
    # Display results
    print("\n" + "=" * 70)
    print("PERFORMANCE RESULTS")
    print("=" * 70)
    print(f"\nDataset Statistics:")
    print(f"  Input rows:       {len(input_df):,}")
    print(f"  Output rows:      {len(output_df):,}")
    print(f"  Output columns:   {len(output_df.columns)}")
    
    print(f"\nTiming Results:")
    print(f"  Loading time:     {format_time(load_time)}")
    print(f"  Processing time:  {format_time(process_time)}")
    print(f"  Total time:       {format_time(total_time)}")
    
    print(f"\nThroughput:")
    print(f"  Time per row:     {time_per_row * 1000:.2f} ms")
    print(f"  Rows per second:  {rows_per_second:.2f}")
    
    print("\n" + "=" * 70)
    
    # Show sample of output columns
    new_columns = [col for col in output_df.columns if col not in input_df.columns]
    if new_columns:
        print(f"\nNew output columns ({len(new_columns)}):")
        for col in new_columns[:10]:  # Show first 10
            print(f"  - {col}")
        if len(new_columns) > 10:
            print(f"  ... and {len(new_columns) - 10} more")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
