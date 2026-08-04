#!/usr/bin/env python3
"""
FLiESANN Model Optimization via Knowledge Distillation

This script creates optimized (slimmer) versions of the FLiESANN model by training
student networks to mimic the original teacher network's outputs. The optimized models
run faster while maintaining accuracy within verification tolerances.

Usage:
    python -m FLiESANN.optimize_model [--test]
    
Options:
    --test    Use only 1000 samples for quick testing
"""

import os
import sys
import warnings
from pathlib import Path
import argparse

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

# Import FLiESANN components
from .load_FLiESANN_model import load_FLiESANN_model
from .ECOv002_calval_FLiESANN_inputs import load_ECOv002_calval_FLiESANN_inputs
from .prepare_FLiESANN_inputs import prepare_FLiESANN_inputs


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def inspect_original_model():
    """Load and inspect the original FLiESANN model."""
    print_section("INSPECTING ORIGINAL FLiESANN MODEL")
    
    model = load_FLiESANN_model()
    
    print("\n📊 MODEL SUMMARY:")
    print("-" * 80)
    model.summary()
    
    total_params = model.count_params()
    
    print("\n📈 STATISTICS:")
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Input dimension:      {model.input_shape[-1]}")
    print(f"  Output dimension:     {model.output_shape[-1]}")
    
    # Get model file size
    model_path = Path(__file__).parent / "FLiESANN.h5"
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 ** 2)
        print(f"  Model file size:      {size_mb:.2f} MB")
    
    print("\n🏗️  LAYER ARCHITECTURE:")
    for i, layer in enumerate(model.layers):
        layer_params = layer.count_params()
        config = layer.get_config()
        activation = config.get('activation', 'none')
        units = getattr(layer, 'units', 'N/A')
        print(f"  [{i}] {layer.name:20s} units={str(units):>4s} "
              f"activation={activation:10s} params={layer_params:,}")
    
    return model, total_params


def create_optimized_architectures(input_dim=14, output_dim=7):
    """Create multiple optimized model architectures for testing."""
    print_section("CREATING OPTIMIZED MODEL ARCHITECTURES")
    
    architectures = {}
    
    # Architecture 1: SLIM - 50% parameter reduction
    print("\n🔹 Architecture: SLIM (50% reduction)")
    print("   Layers: 14 → 80 → 40 → 7")
    model_slim = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(80, activation='relu', name='hidden1'),
        keras.layers.Dense(40, activation='relu', name='hidden2'),
        keras.layers.Dense(output_dim, activation='linear', name='output')
    ], name='FLiESANN_slim')
    params_slim = model_slim.count_params()
    architectures['slim'] = model_slim
    print(f"   Parameters: {params_slim:,}")
    
    # Architecture 2: MINIMAL - 70% parameter reduction
    print("\n🔹 Architecture: MINIMAL (70% reduction)")
    print("   Layers: 14 → 48 → 24 → 7")
    model_minimal = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(48, activation='relu', name='hidden1'),
        keras.layers.Dense(24, activation='relu', name='hidden2'),
        keras.layers.Dense(output_dim, activation='linear', name='output')
    ], name='FLiESANN_minimal')
    params_minimal = model_minimal.count_params()
    architectures['minimal'] = model_minimal
    print(f"   Parameters: {params_minimal:,}")
    
    # Architecture 3: MICRO - 85% parameter reduction
    print("\n🔹 Architecture: MICRO (85% reduction)")
    print("   Layers: 14 → 32 → 16 → 7")
    model_micro = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(32, activation='relu', name='hidden1'),
        keras.layers.Dense(16, activation='relu', name='hidden2'),
        keras.layers.Dense(output_dim, activation='linear', name='output')
    ], name='FLiESANN_micro')
    params_micro = model_micro.count_params()
    architectures['micro'] = model_micro
    print(f"   Parameters: {params_micro:,}")
    
    # Architecture 4: BALANCED - 60% reduction with deeper network
    print("\n🔹 Architecture: BALANCED (60% reduction, 3 hidden layers)")
    print("   Layers: 14 → 64 → 32 → 16 → 7")
    model_balanced = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(64, activation='relu', name='hidden1'),
        keras.layers.Dense(32, activation='relu', name='hidden2'),
        keras.layers.Dense(16, activation='relu', name='hidden3'),
        keras.layers.Dense(output_dim, activation='linear', name='output')
    ], name='FLiESANN_balanced')
    params_balanced = model_balanced.count_params()
    architectures['balanced'] = model_balanced
    print(f"   Parameters: {params_balanced:,}")
    
    return architectures


def prepare_training_data(teacher_model, sample_size=None, test_mode=False):
    """
    Load calibration/validation data and generate teacher model outputs.
    
    Args:
        teacher_model: Original FLiESANN model to generate target outputs
        sample_size: Number of samples to use (None = all data)
        test_mode: If True, use only 1000 samples for quick testing
        
    Returns:
        X_train, Y_train, X_val, Y_val: Training and validation datasets
    """
    print_section("PREPARING TRAINING DATA")
    
    print("\n📁 Loading ECOv002 calibration/validation dataset...")
    df = load_ECOv002_calval_FLiESANN_inputs()
    print(f"   Loaded {len(df):,} samples")
    
    # Sample if requested
    if test_mode:
        sample_size = 1000
        print(f"   🧪 TEST MODE: Using only {sample_size:,} samples")
    
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
        print(f"   Sampled down to {len(df):,} samples")
    
    # Prepare input features
    print("\n🔧 Preparing input features...")
    
    # Check if we need to prepare features
    required_features = ['ctype0', 'ctype1', 'ctype3', 'atype1', 'atype2', 
                        'atype4', 'atype5', 'COT', 'AOT', 'vapor_gccm', 
                        'ozone_cm', 'albedo', 'elevation_km', 'SZA']
    
    if all(f in df.columns for f in required_features):
        print("   ✓ Input features already prepared")
        X = df[required_features].values.astype(np.float32)
    else:
        print("   ⚙️  Preparing features from raw data...")
        # Use prepare_FLiESANN_inputs to create features
        inputs_df = prepare_FLiESANN_inputs(
            atype=df['atype'].values if 'atype' in df.columns else df['atype1'].values,
            ctype=df['ctype'].values if 'ctype' in df.columns else df['ctype0'].values,
            COT=df['COT'].values,
            AOT=df['AOT'].values,
            vapor_gccm=df['vapor_gccm'].values,
            ozone_cm=df['ozone_cm'].values,
            albedo=df['albedo'].values,
            elevation_km=df['elevation_km'].values,
            SZA=df['SZA'].values,
            split_atypes_ctypes=True
        )
        X = inputs_df.values.astype(np.float32)
    
    print(f"   Input shape: {X.shape}")
    
    # Generate teacher outputs (knowledge distillation targets)
    print("\n🎓 Generating teacher model outputs...")
    print("   (This may take a moment...)")
    
    # Reshape for model if needed
    original_shape = X.shape
    if len(teacher_model.input_shape) == 3:
        X_reshaped = X.reshape(X.shape[0], 1, X.shape[1])
    else:
        X_reshaped = X
    
    Y = teacher_model.predict(X_reshaped, batch_size=512, verbose=0)
    
    # Flatten if needed
    if len(Y.shape) == 3:
        Y = Y.squeeze(axis=1)
    
    Y = Y.astype(np.float32)
    print(f"   Output shape: {Y.shape}")
    
    # Check for NaN values
    nan_rows = np.any(np.isnan(X), axis=1) | np.any(np.isnan(Y), axis=1)
    if np.any(nan_rows):
        print(f"   ⚠️  Removing {np.sum(nan_rows):,} rows with NaN values")
        X = X[~nan_rows]
        Y = Y[~nan_rows]
    
    # Split into training and validation sets (80/20)
    split_idx = int(0.8 * len(X))
    
    # Shuffle data
    indices = np.random.RandomState(42).permutation(len(X))
    X = X[indices]
    Y = Y[indices]
    
    X_train = X[:split_idx]
    Y_train = Y[:split_idx]
    X_val = X[split_idx:]
    Y_val = Y[split_idx:]
    
    print(f"\n📊 Dataset split:")
    print(f"   Training:   {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"   Validation: {len(X_val):,} samples ({len(X_val)/len(X)*100:.1f}%)")
    
    # Show data statistics
    print(f"\n📈 Output statistics (teacher model):")
    print(f"   Mean: {Y.mean(axis=0)}")
    print(f"   Std:  {Y.std(axis=0)}")
    print(f"   Min:  {Y.min(axis=0)}")
    print(f"   Max:  {Y.max(axis=0)}")
    
    return X_train, Y_train, X_val, Y_val


def train_student_model(model, X_train, Y_train, X_val, Y_val, 
                        model_name, epochs=200, patience=15):
    """
    Train a student model via knowledge distillation.
    
    Args:
        model: Student model to train
        X_train, Y_train: Training data
        X_val, Y_val: Validation data
        model_name: Name for logging
        epochs: Maximum training epochs
        patience: Early stopping patience
        
    Returns:
        Trained model and training history
    """
    print(f"\n🏋️  Training {model_name.upper()} model...")
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae', 'mse']
    )
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=0
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=0
        ),
        keras.callbacks.TerminateOnNaN()
    ]
    
    # Train
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=epochs,
        batch_size=128,
        callbacks=callbacks,
        verbose=0
    )
    
    # Get final metrics
    final_epoch = len(history.history['loss'])
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    final_train_mae = history.history['mae'][-1]
    final_val_mae = history.history['val_mae'][-1]
    
    print(f"   ✓ Training complete after {final_epoch} epochs")
    print(f"   Train Loss: {final_train_loss:.6f}, MAE: {final_train_mae:.6f}")
    print(f"   Val Loss:   {final_val_loss:.6f}, MAE: {final_val_mae:.6f}")
    
    # Check quality threshold
    if final_val_mae > 0.001:
        print(f"   ⚠️  WARNING: Validation MAE ({final_val_mae:.6f}) exceeds quality threshold (0.001)")
    else:
        print(f"   ✅ Model meets quality threshold (MAE < 0.001)")
    
    return model, history


def save_optimized_model(model, model_name, base_path="FLiESANN"):
    """Save optimized model to disk."""
    base_path = Path(__file__).parent
    output_path = base_path / f"FLiESANN_{model_name}.h5"
    
    model.save(output_path, save_format='h5')
    
    size_mb = output_path.stat().st_size / (1024 ** 2)
    print(f"   💾 Saved: {output_path.name} ({size_mb:.2f} MB)")
    
    return output_path


def compare_predictions(teacher_model, student_model, X_val, Y_val):
    """Compare teacher and student model predictions."""
    print("\n🔍 Comparing predictions...")
    
    # Get predictions
    if len(teacher_model.input_shape) == 3:
        X_val_reshaped = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
    else:
        X_val_reshaped = X_val
    
    Y_pred_student = student_model.predict(X_val_reshaped, verbose=0)
    if len(Y_pred_student.shape) == 3:
        Y_pred_student = Y_pred_student.squeeze(axis=1)
    
    # Calculate differences
    mae = np.mean(np.abs(Y_val - Y_pred_student), axis=0)
    max_error = np.max(np.abs(Y_val - Y_pred_student), axis=0)
    
    output_names = ['atm_trans', 'UV_prop', 'PAR_prop', 'NIR_prop', 
                   'UV_diff', 'PAR_diff', 'NIR_diff']
    
    print("\n   Per-output MAE:")
    for i, name in enumerate(output_names):
        print(f"   {name:12s}: MAE={mae[i]:.6f}, Max Error={max_error[i]:.6f}")
    
    print(f"\n   Overall MAE: {mae.mean():.6f}")
    print(f"   Overall Max Error: {max_error.max():.6f}")
    
    # Check if within verification tolerances
    rtol = 5e-4
    atol = 1e-4
    within_tolerance = np.allclose(Y_val, Y_pred_student, rtol=rtol, atol=atol)
    
    if within_tolerance:
        print(f"   ✅ All predictions within verification tolerances (rtol={rtol}, atol={atol})")
    else:
        mismatch_count = np.sum(~np.isclose(Y_val, Y_pred_student, rtol=rtol, atol=atol, equal_nan=True))
        print(f"   ⚠️  {mismatch_count:,} predictions exceed verification tolerances")
    
    return within_tolerance


def main():
    """Main execution flow."""
    parser = argparse.ArgumentParser(description='Optimize FLiESANN model via knowledge distillation')
    parser.add_argument('--test', action='store_true', 
                       help='Use only 1000 samples for quick testing')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Maximum training epochs (default: 200)')
    parser.add_argument('--sample-size', type=int, default=None,
                       help='Number of samples to use (default: all)')
    args = parser.parse_args()
    
    print("\n🚀 FLiESANN Model Optimization Tool")
    print("   Using Knowledge Distillation\n")
    
    # Step 1: Inspect original model
    teacher_model, original_params = inspect_original_model()
    
    # Step 2: Create optimized architectures
    architectures = create_optimized_architectures()
    
    # Step 3: Prepare training data
    X_train, Y_train, X_val, Y_val = prepare_training_data(
        teacher_model, 
        sample_size=args.sample_size,
        test_mode=args.test
    )
    
    # Step 4: Train each architecture
    print_section("TRAINING OPTIMIZED MODELS")
    
    trained_models = {}
    results = []
    
    for name, model in architectures.items():
        print(f"\n{'─' * 80}")
        student_model, history = train_student_model(
            model, X_train, Y_train, X_val, Y_val,
            model_name=name,
            epochs=args.epochs,
            patience=15
        )
        
        # Compare predictions
        within_tolerance = compare_predictions(teacher_model, student_model, X_val, Y_val)
        
        # Save model
        output_path = save_optimized_model(student_model, name)
        
        trained_models[name] = student_model
        
        # Store results
        results.append({
            'name': name,
            'params': model.count_params(),
            'reduction': (1 - model.count_params() / original_params) * 100,
            'val_mae': history.history['val_mae'][-1],
            'within_tolerance': within_tolerance,
            'file': output_path.name
        })
    
    # Step 5: Summary
    print_section("OPTIMIZATION SUMMARY")
    
    print("\n📊 Results:")
    print(f"\n{'Model':<15} {'Params':<10} {'Reduction':<12} {'Val MAE':<12} {'≈Teacher':<10} {'File'}")
    print("─" * 85)
    
    for r in results:
        tolerance_mark = "✅" if r['within_tolerance'] else "⚠️ "
        print(f"{r['name']:<15} {r['params']:<10,} {r['reduction']:>5.1f}%      "
              f"{r['val_mae']:<12.6f} {tolerance_mark:<10} {r['file']}")
    
    print("\n📝 Next Steps:")
    print("   1. Run verification: python -m FLiESANN.verify")
    print("   2. Benchmark performance: python performance/measure_processing_time.py")
    print("   3. Update load_FLiESANN_model.py to use optimized model")
    print("   4. Consider applying XLA compilation for additional speedup")
    
    print("\n✨ Optimization complete!\n")
    
    return trained_models, results


if __name__ == "__main__":
    main()
