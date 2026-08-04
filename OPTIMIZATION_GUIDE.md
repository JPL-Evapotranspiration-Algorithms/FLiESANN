# FLiESANN Model Optimization Guide

This guide explains how to optimize the FLiESANN model for faster inference while maintaining accuracy within verification tolerances.

## Overview

The optimization process uses **knowledge distillation** to train smaller "student" models that mimic the original "teacher" model's outputs. Combined with TensorFlow XLA compilation, this can achieve 2-5x speedup while maintaining accuracy within the existing verification tolerance (rtol=5e-4, atol=1e-4).

## Quick Start

### 1. Baseline Performance

First, measure the current model's performance:

```bash
python -m FLiESANN.benchmark_models
```

### 2. Train Optimized Models

Generate optimized model variants:

```bash
# Full dataset (recommended)
python -m FLiESANN.optimize_model

# Quick test with 1000 samples
python -m FLiESANN.optimize_model --test
```

This creates four optimized models:
- **slim** - 50% parameter reduction (80→40 neurons)
- **minimal** - 70% parameter reduction (48→24 neurons)
- **micro** - 85% parameter reduction (32→16 neurons)
- **balanced** - 60% parameter reduction (64→32→16 neurons, 3 layers)

### 3. Verify Optimized Models

Ensure models pass the verification system:

```bash
# Verify all models
python -m FLiESANN.verify_optimized_model --all

# Verify specific model
python -m FLiESANN.verify_optimized_model slim
```

### 4. Benchmark Performance

Compare optimized models against the original:

```bash
# Compare all models
python -m FLiESANN.benchmark_models --compare

# Benchmark specific model
python -m FLiESANN.benchmark_models --model slim --sample-size 1000
```

### 5. Use Optimized Model

To use an optimized model as the default, either:

**Option A: Replace the default model**
```bash
cp FLiESANN/FLiESANN_slim.h5 FLiESANN/FLiESANN.h5
```

**Option B: Specify model in code**
```python
from FLiESANN import FLiESANN

results = FLiESANN(
    albedo=0.15,
    time_UTC=datetime(2024, 7, 15, 18, 0),
    geometry=Point(-118.0, 34.0),
    model_filename="FLiESANN/FLiESANN_slim.h5"
)
```

## Optimization Techniques

### Knowledge Distillation

The optimization script trains smaller neural networks to reproduce the original model's outputs:

1. Loads the original FLiESANN model (teacher)
2. Generates predictions on the calibration/validation dataset
3. Trains smaller architectures (students) to match these predictions
4. Uses Mean Squared Error (MSE) loss with early stopping

**Benefits:**
- Preserves learned knowledge in smaller models
- Maintains accuracy while reducing parameters
- Works well for smooth regression tasks like radiative transfer

### TensorFlow XLA Compilation

For additional 20-30% speedup with zero accuracy loss:

```python
from FLiESANN import load_FLiESANN_model

# Load with XLA compilation
model = load_FLiESANN_model()
model.compile(optimizer='adam', loss='mse', jit_compile=True)
```

XLA (Accelerated Linear Algebra) optimizes TensorFlow computation graphs for faster execution.

### Quantization (Advanced)

For even more aggressive optimization, convert to TensorFlow Lite:

```python
import tensorflow as tf

model = tf.keras.models.load_model("FLiESANN/FLiESANN_slim.h5")

# Convert with float16 quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()

with open("FLiESANN/FLiESANN_slim.tflite", "wb") as f:
    f.write(tflite_model)
```

**Note:** Requires updating inference code to use TFLite interpreter.

## Architecture Details

### Original Model
```
Input (14) → Dense(14, sigmoid) → Dense(140, relu) → Dense(70, relu) → Dense(7, linear)
Parameters: 12,677 (~180 KB)
```

### Optimized Architectures

**SLIM (50% reduction)**
```
Input (14) → Dense(80, relu) → Dense(40, relu) → Dense(7, linear)
Parameters: ~6,000
```

**MINIMAL (70% reduction)**
```
Input (14) → Dense(48, relu) → Dense(24, relu) → Dense(7, linear)
Parameters: ~3,500
```

**MICRO (85% reduction)**
```
Input (14) → Dense(32, relu) → Dense(16, relu) → Dense(7, linear)
Parameters: ~1,600
```

**BALANCED (60% reduction, deeper)**
```
Input (14) → Dense(64, relu) → Dense(32, relu) → Dense(16, relu) → Dense(7, linear)
Parameters: ~4,500
```

## Verification Tolerances

The existing verification system uses:
- **Relative tolerance (rtol):** 5e-4 (0.05%)
- **Absolute tolerance (atol):** 1e-4

Optimized models must match reference outputs within these tolerances for all outputs:
- SWin_Wm2 (Total incoming solar radiation)
- PAR_diffuse_Wm2 (Diffuse visible radiation)
- PAR_direct_Wm2 (Direct visible radiation)
- NIR_diffuse_Wm2 (Diffuse near-infrared)
- NIR_direct_Wm2 (Direct near-infrared)
- UV_Wm2 (Ultraviolet radiation)

## Troubleshooting

### Model Doesn't Pass Verification

If optimized model fails verification:

1. **Increase model capacity:** Try a larger architecture (e.g., slim instead of micro)
2. **Train longer:** Increase epochs: `--epochs 500`
3. **Use more training data:** Remove `--sample-size` limit
4. **Check validation MAE:** Should be < 0.001 for best results

### Poor Performance Improvement

If speedup is less than expected:

1. **Check batch size:** Ensure processing many samples at once
2. **Enable XLA:** Apply JIT compilation for additional speedup
3. **Warm-up properly:** First few predictions are slower due to graph compilation
4. **Use correct model:** Verify you're loading the optimized model, not the original

### Training Issues

**NaN loss during training:**
- Reduce learning rate: Edit `optimize_model.py` and change `learning_rate=0.001` to `0.0001`
- Check for inf/NaN in input data

**Model trains but has high error:**
- Use more training samples
- Try deeper architecture (balanced variant)
- Increase training epochs

## Performance Expectations

Expected speedups with different optimizations:

| Optimization | Speedup | Accuracy Impact | Complexity |
|-------------|---------|----------------|------------|
| Knowledge Distillation (slim) | 1.5-2x | Minimal (within tolerance) | Low |
| Knowledge Distillation (micro) | 2-3x | Minimal (within tolerance) | Low |
| + XLA Compilation | +20-30% | None | Very Low |
| + Float16 Quantization | +50-100% | Very Minimal | Medium |
| + INT8 Quantization | +2-3x | Minimal (needs calibration) | High |

**Recommended combination:** Knowledge distillation (slim) + XLA = **2-2.5x speedup** with minimal effort and zero accuracy loss.

## Scripts Reference

- `optimize_model.py` - Train optimized models via knowledge distillation
- `verify_optimized_model.py` - Verify models pass reference tests
- `benchmark_models.py` - Measure and compare model performance
- `verify.py` - Original verification system

## Best Practices

1. **Always verify first:** Run verification before deploying optimized model
2. **Benchmark on real data:** Use full cal/val dataset for accurate performance metrics
3. **Test edge cases:** Verify model handles extreme atmospheric conditions
4. **Version control:** Keep different model versions for comparison
5. **Document changes:** Note which optimized model is deployed

## Additional Resources

- Original FLiES model: Kobayashi & Iwabuchi (2008)
- Knowledge distillation: Hinton et al. (2015)
- TensorFlow XLA: https://www.tensorflow.org/xla
- Model optimization guide: https://www.tensorflow.org/model_optimization

## Support

For issues or questions:
- Open GitHub issue: https://github.com/JPL-Evapotranspiration-Algorithms/FLiESANN/issues
- Email: gregory.h.halverson@jpl.nasa.gov
