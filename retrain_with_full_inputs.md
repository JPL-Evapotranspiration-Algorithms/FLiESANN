# Retraining with Full Inputs

The current optimized models were trained on simplified inputs (default atype/ctype).
To achieve verification-passing accuracy, retrain with full atmospheric type distributions.

## Option 1: Quick Test with Best Model

Test the slim model (current winner) on actual use cases:

```python
from FLiESANN import FLiESANN

# Use optimized model
result = FLiESANN(
    ...,  # your actual inputs
    model_filename='FLiESANN/FLiESANN_slim.h5'
)
```

Compare outputs against original model to verify accuracy in your specific use case.

## Option 2: Retrain with Proper Data

Update `FLiESANN/optimize_model.py` line 160-212 to use actual atype/ctype from data:

```python
# Instead of defaulting to atype1=1, extract from dataset
# This requires having pre-computed atype/ctype columns or
# computing them from KG_climate as done in the original pipeline
```

Then run: `python optimize_fliesann.py` (without --test for full dataset)

## Option 3: Use As-Is for Speed-Critical Applications

If 0.038-0.064 MAE is acceptable for your use case, deploy the slim model:

```bash
# Option A: Replace default model
cp FLiESANN/FLiESANN_slim.h5 FLiESANN/FLiESANN.h5

# Option B: Specify in code
FLiESANN(..., model_filename='FLiESANN/FLiESANN_slim.h5')
```

## Performance Gains

For 1M samples, slim model saves **21.25 seconds** (12.5% improvement).
