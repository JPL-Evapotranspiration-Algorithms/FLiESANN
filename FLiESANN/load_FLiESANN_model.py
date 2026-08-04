from os.path import join, abspath, dirname
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import tensorflow as tf

    # tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    # tf.disable_v2_behavior()
    # tf.logging.set_verbosity(tf.logging.ERROR)
    # from keras.engine.saving import load_model
    from keras.models import load_model
    from keras.saving import register_keras_serializable

DEFAULT_MODEL_FILENAME = join(abspath(dirname(__file__)), "FLiESANN.h5")

@register_keras_serializable()
def mae(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))

def load_FLiESANN_model(model_filename: str = DEFAULT_MODEL_FILENAME, enable_xla: bool = False):
    """
    Load the FLiESANN model from disk.
    
    Args:
        model_filename: Path to model file (default: FLiESANN.h5)
        enable_xla: Enable XLA JIT compilation for 20-30% speedup (default: False)
        
    Returns:
        Loaded Keras model
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = load_model(model_filename, custom_objects={'mae': mae}, compile=False)
        
        # Optionally enable XLA compilation for additional speedup
        if enable_xla:
            model.compile(jit_compile=True)
        
        return model
