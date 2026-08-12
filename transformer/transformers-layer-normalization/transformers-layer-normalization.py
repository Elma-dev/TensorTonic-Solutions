import numpy as np

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Returns: Normalized array of same shape as x
    """
    mean = np.mean(x,axis=-1,keepdims=True)
    var = np.var(x,axis=-1,keepdims=True)
    normalization_result = (x-mean)/(var+eps)**0.5
    return gamma * normalization_result + beta