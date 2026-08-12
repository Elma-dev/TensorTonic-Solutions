import numpy as np
import torch

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Apply position-wise feed-forward network.
    """
    x = torch.tensor(x,dtype=torch.float32)
    W1 = torch.tensor(W1,dtype=torch.float32)
    W2 = torch.tensor(W2,dtype=torch.float32)
    b1 = torch.tensor(b1,dtype=torch.float32)
    b2 = torch.tensor(b2,dtype=torch.float32)
    
    output = (x @ W1 + b1)
    relu_out = torch.where(output<0,0,output)
    out = relu_out @ W2 + b2
    return np.array(out.tolist())