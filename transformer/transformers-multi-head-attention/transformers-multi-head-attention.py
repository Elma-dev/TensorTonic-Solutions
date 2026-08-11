import numpy as np
import torch 
import torch.nn.functional as F
import math 

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """
    Q = torch.tensor(Q)
    K = torch.tensor(K)
    V = torch.tensor(V)
    W_q = torch.tensor(W_q)
    W_k = torch.tensor(W_k)
    W_v = torch.tensor(W_v)
    W_o = torch.tensor(W_o)

    head_dim =  W_q.shape[-1] // num_heads
    
    queries = Q @ W_q
    keys = K @ W_k
    values = V @ W_v
    
    b,num_t,out_dim = queries.shape
    
    queries = queries.view(b,num_t,num_heads,head_dim).transpose(1,2) # b,h,t,head_dim
    keys = keys.view(b,num_t,num_heads,head_dim).transpose(1,2)
    values = values.view(b,num_t,num_heads,head_dim).transpose(1,2)
    S = queries @ keys.transpose(2,3) # (b,h,t,head_dim) @ (b,h,head_dim,t) => (b,h,t,t)
    S = F.softmax(S/math.sqrt(queries.shape[-1]),dim=-1) # (b,h,t,t)
    attention = S @ values # (b,h,t,t) (b,h,t,head_dim) => (b,h,t,head_dim)
    attention = attention.transpose(1,2)
    attention = attention.contiguous().view(b, num_t, out_dim)
    output = attention @ W_o
    return np.array(output.tolist())