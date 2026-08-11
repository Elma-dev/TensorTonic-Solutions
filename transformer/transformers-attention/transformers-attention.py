import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    S =  Q @ K.transpose(dim0=1,dim1=-1) # (tokens,d) @ (d,tokens) ==> (tokens,tokens)
    scores = F.softmax(S/math.sqrt(Q.shape[-1]),dim=-1) # ==> (tokens,tokens)
    return scores @ V # (tokens,tokens) @ (d,tokens)
    