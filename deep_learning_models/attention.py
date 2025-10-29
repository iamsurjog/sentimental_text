
import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, features):
        # features shape: (batch_size, seq_len, hidden_dim)
        attn_weights = F.softmax(self.attn(features), dim=1)
        # attn_weights shape: (batch_size, seq_len, 1)
        context = torch.sum(attn_weights * features, dim=1)
        # context shape: (batch_size, hidden_dim)
        return context, attn_weights

def apply_attention(features):
    """
    Applies a simple attention mechanism.

    Args:
        features: A tensor of features.

    Returns:
        A tensor with attention applied.
    """
    print("Applying attention mechanism...")
    if not isinstance(features, torch.Tensor):
        features = torch.randn(1, 10, 300) # Dummy features

    hidden_dim = features.shape[2]
    attention = Attention(hidden_dim)
    context, _ = attention(features)
    return context
