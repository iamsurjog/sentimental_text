import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerModel(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward):
        super(TransformerModel, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(
            d_model=d_model, 
            nhead=nhead, 
            num_encoder_layers=num_encoder_layers, 
            num_decoder_layers=num_decoder_layers, 
            dim_feedforward=dim_feedforward
        )

    def forward(self, src, tgt):
        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)
        output = self.transformer(src, tgt)
        return output

def run_transformer(features, d_model=300, nhead=6, num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=1024):
    """
    Runs a Transformer model on the given features.

    Args:
        features: A tensor of features.
        d_model: The model dimension.
        nhead: The number of attention heads.
        num_encoder_layers: The number of encoder layers.
        num_decoder_layers: The number of decoder layers.
        dim_feedforward: The dimension of the feedforward network.

    Returns:
        The output of the Transformer model.
    """
    print("Running Transformer model...")
    if not isinstance(features, torch.Tensor):
        features = torch.randn(10, 1, d_model) # (seq_len, batch_size, d_model)

    # The transformer model expects src and tgt inputs.
    # For many tasks, src and tgt can be the same.
    src = features
    tgt = features

    model = TransformerModel(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward)
    model.eval()
    with torch.no_grad():
        output = model(src, tgt)
    return output