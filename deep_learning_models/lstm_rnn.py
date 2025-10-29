import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, dropout):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)

    def forward(self, x):
        output, _ = self.lstm(x)
        return output

class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, dropout):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)

    def forward(self, x):
        output, _ = self.rnn(x)
        return output

def run_lstm(features, input_dim=300, hidden_dim=128, n_layers=2, dropout=0.5):
    """
    Runs an LSTM model on the given features.

    Args:
        features: A tensor of features.
        input_dim: The input dimension.
        hidden_dim: The hidden dimension.
        n_layers: The number of layers.
        dropout: The dropout rate.

    Returns:
        The output of the LSTM model.
    """
    print("Running LSTM model...")
    if not isinstance(features, torch.Tensor):
        features = torch.randn(1, 10, input_dim) # Dummy features

    model = LSTMModel(input_dim, hidden_dim, n_layers, dropout)
    model.eval()
    with torch.no_grad():
        output = model(features)
    return output

def run_rnn(features, input_dim=300, hidden_dim=128, n_layers=2, dropout=0.5):
    """
    Runs an RNN model on the given features.

    Args:
        features: A tensor of features.
        input_dim: The input dimension.
        hidden_dim: The hidden dimension.
        n_layers: The number of layers.
        dropout: The dropout rate.

    Returns:
        The output of the RNN model.
    """
    print("Running RNN model...")
    if not isinstance(features, torch.Tensor):
        features = torch.randn(1, 10, input_dim) # Dummy features

    model = RNNModel(input_dim, hidden_dim, n_layers, dropout)
    model.eval()
    with torch.no_grad():
        output = model(features)
    return output