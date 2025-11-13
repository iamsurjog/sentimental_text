import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, dropout, num_classes):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        output, _ = self.lstm(x)
        # Get the last time step's output
        output = output[:, -1, :]
        output = self.fc(output)
        return output

class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, dropout, num_classes):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        output, _ = self.rnn(x)
        # Get the last time step's output
        output = output[:, -1, :]
        output = self.fc(output)
        return output

def run_lstm(features, input_dim=300, hidden_dim=128, n_layers=2, dropout=0.5, num_classes=2):
    """
    Runs an LSTM model on the given features.

    Args:
        features: A tensor of features.
        input_dim: The input dimension.
        hidden_dim: The hidden dimension.
        n_layers: The number of layers.
        dropout: The dropout rate.
        num_classes: The number of output classes.

    Returns:
        The output of the LSTM model.
    """
    print("Running LSTM model...")
    if not isinstance(features, torch.Tensor):
        features = torch.randn(1, 10, input_dim) # Dummy features

    model = LSTMModel(input_dim, hidden_dim, n_layers, dropout, num_classes)
    model.eval()
    with torch.no_grad():
        output = model(features)
    return output

def run_rnn(features, input_dim=300, hidden_dim=128, n_layers=2, dropout=0.5, num_classes=2):
    """
    Runs an RNN model on the given features.

    Args:
        features: A tensor of features.
        input_dim: The input dimension.
        hidden_dim: The hidden dimension.
        n_layers: The number of layers.
        dropout: The dropout rate.
        num_classes: The number of output classes.

    Returns:
        The output of the RNN model.
    """
    print("Running RNN model...")
    if not isinstance(features, torch.Tensor):
        features = torch.randn(1, 10, input_dim) # Dummy features

    model = RNNModel(input_dim, hidden_dim, n_layers, dropout, num_classes)
    model.eval()
    with torch.no_grad():
        output = model(features)
    return output