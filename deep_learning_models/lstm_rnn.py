
import torch
import torch.nn as nn

def run_lstm(features):
    """
    Runs an LSTM model on the given features.

    Args:
        features: A tensor of features.

    Returns:
        The output of the LSTM model.
    """
    print("Running LSTM model...")
    if not isinstance(features, torch.Tensor):
        features = torch.randn(1, 1, 300) # Dummy features

    lstm = nn.LSTM(features.shape[2], 128)
    output, _ = lstm(features)
    return output

def run_rnn(features):
    """
    Runs an RNN model on the given features.

    Args:
        features: A tensor of features.

    Returns:
        The output of the RNN model.
    """
    print("Running RNN model...")
    if not isinstance(features, torch.Tensor):
        features = torch.randn(1, 1, 300) # Dummy features

    rnn = nn.RNN(features.shape[2], 128)
    output, _ = rnn(features)
    return output
