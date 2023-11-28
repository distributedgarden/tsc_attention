import torch
import torch.nn as nn


class LSTM(nn.Module):
    """
    A basic Long Short-Term Memory (LSTM) network for sequence modeling and classification tasks.
    This model consists of LSTM layers followed by a fully connected layer for output.

    Attributes:
        lstm (nn.LSTM): LSTM layers for processing sequential data.
        fc (nn.Linear): A fully connected (dense) layer for output classification or regression.

    Args:
        input_size (int): The number of input features in each element of the sequence.
        hidden_units (int): The number of features in the hidden state of each LSTM layer.
        num_lstm_layers (int): The number of recurrent layers in the LSTM module.
        num_classes (int): The number of output classes (for classification tasks) or output features (for regression tasks).
    """

    def __init__(
        self, input_size: int, hidden_units: int, num_lstm_layers: int, num_classes: int
    ):
        super(LSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_units,
            num_layers=num_lstm_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(in_features=hidden_units, out_features=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the LSTM model.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, sequence_length, input_size).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, num_classes).
        """
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Extract the output from the last time step
        out = self.fc(out)  # Apply the fully connected layer

        return out
