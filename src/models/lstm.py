import torch.nn as nn


class LSTM(nn.Module):
    """
    Description:
        - Long Short-Term Memory (LSTM)
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

    def forward(self, x):
        """
        Description:
            - forward pass
        """
        # LSTM layers
        out, _ = self.lstm(x)

        # output from the last time step
        out = out[:, -1, :]

        # fully connected layer
        out = self.fc(out)

        return out


def example():
    """
    Description:
        - example usage
    """
    input_dim = 1
    hidden_dim = 128
    num_layers = 2
    num_classes = 5

    model = LSTM(input_dim, hidden_dim, num_layers, num_classes)
    print(model)
