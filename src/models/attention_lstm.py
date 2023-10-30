import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.attention import SelfAttention


class AttentionLSTM(nn.Module):
    """
    Description:
        - Long Short-Term Memory with self-attention
    """

    def __init__(
        self, input_dim: int, hidden_dim: int, num_layers: int, output_dim: int
    ):
        super(AttentionLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers, batch_first=True
        )
        self.attention = SelfAttention(hidden_dim)

        # fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Description:
            - forward pass

        x shape: [batch_size, seq_len, input_dim]
        """

        # LSTM forward pass
        lstm_out, _ = self.lstm(x)

        # self-attention
        attended, attention_weights = self.attention(lstm_out)

        # use the last time-step of the attended sequence for classification
        out = self.fc(attended[:, -1, :])

        return F.softmax(out, dim=1)


def example():
    """
    Description:
        - example usage
    """
    input_dim = 1
    hidden_dim = 64
    num_layers = 2
    output_dim = 5
    model = AttentionLSTM(input_dim, hidden_dim, num_layers, output_dim)
    print(model)
