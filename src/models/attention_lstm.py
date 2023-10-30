import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.attention import SelfAttention


class AttentionLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(AttentionLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers, batch_first=True
        )
        self.attention = SelfAttention(hidden_dim)

        # Final dense layer for classification
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]

        # LSTM forward pass
        lstm_out, _ = self.lstm(x)

        # Self Attention on LSTM outputs
        attended, attention_weights = self.attention(lstm_out)

        # Use the last time-step of the attended sequence for classification
        out = self.fc(attended[:, -1, :])

        return F.softmax(out, dim=1)


# Example usage:
input_dim = 1  # For univariate time series
hidden_dim = 64
num_layers = 2
output_dim = 5  # Assuming 5 classes for classification
model = AttentionLSTM(input_dim, hidden_dim, num_layers, output_dim)
print(model)
