import torch
import torch.nn as nn

from torch import Tensor

from src.models.attention import SelfAttention


class AttentionLSTM(nn.Module):
    """
    Description:
        - An LSTM model with a self-attention mechanism.
            - LSTM layer
            - self-attention layer
            - fully connected layer.

    Attributes:
        - lstm (nn.LSTM): The LSTM layer.
        - attention (SelfAttention): The self-attention layer.
        - fc (nn.Linear): A fully connected layer for output.

    Args:
        - input_dim (int): the number of input features.
        - hidden_dim (int): the number of features in the hidden state of the LSTM.
        - num_layers (int): the number of recurrent layers in the LSTM.
        - output_dim (int): the number of output features.
    """

    def __init__(
        self, input_dim: int, hidden_dim: int, num_layers: int, output_dim: int
    ) -> None:
        super(AttentionLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers, batch_first=True
        )
        self.attention = SelfAttention(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        Description:
            - The forward pass of the model.
            - Steps:
                - the input data is passed through the LSTM layer
                - the output of the LSTM layer is passed through the self-attention layer
                - a context vector is obtained by summing over the attended outputs
                - the context vector is passed through a fully connected layer
                - the fully connected layer output is returned

        Args:
            - x (Tensor): input tensor; [batch_size, seq_len, input_dim].

        Returns:
            - Tensor: output tensor of raw logits; [batch_size, output_dim].
        """
        lstm_out, _ = self.lstm(x)
        attended, _ = self.attention(lstm_out)
        context_vector = torch.sum(attended, dim=1)
        out = self.fc(context_vector)

        return out


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
