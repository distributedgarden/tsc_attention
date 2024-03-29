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
        - input_size (int): the number of input features.
        - hidden_size (int): the number of features in the hidden state of the LSTM.
        - num_layers (int): the number of recurrent layers in the LSTM.
        - output_dim (int): the number of output features.
    """

    def __init__(
        self, input_size: int, hidden_size: int, num_layers: int, output_dim: int
    ) -> None:
        super(AttentionLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers=num_layers, batch_first=True
        )
        self.attention = SelfAttention(hidden_size)
        self.fc = nn.Linear(hidden_size, output_dim)
        self.attention_weights = None

    def forward(self, x: Tensor) -> Tensor:
        """
        Description:
            - The forward pass of the model.
            - Steps:
                - the input data is passed through the LSTM layer
                - the output of the LSTM layer is passed through the self-attention layer
                - record attention weights
                - a context vector is obtained by summing over the attended outputs
                - the context vector is passed through a fully connected layer
                - the fully connected layer output is returned

        Args:
            - x (Tensor): input tensor; [batch_size, seq_len, input_dim].

        Returns:
            - Tensor: output tensor of raw logits; [batch_size, output_dim].
        """
        lstm_out, _ = self.lstm(x)
        attended, attention_weights = self.attention(lstm_out)

        self.attention_weights = attention_weights

        context_vector = torch.sum(attended, dim=1)
        out = self.fc(context_vector)

        return out


class MultiheadAttentionLSTM(nn.Module):
    """
    Description:
        - An LSTM model with a Multihead Attention mechanism.
        - LSTM layer
        - Multihead Attention layer
        - fully connected layer

    Attributes:
        - embedding (nn.Embedding): Embedding layer for input data.
        - lstm (nn.LSTM): The LSTM layer.
        - attention (nn.MultiheadAttention): The Multihead Attention layer.
        - fc (nn.Linear): A fully connected layer for output.

    Args:
        - input_size (int): the number of input features.
        - hidden_size (int): the number of features in the hidden state of the LSTM.
        - num_layers (int): the number of recurrent layers in the LSTM.
        - num_classes (int): the number of output features.
        - num_heads (int): the number of attention heads.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        num_heads: int,
    ):
        super(MultiheadAttentionLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads
        )
        self.fc = nn.Linear(in_features=hidden_size, out_features=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Description:
            - The forward pass of the model.
            - Steps:
                - the input data is passed through the LSTM layer
                - the output of the LSTM layer is passed through the Multihead Attention layer
                - a context vector is obtained by summing over the attended outputs
                - the context vector is passed through a fully connected layer
                - the fully connected layer output is returned

        Args:
            - x (torch.Tensor): input tensor; [batch_size, seq_len].

        Returns:
            - torch.Tensor: output tensor of raw logits; [batch_size, num_classes].
        """
        lstm_out, _ = self.lstm(x)

        attention_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        out = attention_out[:, -1, :]
        out = self.fc(out)

        return out
