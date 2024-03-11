import torch
import torch.nn as nn

from flash_attn import flash_attn_func
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


class FlashAttentionLSTM(nn.Module):
    """
    Description:
        - An LSTM model with a Flash Attention mechanism.
        - LSTM layer
        - Flash Attention layer
        - fully connected layer.

    Attributes:
        - lstm (nn.LSTM): The LSTM layer.
        - attention (nn.Linear): Linear layer to generate QKV tensors for Flash Attention.
        - fc (nn.Linear): A fully connected layer for output.

    Args:
        - input_size (int): the number of input features.
        - hidden_size (int): the number of features in the hidden state of the LSTM.
        - num_layers (int): the number of recurrent layers in the LSTM.
        - num_heads (int): the number of attention heads.
        - head_dim (int): the dimension of each attention head.
        - output_dim (int): the number of output features.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        output_dim: int,
    ) -> None:
        super(FlashAttentionLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers=num_layers, batch_first=True
        )
        self.attention = nn.Linear(hidden_size, num_heads * head_dim * 3)
        self.fc = nn.Linear(hidden_size, output_dim)
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.attn_output = None

        self.lstm = self.lstm.to(torch.float16)
        self.attention = self.attention.to(torch.float16)
        self.fc = self.fc.to(torch.float16)

    def forward(self, x: Tensor) -> Tensor:
        """
        Description:
        - The forward pass of the model.
        - Steps:
            - the input data is passed through the LSTM layer
            - the output of the LSTM layer is used to generate QKV tensors for Flash Attention
            - Flash Attention is applied to the QKV tensors
            - the attention output is saved for visualization
            - a context vector is obtained by summing over the attended outputs
            - the context vector is passed through a fully connected layer
            - the fully connected layer output is returned

        Args:
        - x (Tensor): input tensor; [batch_size, seq_len, input_dim].

        Returns:
        - Tensor: output tensor of raw logits; [batch_size, output_dim].
        """
        x = x.to(torch.float16)
        lstm_out, _ = self.lstm(x)

        # generate qkv tensors for flash attention
        qkv = self.attention(lstm_out)
        qkv = qkv.view(qkv.size(0), qkv.size(1), self.num_heads, 3, self.head_dim)
        qkv = qkv.permute(2, 0, 1, 3, 4)
        q, k, v = qkv.unbind(dim=-2)

        # apply flash attention
        attn_out = flash_attn_func(q, k, v)
        self.attn_output = attn_out.permute(1, 2, 0, 3).contiguous()

        # sum over the attended outputs
        context_vector = attn_out.sum(dim=-2).view(attn_out.size(1), -1)
        out = self.fc(context_vector)

        return out
