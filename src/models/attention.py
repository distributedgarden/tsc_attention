import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor


class SelfAttention(nn.Module):
    """
    Description:
        - self-attention mechanism.

    Attributes:
        - query (nn.Linear): linear transformation for the query vector.
        - key (nn.Linear): linear transformation for the key vector.
        - value (nn.Linear): linear transformation for the value vector.

    Args:
        - hidden_dim (int): the dimensionality of the input and output of the self-attention mechanism.
    """

    def __init__(self, hidden_dim: int) -> None:
        super(SelfAttention, self).__init__()

        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Description:
            - the forward pass of the self-attention mechanism.
            - the input tensor is transformed into query, key, and value representations.
            - attention scores are computed by performing a batch matrix multiplication between the query and the key.
            - scores are normalized using softmax
            - these weights are used to create a weighted sum of the value vectors

        Args:
            - x (Tensor): input tensor; [batch_size, seq_len, hidden_dim].

        Returns:
            tuple[Tensor, Tensor]:
                - attended (Tensor): the output tensor after applying attention; shape [batch_size, seq_len, hidden_dim].
                - attention_weights (Tensor): the attention weights; shape [batch_size, seq_len, seq_len].
        """
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        attention_scores = torch.bmm(query, key.transpose(1, 2))
        attention_weights = F.softmax(attention_scores, dim=-1)
        attended = torch.bmm(attention_weights, value)

        return attended, attention_weights


class BahdanauAttention(nn.Module):
    """
    Description:
        - Bahdanau attention
    """

    def __init__(self, hidden_dim):
        super(BahdanauAttention, self).__init__()
        self.key_layer = nn.Linear(hidden_dim, hidden_dim)
        self.query_layer = nn.Linear(hidden_dim, hidden_dim)
        self.energy_layer = nn.Linear(hidden_dim, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, query, values):
        """
        Args:
            - query shape: [batch_size, hidden_dim]
            - values shape: [batch_size, seq_len, hidden_dim]
        """
        # [batch_size, 1, hidden_dim]
        query = query.unsqueeze(1)

        # [batch_size, seq_len, hidden_dim]
        keys = self.key_layer(values)

        # calculate energy scores
        # [batch_size, seq_len, 1]
        energy = self.energy_layer(torch.tanh(query + keys))

        # [batch_size, seq_len]
        attention_weights = self.softmax(energy.squeeze(2))

        # sum the weighted values
        out = torch.bmm(attention_weights.unsqueeze(1), values)
        out = out.squeeze(1)

        return out, attention_weights


class LuongAttention(nn.Module):
    """
    Description:
        - Luong's attention
    """

    def __init__(self, hidden_dim):
        super(LuongAttention, self).__init__()
        self.alignment_score = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, hidden, encoder_outputs):
        """
        Description:
            - forward pass for Luong's Attention

        Args:
            - hidden : the current decoder hidden state, shape [batch_size, hidden_dim]
            - encoder_outputs : the encoder outputs, shape [batch_size, seq_len, hidden_dim]
        """
        # project the encoder outputs to the hidden space
        proj_key = self.alignment_score(
            encoder_outputs
        )  # [batch_size, seq_len, hidden_dim]

        # compute the alignment scores
        # hidden shape is [batch_size, hidden_dim], so we unsqueeze to [batch_size, 1, hidden_dim]
        # after bmm, the shape is [batch_size, 1, seq_len]
        alignment_scores = torch.bmm(hidden.unsqueeze(1), proj_key.transpose(1, 2))

        # compute the attention weights
        # [batch_size, 1, seq_len]
        attn_weights = F.softmax(alignment_scores, dim=-1)

        # compute the context vector as the weighted average of encoder outputs
        # [batch_size, 1, hidden_dim]
        context_vector = torch.bmm(attn_weights, encoder_outputs)

        # [batch_size, hidden_dim]
        context_vector = context_vector.squeeze(1)

        # Return the context vector and the attention weights
        # squeezed to [batch_size, seq_len]
        return context_vector, attn_weights.squeeze(1)
