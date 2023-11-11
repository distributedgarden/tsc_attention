import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """
    Description:
        - self-attention
    """

    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.scale = torch.sqrt(torch.FloatTensor([input_dim]))

    def forward(self, x):
        """
        Description:
            - forward pass

        x shape: [batch_size, seq_len, input_dim]
        """
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        attention_weights = F.softmax(q @ k.transpose(-2, -1) / self.scale, dim=-1)
        out = attention_weights @ v

        return out, attention_weights


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
        Arguments:
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

        Arguments:
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
