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
