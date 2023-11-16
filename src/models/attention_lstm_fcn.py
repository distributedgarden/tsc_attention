import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.attention import SelfAttention


class AttentionLSTMFCN(nn.Module):
    """
    Description:
        - Hybrid Long Short-Term Memory Fully Convolutional Network (LSTM-FCN) with Attention
    """

    def __init__(self, max_sequence_length: int, nb_class: int, num_cells: int = 8):
        super(AttentionLSTMFCN, self).__init__()

        # LSTM layer with self-attention and dropout
        self.lstm = nn.LSTM(max_sequence_length, num_cells, batch_first=True)
        self.attention = SelfAttention(num_cells)
        self.dropout = nn.Dropout(0.8)

        # CNN layers with batch-normalization
        self.conv1 = nn.Conv1d(
            1, 128, kernel_size=8, padding="same", padding_mode="zeros"
        )
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(
            128, 256, kernel_size=5, padding="same", padding_mode="zeros"
        )
        self.bn2 = nn.BatchNorm1d(256)
        self.conv3 = nn.Conv1d(
            256, 128, kernel_size=3, padding="same", padding_mode="zeros"
        )
        self.bn3 = nn.BatchNorm1d(128)

        # global average pooling
        self.global_avg_pooling = nn.AdaptiveAvgPool1d(1)

        # fully connected layer
        self.fc = nn.Linear(num_cells + 128, nb_class)

        self.attention_weights = None

    def forward(self, x):
        # LSTM with dropout forward pass
        h, _ = self.lstm(x)
        h, attention_weights = self.attention(h)
        h = self.dropout(h[:, -1, :])

        self.attention_weights = attention_weights

        # CNN with batch-normalization forward pass
        x_permuted = x.permute(0, 2, 1)
        c = F.relu(self.bn1(self.conv1(x_permuted)))
        c = F.relu(self.bn2(self.conv2(c)))
        c = F.relu(self.bn3(self.conv3(c)))

        # global average pooling
        c = self.global_avg_pooling(c)
        c = c.view(c.size(0), -1)

        # concatenate LSTM and CNN outputs
        combined = torch.cat((h, c), dim=1)

        # fully connected layer with softmax
        out = F.softmax(self.fc(combined), dim=1)

        return out


def example():
    """
    Description:
        - example usage
    """
    max_sequence_length = 100
    nb_class = 5
    model = AttentionLSTMFCN(max_sequence_length, nb_class)
    print(model)
