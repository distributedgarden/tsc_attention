import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.attention import SelfAttention

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLSTMFCN(nn.Module):
    """
    A Hybrid Long Short-Term Memory Fully Convolutional Network (LSTM-FCN) with an Attention mechanism.
    This model enhances the LSTM-FCN by incorporating an attention layer that helps focus on specific parts of the input sequence.

    Attributes:
        lstm (nn.LSTM): LSTM layer for sequential data processing.
        attention (SelfAttention): Self-attention layer for focusing on important features.
        dropout (nn.Dropout): Dropout layer for regularization.
        conv1, conv2, conv3 (nn.Conv1d): Convolutional layers for feature extraction.
        bn1, bn2, bn3 (nn.BatchNorm1d): Batch normalization layers corresponding to each convolutional layer.
        global_avg_pooling (nn.AdaptiveAvgPool1d): Global average pooling layer for feature aggregation.
        fc (nn.Linear): Fully connected layer for final classification.
        attention_weights (torch.Tensor): Stores the attention weights computed during the forward pass.

    Args:
        input_size (int): The number of input features for the LSTM layer.
        num_classes (int): The number of output classes for classification.
        hidden_size (int): The number of features in the hidden state of the LSTM.
        dropout_rate (float): The dropout rate for regularization.
        cnn_filters (Tuple[int, int, int]): Number of filters for each convolutional layer.
    """

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        hidden_size: int = 8,
        num_layers: int = 1,
        dropout_rate: float = 0.8,
        cnn_filters: tuple = (128, 256, 128),
    ):
        super(AttentionLSTMFCN, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.attention = SelfAttention(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

        self.conv1 = nn.Conv1d(
            1, cnn_filters[0], kernel_size=8, padding="same", padding_mode="zeros"
        )
        self.bn1 = nn.BatchNorm1d(cnn_filters[0])
        self.conv2 = nn.Conv1d(
            cnn_filters[0],
            cnn_filters[1],
            kernel_size=5,
            padding="same",
            padding_mode="zeros",
        )
        self.bn2 = nn.BatchNorm1d(cnn_filters[1])
        self.conv3 = nn.Conv1d(
            cnn_filters[1],
            cnn_filters[2],
            kernel_size=3,
            padding="same",
            padding_mode="zeros",
        )
        self.bn3 = nn.BatchNorm1d(cnn_filters[2])

        self.global_avg_pooling = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_size + cnn_filters[2], num_classes)

        self.attention_weights = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the AttentionLSTMFCN model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """
        h, _ = self.lstm(x)
        h, attention_weights = self.attention(h)
        h = self.dropout(h[:, -1, :])

        self.attention_weights = attention_weights

        x_permuted = x.permute(0, 2, 1)
        c = F.relu(self.bn1(self.conv1(x_permuted)))
        c = F.relu(self.bn2(self.conv2(c)))
        c = F.relu(self.bn3(self.conv3(c)))

        c = self.global_avg_pooling(c)
        c = c.view(c.size(0), -1)

        combined = torch.cat((h, c), dim=1)
        out = self.fc(combined)

        return out
