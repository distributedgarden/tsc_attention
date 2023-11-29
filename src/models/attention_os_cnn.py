import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.attention import SelfAttention


class AttentionOSCNN(nn.Module):
    """
    An Omni-Scale 1 Dimensional Convolutional Neural Network (OS-CNN) with Self-Attention mechanism.
    This network uses convolutional layers with specified filter sizes for feature extraction from time-series data,
    combined with a self-attention layer to focus on the most relevant features.

    Attributes:
        conv1, conv2, conv3 (nn.Conv1d): Convolutional layers for feature extraction.
        bn1, bn2, bn3 (nn.BatchNorm1d): Batch normalization layers corresponding to each convolutional layer.
        attention (SelfAttention): Self-attention layer for focusing on relevant features.
        fc (nn.Linear): Fully connected layer for classification.
        attention_weights (torch.Tensor): Stores the attention weights computed during the forward pass.

    Args:
        num_classes (int): The number of classes for classification.
        cnn_filters (Tuple[int, int, int]): Number of filters for each convolutional layer.
    """

    def __init__(self, num_classes: int, cnn_filters: tuple = (128, 256, 128)):
        super(AttentionOSCNN, self).__init__()

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

        self.attention = SelfAttention(cnn_filters[2])
        self.fc = nn.Linear(cnn_filters[2], num_classes)
        self.attention_weights = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the AttentionOSCNN model.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, channels, input_length).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, num_classes).
        """
        c = F.relu(self.bn1(self.conv1(x)))
        c = F.relu(self.bn2(self.conv2(c)))
        c = F.relu(self.bn3(self.conv3(c)))

        attended, attention_weights = self.attention(c.transpose(1, 2))
        attended_sum = torch.sum(attended, dim=1)

        self.attention_weights = attention_weights

        output = self.fc(attended_sum)

        return output
