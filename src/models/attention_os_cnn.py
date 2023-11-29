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
        attention_weights (torch.Tensor): Stores the attention weights computed during the forward pass.
        fc (nn.Linear): Fully connected layer for classification.

    Args:
        num_classes (int): The number of classes for classification.
        cnn_filters (Tuple[int, int, int]): Number of filters for each convolutional layer.
        hidden_size (int): hidden layer size
    """

    def __init__(
        self,
        num_classes: int,
        cnn_filters: tuple = (128, 256, 128),
        hidden_size: int = 128,
    ):
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

        self.intermediate = nn.Linear(cnn_filters[2], hidden_size)

        self.attention = SelfAttention(hidden_size)
        self.attention_weights = None

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Description:
            - Forward pass of the AttentionOSCNN model.
            - Permute the dimensions of the input tensor to match the expected shape for 1D convolution
            - Apply convolutional layers with ReLU and batch normalization,
            - Apply self-attention
            - Attended features are weighted and summed based on attention weights
            - Results are fed into a fully connected layer

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, sequence_length, input_size).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, num_classes).
        """
        x_permuted = x.permute(0, 2, 1)

        conv1_bn = F.relu(self.bn1(self.conv1(x_permuted)))
        conv2_bn = F.relu(self.bn2(self.conv2(conv1_bn)))
        conv3_bn = F.relu(self.bn3(self.conv3(conv2_bn)))

        # transform dimensions to match self-attention
        intermediate_output = self.intermediate(conv3_bn.transpose(1, 2))
        intermediate_output = intermediate_output.transpose(1, 2)

        attended, attention_weights = self.attention(intermediate_output)
        self.attention_weights = attention_weights

        pooled_flat = self.avg_pool(attended).view(x.size(0), -1)
        output = self.fc(pooled_flat)

        return output
