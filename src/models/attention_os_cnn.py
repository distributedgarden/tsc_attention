import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.attention import SelfAttention


class AttentionOSCNN(nn.Module):
    """
    Description:
        - An Omni-Scale 1 Dimensional Convolutional Neural Network (OS-CNN) with Self-Attention mechanism.
        - This network uses convolutional layers with specified filter sizes for feature extraction from time-series data,
          combined with a self-attention layer to focus on the most relevant features.

    Attributes:
        - conv1, conv2, conv3 (nn.Conv1d): Convolutional layers for feature extraction.
        - bn1, bn2, bn3 (nn.BatchNorm1d): Batch normalization layers corresponding to each convolutional layer.
        - attention (SelfAttention): Self-attention layer for focusing on relevant features.
        - attention_weights (torch.Tensor): Stores the attention weights computed during the forward pass.
        - fc (nn.Linear): Fully connected layer for classification.

    Args:
        - num_classes (int): The number of classes for classification.
        - cnn_filters (Tuple[int, int, int]): Number of filters for each convolutional layer.
        - hidden_size (int): hidden layer size
        - dropout_rate (float): droput rate
        - input_size (int): instance input size
        - sequence_length (int): instance sequence length
        - batch_size (int): training batch size
    """

    def __init__(
        self,
        num_classes: int,
        cnn_filters: tuple = (128, 256, 128),
        hidden_size: int = 128,
        dropout_rate: float = 0.8,
        input_size: int = 1,
        sequence_length: int = 187,
        batch_size: int = 32,
    ):
        super(AttentionOSCNN, self).__init__()

        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.conv1 = nn.Conv1d(
            input_size,
            cnn_filters[0],
            kernel_size=8,
            padding="same",
            padding_mode="zeros",
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

        self.intermediate = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

        self.attention = SelfAttention(hidden_size)
        self.attention_weights = None

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_size + cnn_filters[2], num_classes)

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

        c = self.avg_pool(conv3_bn)
        pooled_flat = c.view(c.size(0), -1)

        x_reshaped = x.view(-1, 1)
        x_expanded = self.intermediate(x_reshaped)
        x_transformed = x_expanded.view(
            x.size(0), self.sequence_length, self.hidden_size
        )

        attended, attention_weights = self.attention(x_transformed)
        attended = self.dropout(attended[:, -1, :])
        self.attention_weights = attention_weights

        combined = torch.cat((attended, pooled_flat), dim=1)
        output = self.fc(combined)

        return output


class MultiheadAttentionOSCNN(nn.Module):
    """
    Description:
        - An Omni-Scale 1 Dimensional Convolutional Neural Network (OS-CNN) with Multi-Head Attention mechanism.
        - This network uses convolutional layers with specified filter sizes for feature extraction from time-series data,
          combined with a multi-head attention layer to focus on the most relevant features.

    Attributes:
        - conv1, conv2, conv3 (nn.Conv1d): Convolutional layers for feature extraction.
        - bn1, bn2, bn3 (nn.BatchNorm1d): Batch normalization layers corresponding to each convolutional layer.
        - attention (nn.MultiheadAttention): Multi-head attention layer for focusing on relevant features.
        - fc (nn.Linear): Fully connected layer for classification.

    Args:
        - num_classes (int): The number of classes for classification.
        - cnn_filters (Tuple[int, int, int]): Number of filters for each convolutional layer.
        - hidden_size (int): hidden layer size
        - dropout_rate (float): dropout rate
        - input_size (int): instance input size
        - sequence_length (int): instance sequence length
        - batch_size (int): training batch size
        - num_heads (int): number of attention heads
    """

    def __init__(
        self,
        num_classes: int,
        cnn_filters: tuple = (128, 256, 128),
        hidden_size: int = 128,
        dropout_rate: float = 0.8,
        input_size: int = 1,
        sequence_length: int = 187,
        batch_size: int = 32,
        num_heads: int = 4,
    ):
        super(MultiheadAttentionOSCNN, self).__init__()

        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.conv1 = nn.Conv1d(
            input_size,
            cnn_filters[0],
            kernel_size=8,
            padding="same",
            padding_mode="zeros",
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

        self.intermediate = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

        self.attention = nn.MultiheadAttention(hidden_size, num_heads)

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_size + cnn_filters[2], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Description:
            - Forward pass of the AttentionOSCNN model.
            - Permute the dimensions of the input tensor to match the expected shape for 1D convolution
            - Apply convolutional layers with ReLU and batch normalization,
            - Apply multi-head attention
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

        c = self.avg_pool(conv3_bn)
        pooled_flat = c.view(c.size(0), -1)

        x_reshaped = x.view(-1, 1)
        x_expanded = self.intermediate(x_reshaped)
        x_transformed = x_expanded.view(
            x.size(0), self.sequence_length, self.hidden_size
        )

        attended, _ = self.attention(x_transformed, x_transformed, x_transformed)
        attended = self.dropout(attended[:, -1, :])

        combined = torch.cat((attended, pooled_flat), dim=1)
        output = self.fc(combined)

        return output
