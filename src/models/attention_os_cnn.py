import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.attention import SelfAttention


class AttentionOSCNN(nn.Module):
    """
    An Omni-Scale 1 Dimensional Convolutional Neural Network (OS-CNN) with Self-Attention mechanism.
    This network uses convolutional layers with multiple filter sizes for feature extraction from time-series data,
    combined with a self-attention layer to focus on the most relevant features.

    Attributes:
        conv1_filters (nn.ModuleList): First set of convolutional layers with varying kernel sizes.
        conv2_filters (nn.ModuleList): Second set of convolutional layers with varying kernel sizes.
        conv3_filters (nn.ModuleList): Third set of convolutional layers with varying kernel sizes.
        attention (SelfAttention): Self-attention layer for focusing on relevant features.
        fc (nn.Linear): Fully connected layer for classification.
        attention_weights (torch.Tensor): Stores the attention weights computed during the forward pass.

    Args:
        num_classes (int): The number of classes for classification.
    """

    def __init__(self, num_classes: int):
        super(AttentionOSCNN, self).__init__()

        self.conv1_filters = nn.ModuleList(
            [nn.Conv1d(1, 32, kernel_size=ks) for ks in [1, 2, 3, 5, 7, 11]]
        )
        self.conv2_filters = nn.ModuleList(
            [nn.Conv1d(192, 64, kernel_size=ks) for ks in [1, 2, 3, 5, 7, 11]]
        )
        self.conv3_filters = nn.ModuleList(
            [nn.Conv1d(384, 128, kernel_size=ks) for ks in [1, 2]]
        )

        self.attention = SelfAttention(256)
        self.fc = nn.Linear(256, num_classes)
        self.attention_weights = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the AttentionOSCNN model.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, channels, input_length).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, num_classes).
        """
        conv1_list = [F.relu(conv(x)) for conv in self.conv1_filters]
        conv1_cat = torch.cat(conv1_list, 1)
        conv1_bn = F.relu(nn.BatchNorm1d(192)(conv1_cat))

        conv2_list = [F.relu(conv(conv1_bn)) for conv in self.conv2_filters]
        conv2_cat = torch.cat(conv2_list, 1)
        conv2_bn = F.relu(nn.BatchNorm1d(384)(conv2_cat))

        conv3_list = [F.relu(conv(conv2_bn)) for conv in self.conv3_filters]
        conv3_cat = torch.cat(conv3_list, 1)
        conv3_bn = F.relu(nn.BatchNorm1d(256)(conv3_cat))

        attended, attention_weights = self.attention(conv3_bn.transpose(1, 2))
        attended_sum = torch.sum(attended, dim=1)

        self.attention_weights = attention_weights

        output = self.fc(attended_sum)

        return output
