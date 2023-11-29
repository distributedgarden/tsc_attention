import torch
import torch.nn as nn
import torch.nn.functional as F


class OSCNN(nn.Module):
    """
    Omni-Scale 1 Dimensional Convolutional Neural Network (OS-CNN), based on the architecture proposed by Tang et al.
    This model uses convolutional layers with multiple kernel sizes for feature extraction from time-series data,
    followed by global average pooling and a fully connected layer for classification.

    Attributes:
        conv1_filters (nn.ModuleList): First set of convolutional layers with varying kernel sizes.
        conv2_filters (nn.ModuleList): Second set of convolutional layers with varying kernel sizes.
        conv3_filters (nn.ModuleList): Third set of convolutional layers with varying kernel sizes.
        avg_pool (nn.AdaptiveAvgPool1d): Global average pooling layer.
        fc (nn.Linear): Fully connected layer for classification.

    Args:
        num_classes (int): The number of classes for classification.
    """

    def __init__(self, num_classes: int):
        super(OSCNN, self).__init__()

        self.conv1_filters = nn.ModuleList(
            [nn.Conv1d(1, 32, kernel_size=ks) for ks in [1, 2, 3, 5, 7, 11]]
        )
        self.conv2_filters = nn.ModuleList(
            [nn.Conv1d(192, 64, kernel_size=ks) for ks in [1, 2, 3, 5, 7, 11]]
        )
        self.conv3_filters = nn.ModuleList(
            [nn.Conv1d(384, 128, kernel_size=ks) for ks in [1, 2]]
        )

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the OSCNN model.

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

        pooled = self.avg_pool(conv3_bn)
        pooled_flat = pooled.view(pooled.size(0), -1)

        output = self.fc(pooled_flat)

        return output
