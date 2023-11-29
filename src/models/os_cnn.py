import torch
import torch.nn as nn
import torch.nn.functional as F


class OSCNN(nn.Module):
    """
    Omni-Scale 1 Dimensional Convolutional Neural Network (OS-CNN).
    This model uses convolutional layers with specified kernel sizes and filters for feature extraction from time-series data,
    followed by global average pooling and a fully connected layer for classification.

    Attributes:
        conv1, conv2, conv3 (nn.Conv1d): Convolutional layers for feature extraction.
        bn1, bn2, bn3 (nn.BatchNorm1d): Batch normalization layers corresponding to each convolutional layer.
        avg_pool (nn.AdaptiveAvgPool1d): Global average pooling layer.
        fc (nn.Linear): Fully connected layer for classification.

    Args:
        num_classes (int): The number of classes for classification.
        cnn_filters (Tuple[int, int, int]): Number of filters for each convolutional layer.
    """

    def __init__(self, num_classes: int, cnn_filters: tuple = (128, 256, 128)):
        super(OSCNN, self).__init__()

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

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(cnn_filters[2], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the OSCNN model.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, channels, input_length).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, num_classes).
        """
        c = F.relu(self.bn1(self.conv1(x)))
        c = F.relu(self.bn2(self.conv2(c)))
        c = F.relu(self.bn3(self.conv3(c)))

        pooled = self.avg_pool(c)
        pooled_flat = pooled.view(pooled.size(0), -1)

        output = self.fc(pooled_flat)

        return output
