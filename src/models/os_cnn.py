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
        Description:
            Forward pass of the OSCNN model. This method first permutes the dimensions of the input tensor to match
            the expected shape for 1D convolution, applies convolutional layers with ReLU and batch normalization,
            and then performs global average pooling. Finally, it feeds the pooled features into a fully connected layer.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, sequence_length, input_size).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, num_classes).
        """
        x_permuted = x.permute(0, 2, 1)

        conv1_list = [F.relu(conv(x_permuted)) for conv in self.conv1_filters]
        conv1_cat = torch.cat(conv1_list, 1)
        conv1_bn = F.relu(self.bn1(conv1_cat))

        conv2_list = [F.relu(conv(conv1_bn)) for conv in self.conv2_filters]
        conv2_cat = torch.cat(conv2_list, 1)
        conv2_bn = F.relu(self.bn2(conv2_cat))

        conv3_list = [F.relu(conv(conv2_bn)) for conv in self.conv3_filters]
        conv3_cat = torch.cat(conv3_list, 1)
        conv3_bn = F.relu(self.bn3(conv3_cat))

        pooled = self.avg_pool(conv3_bn)
        pooled_flat = pooled.view(pooled.size(0), -1)

        output = self.fc(pooled_flat)

        return output
