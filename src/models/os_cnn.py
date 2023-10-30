import torch
import torch.nn as nn
import torch.nn.functional as F


class OSCNN(nn.Module):
    """
    Description:
        - Omni-Scale 1 Dimensional Convolutional Neural Network (OS-CNN)
        - based on the architecture proposed by Tang et al.
    """

    def __init__(self, input_length: int, num_classes: int):
        super(OSCNN, self).__init__()

        # convolution layers with filter sizes
        self.conv1_filters = nn.ModuleList(
            [nn.Conv1d(1, 32, kernel_size=ks) for ks in [1, 2, 3, 5, 7, 11]]
        )
        self.conv2_filters = nn.ModuleList(
            [nn.Conv1d(192, 64, kernel_size=ks) for ks in [1, 2, 3, 5, 7, 11]]
        )
        self.conv3_filters = nn.ModuleList(
            [nn.Conv1d(384, 128, kernel_size=ks) for ks in [1, 2]]
        )

        # global average pooling
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        # fully connected layer
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        """
        description:
            - forward pass
            - 3 convolution layers with batch normalization
            - global average pooling
            - fully connected layer
        """
        # convolution layers with batch normalization
        conv1_list = [F.relu(conv(x)) for conv in self.conv1_filters]
        conv1_cat = torch.cat(conv1_list, 1)
        conv1_bn = F.relu(nn.BatchNorm1d(192)(conv1_cat))

        conv2_list = [F.relu(conv(conv1_bn)) for conv in self.conv2_filters]
        conv2_cat = torch.cat(conv2_list, 1)
        conv2_bn = F.relu(nn.BatchNorm1d(384)(conv2_cat))

        conv3_list = [F.relu(conv(conv2_bn)) for conv in self.conv3_filters]
        conv3_cat = torch.cat(conv3_list, 1)
        conv3_bn = F.relu(nn.BatchNorm1d(256)(conv3_cat))

        # global average pooling
        pooled = self.avg_pool(conv3_bn)
        pooled_flat = pooled.view(pooled.size(0), -1)

        # fully connected layer
        output = self.fc(pooled_flat)

        return output


def example():
    """
    Description:
        - example usage
    """
    model = OSCNN(input_length=125, num_classes=5)
    print(model)
