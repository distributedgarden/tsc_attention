import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.attention import SelfAttention


class AttentionOSCNN(nn.Module):
    """
    Description:
        - Omni-Scale 1 Dimensional Convolutional Neural Network (OS-CNN) with Self-Attention
    """

    def __init__(self, input_length, num_classes):
        super(AttentionOSCNN, self).__init__()

        # convolutional layers with multiple filter sizes
        self.conv1_filters = nn.ModuleList(
            [nn.Conv1d(1, 32, kernel_size=ks) for ks in [1, 2, 3, 5, 7, 11]]
        )
        self.conv2_filters = nn.ModuleList(
            [nn.Conv1d(192, 64, kernel_size=ks) for ks in [1, 2, 3, 5, 7, 11]]
        )
        self.conv3_filters = nn.ModuleList(
            [nn.Conv1d(384, 128, kernel_size=ks) for ks in [1, 2]]
        )

        # self-attention layer
        self.attention = SelfAttention(256)

        # fully connected layer
        self.fc = nn.Linear(256, num_classes)

        self.attention_weights = None

    def forward(self, x):
        # convolutional layers with batch normalization and ReLU activation
        conv1_list = [F.relu(conv(x)) for conv in self.conv1_filters]
        conv1_cat = torch.cat(conv1_list, 1)
        conv1_bn = F.relu(nn.BatchNorm1d(192)(conv1_cat))

        conv2_list = [F.relu(conv(conv1_bn)) for conv in self.conv2_filters]
        conv2_cat = torch.cat(conv2_list, 1)
        conv2_bn = F.relu(nn.BatchNorm1d(384)(conv2_cat))

        conv3_list = [F.relu(conv(conv2_bn)) for conv in self.conv3_filters]
        conv3_cat = torch.cat(conv3_list, 1)
        conv3_bn = F.relu(nn.BatchNorm1d(256)(conv3_cat))

        # self-attention
        attended, attention_weights = self.attention(conv3_bn.transpose(1, 2))
        attended_sum = torch.sum(attended, dim=1)

        self.attention_weights = attention_weights

        # fully connected layer
        output = self.fc(attended_sum)

        return output


def example():
    """
    Description:
        - example usage
    """
    model = AttentionOSCNN(input_length=125, num_classes=5)
    print(model)
