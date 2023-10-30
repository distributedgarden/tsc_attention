import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMFCN(nn.Module):
    """
    Description:
        - Hybrid Long Short-Term Memory Fully Convolutional Network (LSTM-FCN)
    """

    def __init__(self, MAX_SEQUENCE_LENGTH, NB_CLASS, NUM_CELLS=8):
        super(LSTMFCN, self).__init__()

        # LSTM layer with dropout
        self.lstm = nn.LSTM(MAX_SEQUENCE_LENGTH, NUM_CELLS, batch_first=True)
        self.dropout = nn.Dropout(0.8)

        # CNN layers with batch normalization
        self.conv1 = nn.Conv1d(
            1, 128, kernel_size=8, padding="same", padding_mode="zeros"
        )
        self.bn1 = nn.BatchNorm1d(128)

        self.conv2 = nn.Conv1d(
            128, 256, kernel_size=5, padding="same", padding_mode="zeros"
        )
        self.bn2 = nn.BatchNorm1d(256)

        self.conv3 = nn.Conv1d(
            256, 128, kernel_size=3, padding="same", padding_mode="zeros"
        )
        self.bn3 = nn.BatchNorm1d(128)

        # global average pooling
        self.global_avg_pooling = nn.AdaptiveAvgPool1d(1)

        # fully connected layer
        # LSTM output + CNN output concatenation
        self.fc = nn.Linear(NUM_CELLS + 128, NB_CLASS)

    def forward(self, x):
        """
        Description:
            - LSTM forward pass
            - CNN forward pass
        """
        # LSTM with dropout
        h, _ = self.lstm(x)
        h = self.dropout(h[:, -1, :])

        # CNN forward
        # permute to get the correct input shape for Conv1D
        x_permuted = x.permute(0, 2, 1)
        c = F.relu(self.bn1(self.conv1(x_permuted)))
        c = F.relu(self.bn2(self.conv2(c)))
        c = F.relu(self.bn3(self.conv3(c)))

        # global average pooling
        c = self.global_avg_pooling(c)

        # flatten
        c = c.view(c.size(0), -1)

        # concatenate LSTM and CNN outputs
        combined = torch.cat((h, c), dim=1)

        # Dense layer
        out = F.softmax(self.fc(combined), dim=1)

        return out


def example():
    """
    Description:
        - example usage
    """
    max_sequence_length = 100
    nb_class = 5
    model = LSTMFCN(max_sequence_length, nb_class)
    print(model)
