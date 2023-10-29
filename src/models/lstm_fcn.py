import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMFCN(nn.Module):
    def __init__(self, MAX_SEQUENCE_LENGTH, NB_CLASS, NUM_CELLS=8):
        super(LSTMFCN, self).__init__()

        # LSTM layer
        self.lstm = nn.LSTM(MAX_SEQUENCE_LENGTH, NUM_CELLS, batch_first=True)
        self.dropout = nn.Dropout(0.8)

        # CNN layers
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

        self.global_avg_pooling = nn.AdaptiveAvgPool1d(1)

        # Fully Connected layer
        self.fc = nn.Linear(NUM_CELLS + 128, NB_CLASS)  # LSTM output + CNN output

    def forward(self, x):
        # LSTM forward
        h, _ = self.lstm(x)
        h = self.dropout(
            h[:, -1, :]
        )  # Taking the last output only, similar to the Keras LSTM behavior with default return_sequences=False

        # CNN forward
        x_permuted = x.permute(0, 2, 1)  # Permute to get correct input shape for Conv1D
        c = F.relu(self.bn1(self.conv1(x_permuted)))
        c = F.relu(self.bn2(self.conv2(c)))
        c = F.relu(self.bn3(self.conv3(c)))

        c = self.global_avg_pooling(c)
        c = c.view(c.size(0), -1)  # Flatten

        # Concatenate LSTM and CNN outputs
        combined = torch.cat((h, c), dim=1)

        # Dense layer
        out = F.softmax(self.fc(combined), dim=1)

        return out


# Example usage:
MAX_SEQUENCE_LENGTH = 100  # for example
NB_CLASS = 5  # for example
model = LSTMFCN(MAX_SEQUENCE_LENGTH, NB_CLASS)
print(model)
