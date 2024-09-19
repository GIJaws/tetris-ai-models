import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CNNLSTMDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(CNNLSTMDQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        # Calculate the size after convolutions
        conv_out_size = self._get_conv_output(input_shape)

        self.lstm = nn.LSTM(conv_out_size, 256, batch_first=True)
        self.fc = nn.Linear(256, n_actions)

    def _get_conv_output(self, shape):
        o = self.conv1(torch.zeros(1, 1, *shape))
        o = self.conv2(o)
        o = self.conv3(o)
        return int(np.prod(o.size()))

    def forward(self, x):
        # x shape: (batch_size, sequence_length, height, width)
        batch_size, sequence_length, height, width = x.size()

        # Reshape to (batch_size * sequence_length, 1, height, width)
        x = x.view(-1, 1, height, width)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Reshape back to (batch_size, sequence_length, -1)
        conv_out = x.view(batch_size, sequence_length, -1)

        lstm_out, _ = self.lstm(conv_out)
        x = self.fc(lstm_out[:, -1, :])
        return x
