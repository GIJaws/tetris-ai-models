import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CNNLSTMDQN(nn.Module):
    def __init__(self, input_shape, n_actions, sequence_length=4, dropout=0.5):
        super(CNNLSTMDQN, self).__init__()
        self.sequence_length = sequence_length
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        conv_out_size = self._get_conv_output(input_shape)

        self.lstm = nn.LSTM(conv_out_size, 256, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(256, n_actions)

        self._initialize_weights()

    def _get_conv_output(self, shape):
        with torch.no_grad():
            o = self.conv1(torch.zeros(1, 1, *shape))
            o = self.bn1(o)
            o = F.relu(o)
            o = self.conv2(o)
            o = self.bn2(o)
            o = F.relu(o)
            o = self.conv3(o)
            o = self.bn3(o)
            o = F.relu(o)
            return int(np.prod(o.size()))

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if "weight_ih" in name:
                        nn.init.xavier_uniform_(param.data)
                    elif "weight_hh" in name:
                        nn.init.orthogonal_(param.data)
                    elif "bias" in name:
                        nn.init.constant_(param.data, 0)

    def forward(self, x):
        batch_size, seq_len, height, width = x.size()
        x = x.view(batch_size * seq_len, 1, height, width)

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        conv_out = x.view(batch_size, seq_len, -1)
        lstm_out, _ = self.lstm(conv_out)
        lstm_out = self.dropout(lstm_out[:, -1, :])
        x = self.fc(lstm_out)
        return x
