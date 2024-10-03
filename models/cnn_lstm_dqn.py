import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing_extensions import deprecated


@deprecated("Use cnn_lstm.py instead")
class CNNLSTMDQN(nn.Module):
    def __init__(self, input_shape, n_actions, n_features=11):
        super(CNNLSTMDQN, self).__init__()

        # Existing CNN layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.dropout1 = nn.Dropout2d(p=0.1)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.dropout2 = nn.Dropout2d(p=0.2)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.dropout3 = nn.Dropout2d(p=0.3)

        self.conv_out_size = self._get_conv_output(input_shape)

        # LSTM for processing CNN output
        self.lstm = nn.LSTM(self.conv_out_size, 256, batch_first=True)

        # Linear layer for processing additional features
        self.features_fc = nn.Linear(n_features, 64)

        # Combine LSTM output and processed features
        self.combine_fc = nn.Linear(256 + 64, 128)

        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(128, n_actions)

    def _get_conv_output(self, shape) -> int:
        with torch.no_grad():
            o = self.conv1(torch.zeros(1, 1, *shape))
            o = self.bn1(o)
            o = F.leaky_relu(o)  # Changed to LeakyReLU
            o = self.dropout1(o)

            o = self.conv2(o)
            o = self.bn2(o)
            o = F.leaky_relu(o)  # Changed to LeakyReLU
            o = self.dropout2(o)

            o = self.conv3(o)
            o = self.bn3(o)
            o = F.leaky_relu(o)  # Changed to LeakyReLU
            o = self.dropout3(o)

            return int(np.prod(o.size()))

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Changed nonlinearity to 'leaky_relu'
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
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
        state, features = x
        batch_size, seq_len, height, width = state.size()

        # Process the board state through CNN and LSTM
        x = state.view(batch_size * seq_len, 1, height, width)
        x = self.process_cnn(x)
        x = x.view(batch_size, seq_len, -1)
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # Take the last output

        # Process additional features
        features = F.relu(self.features_fc(features))

        # Combine LSTM output and processed features
        combined = torch.cat((lstm_out, features), dim=1)
        combined = F.relu(self.combine_fc(combined))

        # Final output
        q_values = self.fc(self.dropout(combined))

        return q_values

    def process_cnn(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.leaky_relu(x)
        x = self.dropout3(x)

        return x.view(x.size(0), -1)
