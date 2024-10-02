from math import e
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CNNGRU(nn.Module):
    def __init__(self, board_shape, n_actions: int, temporal_features: list[str], current_features: list[str]):
        super(CNNGRU, self).__init__()

        self.n_actions = n_actions
        self.board_shape = board_shape
        self.temporal_features = temporal_features
        self.current_features = current_features

        self._init_cnn_layers()
        self._init_gru_layers()
        self._init_fc_layers()

    def _init_cnn_layers(self):
        """
        Initialize the convolutional layers for extracting spatial features from the board state.

        The architecture consists of three convolutional layers with max pooling and dropout.
        The output of the last convolutional layer is flattened to be used as input to the gated recurrent layers.
        """
        self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(self.conv1.out_channels)
        self.dropout1 = nn.Dropout2d(p=0.0)

        self.conv2 = nn.Conv2d(self.conv1.out_channels, 64, kernel_size=5, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(self.conv2.out_channels)
        self.dropout2 = nn.Dropout2d(p=0.2)

        self.conv3 = nn.Conv2d(self.conv2.out_channels, 32, kernel_size=4, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(self.conv3.out_channels)
        self.dropout3 = nn.Dropout2d(p=0.2)

        self.conv_out_size = self._get_conv_output()

    def _init_gru_layers(self):
        # GRU input size: CNN output + temporal features
        """
        Initialize the GRU layers to capture temporal information evolving across board states.

        The input size to the GRU is the sum of the CNN output size and the number of temporal features.

        The GRU is set up to have 2 layers, with a hidden size of 256, and dropout of 0.2.

        The output size of the GRU is the hidden size, which is used as input to the fully connected layers.

        """
        self.gru_input_size = self.conv_out_size + len(self.temporal_features)

        self.gru = nn.GRU(input_size=self.gru_input_size, hidden_size=512, num_layers=2, batch_first=True, dropout=0.2)
        self.gru_out_size = self.gru.hidden_size

    def _init_fc_layers(self):
        """
        Initialize the fully connected layers for computing the Q-values.

        The input size to the first fully connected layer is the sum of the GRU output size and the number of current features.

        The first fully connected layer has 256 neurons, and the second fully connected layer has the number of actions as neurons.

        The output of the second fully connected layer is the Q-values.

        The dropout rate is set to 0.4.
        """
        fc_input_size = self.gru_out_size + len(self.current_features)

        self.fc1 = nn.Linear(fc_input_size, 256)

        self.fc2 = nn.Linear(self.fc1.out_features, self.n_actions)
        self.dropout = nn.Dropout(p=0.0)

    def forward(self, x):
        """
        The forward method of the CNN-GRU model.

        The input is a tuple of three elements: boards, temporal_features, and current_features.

        The boards are processed through the CNN, and the output is combined with the temporal features.

        The combined output is then processed through the GRU, and the last output of the GRU is combined with the current features.

        The final output is computed by passing the combined output through two fully connected layers.

        The output of the model is the Q-values.

        :param x: tuple of three elements: boards, temporal_features, and current_features
        :return: Q-values
        """
        boards, temporal_features, current_features = x
        # boards shape should be [batch_size, sequence_length, height, width]
        if boards.dim() == 3:
            boards = boards.unsqueeze(0)

        if temporal_features.dim() == 2:
            temporal_features = temporal_features.unsqueeze(0)

        # Process all boards at once
        processed_boards = self._process_cnn(boards)  # output shape: [20, 1, 13440]

        # Combine CNN output with temporal features
        gru_input = torch.cat([processed_boards, temporal_features], dim=2).float()
        # output shape: [Sequence length=20, IDK=1, 13440+len(temporal_features)]
        # TODO can we make this stack?

        # Process through GRU
        gru_out_all, _ = self.gru(gru_input)
        gru_out = gru_out_all[:, -1, :]  # Take the last output

        # Combine GRU output with current features
        combined = torch.cat([gru_out, current_features], dim=1)

        # Final FC layers
        x = F.leaky_relu(self.fc1(combined))
        x = self.dropout(x)
        q_values = self.fc2(x)

        return q_values

    def _process_cnn(self, x):
        # x shape: [batch_size, sequence_length, height, width]
        batch_size = x.size(0)
        seq_len, height, width = x.size(1), x.size(2), x.size(3)

        # Reshape to [batch_size * seq_len, 1, height, width]
        x = x.view(-1, 1, height, width)

        # Process through CNN layers
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

        # Flatten the spatial dimensions
        x = x.view(x.size(0), -1)

        # Restore batch and sequence dimensions
        return x.view(batch_size, seq_len, -1)

    def _get_conv_output(self):
        """
        Calculate the output size of the convolutional layers given the input shape.

        Parameters
        ----------
        self

        Returns
        -------
        int
            The output size of the convolutional layers
        """
        with torch.no_grad():
            o = torch.zeros(1, 1, *self.board_shape)
            o = self._process_cnn(o)
        return int(np.prod(o.size()))
