import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNLSTMDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(CNNLSTMDQN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        conv_out_size = self._get_conv_output(input_shape)

        self.lstm = nn.LSTM(conv_out_size, 256, batch_first=True)
        self.fc = nn.Linear(256, n_actions)

    def _get_conv_output(self, shape):
        o = self.conv1(torch.zeros(1, *shape))
        o = self.conv2(o)
        o = self.conv3(o)
        return int(np.prod(o.size()))

    def forward(self, x, hidden=None):
        batch_size = x.size(0)
        sequence_length = x.size(1)
        x = x.view(batch_size * sequence_length, x.size(2), x.size(3), x.size(4))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(batch_size, sequence_length, -1)
        x, hidden = self.lstm(x, hidden)
        x = self.fc(x[:, -1, :])
        return x, hidden

    def init_hidden(self, batch_size):
        return (torch.zeros(1, batch_size, 256), torch.zeros(1, batch_size, 256))
