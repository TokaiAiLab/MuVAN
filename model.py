import torch
import torch.nn as nn
import torch.nn.functional as F

from MuVAN.utils import hybrid_focus


class Energy(nn.Module):
    def __init__(self, in_features, output_features):
        super(Energy, self).__init__()
        self.in_features = in_features
        self.output_features = output_features

        w = torch.empty(in_features, output_features)
        self.weight = nn.Parameter(nn.init.kaiming_normal_(w))
        self.bias = nn.Parameter(torch.empty(in_features))

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


class MuVAN(nn.Module):
    def __init__(self, input_size):
        super(MuVAN, self).__init__()
        self.bgru = nn.GRU(input_size, 64, batch_first=True, bidirectional=True)
        self.energy = Energy(2*64, 50)
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, 2),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Linear()

    def forward(self, inputs):
        e_out = []
        output, hidden = self.bgru(inputs)
        for i in range(output.shape[1]):
            e_out.append(self.energy(output[:, i, :]))

        a_t = hybrid_focus(e_out)

        c_t = torch.bmm(a_t, output[:, :-1, :])

        stacks = torch.stack([output[:, -1, :], c_t])
        stacks = self.conv(stacks)
        stacks = self.fc(stacks)
        return torch.softmax(stacks, dim=1)
