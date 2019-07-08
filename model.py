import torch
import torch.nn as nn
import torch.nn.functional as F

# from MuVAN.utils import hybrid_focus


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

"""
class MuVAN(nn.Module):
    def __init__(self):
        super(MuVAN, self).__init__()
        self.bgru = nn.GRU(23, 64, batch_first=True, bidirectional=True, num_layers=2)
        self.energy = Energy(2*64, 50)
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Linear()

    def forward(self, inputs):
        e_out = []
        output, hidden = self.bgru(inputs)

        o_t = output[:, -1, :]
        o_t_1 = output[:, :-1, :]

        for i in range(output.shape[1]):
            e_out.append(self.energy(output[:, i, :]))

        a_t = hybrid_focus(e_out)

        c_t = torch.bmm(a_t, output[:, :-1, :])

        stacks = torch.stack([output[:, -1, :], c_t])
        stacks = self.conv(stacks)
        stacks = self.fc(stacks)
        return torch.softmax(stacks, dim=1)


class Net(nn.Module):
    def __init__(self, in_size, h_size):
        super(Net, self).__init__()
        self.f_gru_1 = nn.GRUCell(in_size, h_size)
        self.f_gru_2 = nn.GRUCell(h_size, h_size)

    def forward(self, x):
        out_seq = []
        tmp = []

        for i in range(x.shape[1]):
            f_1_h = self.f_gru_1(x[:, i, :])
            tmp.append(f_1_h)

        tmp = torch.stack(tmp, 1)

        for t in range(tmp.shape[1]):
            f_2_h = self.f_gru_2(tmp[:, t, :])
            out_seq.append(f_2_h)

        out = torch.stack(out_seq, dim=1)

        return out_seq, out
"""


class MuVANminus(nn.Module):
    def __init__(self, h_size, fc_in, n_featreus):
        super(MuVANminus, self).__init__()
        self.fc_in = fc_in

        self.rnn = nn.ModuleList([
            nn.GRU(1, h_size, batch_first=True, bidirectional=True, num_layers=2) for _ in range(n_featreus)
        ])
        self.conv = nn.Sequential(
            nn.Conv2d(256, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(fc_in, 12)

    def forward(self, x):
        arry = []

        for i, l in enumerate(self.rnn):
            tmp = x[:, :, i].unsqueeze(2)
            o, _ = l(tmp)
            arry.append(o)

        arry = torch.stack(arry, 3)
        output = self.conv(arry)
        output = output.view((-1, self.fc_in))

        output = self.fc(output)
        return output
