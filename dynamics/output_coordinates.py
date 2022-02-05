from torch import nn as nn


class DefaultOutputFun(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, h):
        return h[-1]

class FirstNOutput(nn.Module):
    def __init__(self, out_size):
        super().__init__()
        self.out_size = out_size

    def forward(self, h):
        #(h = (t, batch, dim)
        return h[-1][:, :, :self.out_size]


class LinearLastOutput(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.fc = nn.Linear(in_features=in_size, out_features=out_size, bias=False)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, h):
        return self.fc(h[-1])