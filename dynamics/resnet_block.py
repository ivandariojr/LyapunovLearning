import torch as th
from torch import nn as nn


class ResNetBlockDyn(nn.Module):
    def __init__(self, n_in_channels=1, n_hidden=10):
        super().__init__()
        n_hidden_channels = 64
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=n_in_channels, out_channels=n_hidden_channels,
                      kernel_size=7,
                      stride=2, padding=3, bias=False),
            nn.BatchNorm2d(n_hidden_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=7, stride=2),
            nn.Conv2d(n_hidden_channels, n_hidden_channels, 3, 1, padding=1,
                      bias=False),
            nn.BatchNorm2d(n_hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_hidden_channels, n_hidden_channels, 3, 1, padding=1,
                      bias=False),
            nn.BatchNorm2d(n_hidden_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=n_hidden_channels + n_hidden,
                      out_features=128),
            nn.GELU(),
            nn.Linear(in_features=128, out_features=128),
            nn.GELU(),
            nn.Linear(in_features=128, out_features=n_hidden),
        )
        self.apply(self._init_parameters)

    def _init_parameters(self, m):
        with th.no_grad():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
                # nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data,
                                       gain=nn.init.calculate_gain('relu'))
                nn.init.constant_(m.bias.data, 0)

    def forward(self, h, x):
        x = self.layer1(x).flatten(1, -1)
        return self.classifier(th.cat([x, h], dim=-1))