import math
from math import sqrt, log

import torch as th
import torch.nn.functional as F
from torch import nn as nn

class MLPDyn(nn.Module):

    def __init__(self, n_input, n_output, continuous):
        super().__init__()
        self.continuous = continuous
        self.n_input = n_input
        self.n_output = n_output
        self.l1 = nn.Linear(n_input, 128)
        self.l2 = nn.Linear(128, 256)
        self.l3 = nn.Linear(256, n_output)
        hdist_lim = 15
        self.register_buffer("gain", th.ones((1,)) *100.0)
        self.max_factor = log(sqrt(n_output * hdist_lim ** 2))
        self.activation = nn.GELU()
        self.alpha = 0

    def regularization(self):
        return 0

    def eval_dot(self, t, h_tuple, x):
        x=th.cat([h_tuple[0], x], dim=-1)
        x = self.activation(self.l1(x))
        x = self.activation(self.l2(x))
        if self.continuous:
            return self.l3(x)
        return F.softmax(self.l3(x), dim=1) * self.max_factor

    def forward(self, t, h_tuple):
        return self.eval_dot(t, h_tuple, self.static_state)


class VaryingFeatureMap(nn.Module):

    def __init__(self, n_h, in_channels, out_channels, width, padding,
                 n_hidden=128):
        super().__init__()
        self.t_to_hidden_t = th.nn.Linear(1, 128)
        self.hidden_t_to_wieght = th.nn.Linear(128,
                                               out_channels * in_channels * width * width)

        self.h_to_hidden_h = th.nn.Linear(n_h, 128)
        self.hidden_h_to_bias = th.nn.Linear(128, out_channels * width * width)
        self.dropout = nn.Dropout(0.5)
        self.weight_shape = (out_channels, in_channels, width, width)
        self.bias_shape = (out_channels, width, width)
        self.activation = nn.PReLU()
        self.padding = padding

    def forward(self, t, h, img):
        hidden_t = self.t_to_hidden_t(t[None])
        hidden_t = self.dropout(hidden_t)
        weight = self.hidden_t_to_wieght(self.activation(hidden_t)).view(
            *self.weight_shape)
        weight = self.dropout(weight)

        hidden_h = self.h_to_hidden_h(h)
        hidden_h = self.dropout(hidden_h)
        bias = self.hidden_h_to_bias(self.activation(hidden_h)).view(-1,
                                                                     *self.bias_shape)

        out = F.conv2d(input=img,
                       weight=weight,
                       padding=self.padding,
                       stride=1)
        return out + bias


class VaryingClassDyn(nn.Module):
    def __init__(self, n_in_channels=1,
                 n_hidden=10,
                 activation='ReLu',
                 dropout=0.5,
                 mlp_size=128,
                 last_chan=64
                 ):
        super().__init__()

        act_maker = getattr(nn, activation)

        self.variational = VaryingFeatureMap(
            n_h=n_hidden, in_channels=n_in_channels, out_channels=last_chan,
            width=3, padding=1, n_hidden=128)
        self.var_to_last_chan = nn.Sequential(
            nn.BatchNorm2d(last_chan),
            # act_maker(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=last_chan, out_features=n_hidden),
            # nn.Linear(in_features=last_chan + n_hidden, out_features=mlp_size),
            # nn.Dropout(dropout),
            # act_maker(),
            # nn.Linear(in_features=mlp_size, out_features=mlp_size//2),
            # nn.Dropout(dropout),
            # act_maker(),
            # nn.Linear(in_features=mlp_size//2, out_features=n_hidden)
        )
        self.static_state = None
        self.output_dim = n_hidden
        self.apply(self._init_parameters)

    def _init_parameters(self, m):
        with th.no_grad():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def eval_dot(self, t, h_tuple, x):
        h = h_tuple[0]
        features = self.variational(t, h, x)
        features = self.var_to_last_chan(features)
        # return self.classifier(th.cat([ features.flatten(1, -1), h], dim=-1))
        return self.classifier(features.flatten(1, -1))

    def forward(self, t, h_tuple):
        assert self.static_state is not None, "[ERROR] You forgot to set " \
                                              "static state before calling " \
                                              "forward."
        return self.eval_dot(t, h_tuple, self.static_state)



class ClassDyn(nn.Module):
    def __init__(self, n_hidden=10,
                 activation='ReLu',
                 dropout=0.5,
                 mlp_size=128,
                 n_param_features=64,
                 gain=50,
                 restrict_to_simplex=True
                 ):
        super().__init__()

        act_maker = getattr(nn, activation)

        # self.classifier = nn.Sequential(
        #     nn.Linear(in_features=n_param_features + n_hidden,
        #               out_features=mlp_size),
        #     nn.Dropout(dropout),
        #     act_maker(),
        #     nn.Linear(in_features=mlp_size, out_features=mlp_size // 2),
        #     nn.Dropout(dropout),
        #     act_maker(),
        #     nn.Linear(in_features=mlp_size // 2, out_features=n_hidden))
        self.hidden_to_mlp = nn.Linear(in_features=n_hidden,
                                       out_features=mlp_size)
        self.activation = act_maker()
        self.dropout = nn.Dropout(dropout)
        self.mlp_to_hidden = nn.Linear(in_features=mlp_size,
                                       out_features=n_hidden)
        self.mlp_size = mlp_size
        self.output_dim = n_hidden
        h_dist_lim = 15
        self.register_buffer("class_simplex", -th.ones(n_hidden,n_hidden)*h_dist_lim)
        self.register_buffer("gain", th.ones((1,))*gain)
        self.class_simplex.scatter_(1, th.arange(n_hidden)[:, None], 1.*h_dist_lim)


        self.max_factor = log(sqrt(n_hidden*h_dist_lim**2) / 1e-0)
        self.static_state = None
        self.restrict_to_simplex=restrict_to_simplex
        self.apply(self._init_parameters)

    def _init_parameters(self, m):
        with th.no_grad():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def eval_dot(self, t, h_tuple, x):
        h = h_tuple[0]
        h_dot_raw = self.hidden_to_mlp(h)
        h_dot_raw = self.activation(self.dropout(h_dot_raw))
        h_dot_raw = th.bmm(x.view(h.shape[0], round(sqrt(x.shape[1])),
                                  round(sqrt(x.shape[1]))),
                           h_dot_raw.unsqueeze(-1))[:, :, 0]
        h_dot_raw = self.activation(self.dropout(h_dot_raw))
        h_dot_raw = self.mlp_to_hidden(h_dot_raw)
        diff = self.class_simplex[None] - h[:, None, :]
        # h_dot_raw = self.classifier(th.cat([x,h], dim=-1))

        if self.restrict_to_simplex:
            h_dot_raw = h_dot_raw.softmax(dim=-1)
            # h_dot_raw = F.normalize(h_dot_raw.abs(), p=1, dim=-1)
            h_dot_raw = (diff * h_dot_raw.unsqueeze(-1)).mean(dim=1)
            return h_dot_raw * self.max_factor * self.gain
        else:
            h_dot_normed = F.normalize(h_dot_raw, p=2, dim=-1)
            dots = th.bmm(diff, h_dot_normed[:, :, None])[:, :, 0]
            distances_normed = th.cdist(h, self.class_simplex)
            # norm_weight = (distances_normed *
            #                F.normalize(dots - dots.min(dim=-1).values[:, None], p=1, dim=1)).mean(dim=-1)[:, None]
            norm_weight = (distances_normed * dots.softmax(dim=1)).mean(dim=-1)[:, None]

            return self.max_factor * \
                   norm_weight * \
                   h_dot_normed * self.gain

    def forward(self, t, h_tuple):
        assert self.static_state is not None, "[ERROR] You forgot to set " \
                                              "static state before calling " \
                                              "forward."
        return self.eval_dot(t, h_tuple, self.static_state)
