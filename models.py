from math import sqrt

import torch
import torch as th
import torch.nn.functional as F
import torchvision.models
from torch import nn as nn
from torchdiffeq import odeint, odeint_adjoint

from dynamics.output_coordinates import DefaultOutputFun
from math import log, sqrt

class MLP(nn.Module):

    def __init__(self, n_input, n_output, continuous):
        super().__init__()
        self.continuous = continuous
        self.n_input = n_input
        self.n_output = n_output
        self.l1 = nn.Linear(n_input, 128)
        self.l2 = nn.Linear(128, 256)
        self.l3 = nn.Linear(256, n_output)
        self.activation = nn.GELU()
        self.alpha = 0

    def regularization(self):
        return 0

    def forward(self, *xs):
        if len(xs) == 1:
            x = xs[0]
        elif len(xs) == 3:
            # assume t, x, u
            input_list = []
            for i, xs_el in enumerate(xs):
                if i == 0:  # not time dependent
                    continue
                elif i == 1:
                    input_list.append(xs_el[0])  # only first element for now
                else:
                    input_list.append(xs_el)  # append control input
            x = th.cat(input_list, dim=-1)
        x = self.activation(self.l1(x))
        x = self.activation(self.l2(x))
        if self.continuous:
            return self.l3(x)
        return F.softmax(self.l3(x), dim=1)


def make_alex_net(n_in_channels, n_outputs):
    model = th.hub.load('pytorch/vision:v0.9.0', 'alexnet',
                        pretrained=False)
    model.classifier[-1] = th.nn.Linear(4096, n_outputs, True)
    model.features[0] = nn.Sequential(
        th.nn.Upsample(size=(64, 64)),
        th.nn.Conv2d(n_in_channels, 64, (11, 11), (4, 4), (2, 2)))
    return model


def make_vgg16(n_in_channels, n_outputs):
    model = th.hub.load('pytorch/vision:v0.9.0', 'vgg16',
                        pretrained=False)
    model.classifier[-1] = th.nn.Linear(4096, n_outputs)
    model.features[0] = th.nn.Conv2d(n_in_channels, 64, (3, 3),
                                     (1, 1), (1, 1))
    return model


def make_resnet50(n_in_channels, n_outputs):
    model = th.hub.load('pytorch/vision:v0.9.0', 'resnet50',
                        pretrained=False)
    model.fc = th.nn.Linear(2048, n_outputs, True)
    model.conv1 = th.nn.Conv2d(n_in_channels,
                               64, (7, 7), (2, 2), (3, 3),
                               bias=False)
    return model


def make_resnet18(n_in_channels, n_outputs):
    model = th.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=False)
    model.fc = th.nn.Linear(512, n_outputs, True)
    model.conv1 = th.nn.Conv2d(n_in_channels, 64, (7, 7), (2, 2), (3, 3), bias=False)
    return model


def resnet18_features(n_in_channels, last_chan=512):
    model = th.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=False)
    features = nn.Sequential(
        model.conv1,
        model.bn1,
        model.relu,
        model.maxpool,
        model.layer1,
        model.layer2,
        model.layer3,
        model.layer4,
    )
    # from torchvision.models.resnet
    return features

class IVP(nn.Module):
    def __init__(self,
                 n_input,
                 n_output,
                 dyn_fun,
                 init_coordinates,
                 output_fun=DefaultOutputFun(),
                 ode_tol=1e-2,
                 ts=th.linspace(0, 1, 200)):
        super().__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.dyn_fun = dyn_fun
        self.ode_tol = ode_tol
        self.register_buffer('ts', ts)
        self.output_fun = output_fun
        self.init_coordinates = init_coordinates

    def h_dot(self, t, h):
        return self.dyn_fun(t, h)

    def forward(self, x, ts=None, int_params=None, use_adjoint=False):
        solution = self.integrate(x, ts=ts, int_params=int_params,
                                  use_adjoint=use_adjoint)
        return self.output_fun(solution)[-1]

    def integrate(self, x, ts=None, int_params=None, use_adjoint=False):
        if ts is None:
            ts = self.ts
        if int_params is None:
            int_params = dict(
                rtol=self.ode_tol,
                atol=self.ode_tol
            )
        static_state, state = self.init_coordinates(x, self.dyn_fun)
        self.dyn_fun.static_state = static_state
        if use_adjoint:
            ode_call = odeint_adjoint
            # if we are differentiating the model but not training,
            # and the inputs require gradient,
            # we are probably computing adversarial robustness compute gradients
            # w.r.t to the inputs rather than the parameters.
            if not self.training and torch.is_grad_enabled() and x.requires_grad:
                int_params['adjoint_params'] = (x,)
            else:
                int_params['adjoint_params'] = tuple(self.parameters())
            int_params["adjoint_options"] = dict(norm="seminorm")
            int_params["adjoint_atol"] = int_params["atol"]
            int_params["adjoint_rtol"] = int_params["rtol"]
        else:
            ode_call = odeint
        solution = ode_call(self.h_dot, state, ts,
                            **int_params,
                            # method='dopri8',
                            # method='rk4',
                            # options=dict(step_size=self.ode_tol, perturb=True)
                            )
        return solution


class SimpleFeatureTensor(nn.Module):

    def __init__(self, last_chan, n_in_channels,
                 activation='ReLu',
                 conv_bias=False):
        super().__init__()
        act_maker = getattr(nn, activation)
        conv1_chan = last_chan // 4
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=n_in_channels, out_channels=conv1_chan,
                      kernel_size=7, stride=4, padding=7, bias=conv_bias),
            nn.BatchNorm2d(conv1_chan),
            act_maker(),
            nn.Conv2d(in_channels=conv1_chan, out_channels=conv1_chan,
                      kernel_size=5, stride=2, bias=conv_bias),
            nn.Conv2d(in_channels=conv1_chan, out_channels=last_chan,
                      kernel_size=5, stride=1, padding=2, bias=conv_bias),
            nn.BatchNorm2d(last_chan),
            act_maker(),
            nn.Conv2d(in_channels=last_chan, out_channels=last_chan,
                      kernel_size=3, stride=2, bias=conv_bias))
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

    def forward(self, x):
        return self.features(x)


class SimpleFeatures(nn.Module):
    def __init__(self, last_chan,
                 activation='ReLu',
                 n_in_channels=1,
                 bottleneck="max",
                 conv_bias=False):
        super().__init__()
        act_maker = getattr(nn, activation)

        def bottleneck_make(in_channels, out_channels, kernel_size, stride):
            if bottleneck == "conv":
                return nn.Conv2d(in_channels=in_channels,
                                 out_channels=out_channels,
                                 kernel_size=kernel_size,
                                 stride=stride,
                                 bias=conv_bias)
            elif bottleneck == "max":
                return nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
            else:
                raise RuntimeError("[ERROR] Invalid Bottleneck Value")

        conv2_chan = last_chan // 4
        conv1_chan = last_chan // 8
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=n_in_channels, out_channels=conv1_chan,
                      kernel_size=7, stride=4, padding=7, bias=conv_bias),
            nn.BatchNorm2d(conv1_chan),
            act_maker(),
            bottleneck_make(in_channels=conv1_chan, out_channels=conv1_chan,
                            kernel_size=5, stride=2),
            nn.Conv2d(in_channels=conv1_chan, out_channels=conv2_chan,
                      kernel_size=5, stride=1, padding=2, bias=conv_bias),
            nn.BatchNorm2d(conv2_chan),
            act_maker(),
            bottleneck_make(in_channels=conv2_chan, out_channels=conv2_chan,
                            kernel_size=3, stride=2),
            nn.Conv2d(in_channels=conv2_chan, out_channels=last_chan,
                      kernel_size=3, stride=1, padding=1, bias=conv_bias),
            nn.BatchNorm2d(last_chan),
            act_maker(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten())
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

    def forward(self, x):
        return self.features(x)
