import torch as th
from torch import nn as nn


class DefaultInitFun(nn.Module):

    def __init__(self,
                 h_dims,
                 param_map=nn.Identity()):
        super().__init__()
        self.h_dims = tuple(h_dims)
        self.param_map = param_map
        self.h0s = list()
        for i, dim_hidden in enumerate(self.h_dims):
            self.register_buffer(f"h0_{i}", th.zeros((dim_hidden,)))
            self.h0s.append(getattr(self, f"h0_{i}"))

    def update_prior(self, h_prior):
        assert len(h_prior) == len(self.h0s), \
            "[ERROR] Inconsistent attribute lengths"
        for i, dim_hidden in enumerate(self.h_dims):
            setattr(self, f"h0_{i}", h_prior[i])
            self.h0s[i] = h_prior[i]

    def forward(self, x, dyn):
        if callable(getattr(dyn, "state_init", None)):
            state = tuple(dyn.state_init(x))
        elif self.h0s[0].ndim == 2:
            state = tuple(h0i.clone().to(x.device) for h0i in self.h0s)
        else:
            state = tuple(
                (h0i.clone()[None, :].repeat(x.shape[0], 1).to(x.device)
                 for h0i in self.h0s))

        return self.param_map(x), state
