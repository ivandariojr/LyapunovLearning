from typing import Any
from math import sqrt
import numpy as np
import pytorch_lightning as pl
import torch
import torch as th
from torch import nn
from torch.autograd.functional import jvp
from torch.nn import functional as F

from models import IVP

ADAPTIVE_SOLVERS = ['dopri8', 'dopri5', 'bosh3', 'fehlberg2', 'adaptive_heun',
                    'scipy_solver']
FIXED_SOVLERS = ['euler', 'midpoint', 'rk4', 'explicit_adams',
                 'implicit_adams', 'fixed_adams']


def make_solver_params(solver_name, ode_tol):
    if solver_name in ADAPTIVE_SOLVERS:
        return dict(method=solver_name, rtol=ode_tol, atol=ode_tol)
    elif solver_name in FIXED_SOVLERS:
        return dict(
            method=solver_name,
            options=dict(
                step_size=ode_tol
            )
        )
    else:
        raise RuntimeError('[ERROR] Invalid Solver Name')


class AdversarialLearning(pl.LightningModule):

    def __init__(self, attacker, model) -> None:
        super().__init__()
        self.attacker = attacker
        self.model = model
        self.run_adv = False

    def test_step(self, batch, batch_idx):
        im, label = batch
        if self.run_adv:
            self.attacker.device = im.device
            with torch.enable_grad():
                # self.model.use_adjoint = True
                im_adv = self.attacker(im, label)
                # self.model.use_adjoint = False
            with torch.no_grad():
                net_out = self.model(im_adv)
            y_hat = net_out.argmax(dim=-1)
            error = (y_hat != label).float().mean()
            self.log('adv_test_error', error, on_epoch=True, on_step=False, logger=True)
            return error
        else:
            net_out = self.model(im)
            _, y_hat = th.max(net_out, dim=-1)
            error = (y_hat != label).float().mean()
            self.log('nominal_test_error', error, on_epoch=True, on_step=False, logger=True)
            return error


class PriorInference(pl.LightningModule):

    def __init__(self, model) -> None:
        super().__init__()
        self.model = model
        self.give_prior = False

    def test_step(self, batch, batch_idx):
        im, label = batch
        if self.give_prior:
            with torch.no_grad():
                prior = F.one_hot(label, num_classes=self.model.model.n_output)
                prior[prior == 0] = -1
                # add vectory of magnitude 5
                try:
                    max_dist = sqrt(self.model.model.n_output * self.model.h_dist_lim ** 2)
                except:
                    max_dist = sqrt(self.model.model.n_output * 15. ** 2)
                prior = F.normalize(prior.float(), p=2, dim=-1) * 0.1 * max_dist
                #add rnadom noise with std 1
                # prior += th.randn(*prior.shape).to(prior.device) * 0.1 * max_dist
                # prior = prior + .25* th.randn(*prior.shape).to(prior.device)
                # prior = prior * 10
                self.model.model.init_coordinates.update_prior([prior])
                net_out = self.model(im)
            y_hat = net_out.argmax(dim=-1)
            error = (y_hat != label).float().mean()
            self.log('prior_test_error', error, on_epoch=True, on_step=False,
                     logger=True)
            return error
        else:
            net_out = self.model(im)
            _, y_hat = th.max(net_out, dim=-1)
            error = (y_hat != label).float().mean()
            self.log('nominal_test_error', error, on_epoch=True, on_step=False, logger=True)
            return error

class GeneralLearning(pl.LightningModule):
    def __init__(self, opt_name="SGD",
                 lr=1e-3, momentum=0.9, weight_decay=1e-4,
                 decay_epochs=[30, 60, 90],
                  beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__()
        self.opt_name = opt_name
        self.lr = lr
        self.betas = (beta1, beta2)
        self.eps = eps
        self.decay_epochs = decay_epochs
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.criterion = nn.CrossEntropyLoss()
        self.criterion_per_el = nn.CrossEntropyLoss(reduction='none')

    def configure_optimizers(self):
        if self.opt_name == 'Adam':
            optimizer = th.optim.Adam(self.parameters(),
                                      lr=self.lr,
                                      weight_decay=self.weight_decay,
                                      amsgrad=False,
                                      betas=self.betas, eps=self.eps)
        elif self.opt_name == "AdamW":
            optimizer = th.optim.AdamW(self.parameters(), lr=self.lr,
                                       weight_decay=self.weight_decay,
                                       amsgrad=False,
                                       betas=self.betas, eps=self.eps)
        elif self.opt_name == 'SGD':
            optimizer = torch.optim.SGD(self.parameters(),
                                        lr=self.lr,
                                        momentum=self.momentum,
                                        weight_decay=self.weight_decay)
        elif self.opt_name == 'RMSprop':
            optimizer = torch.optim.RMSprop(self.parameters(),
                                            lr=self.lr,
                                            centered=True,
                                            momentum=self.momentum,
                                            weight_decay=self.weight_decay)
        elif self.opt_name == 'Nero':
            from libs.nero.optim.nero import Nero
            biases = list()
            weights = list()
            for name, param in self.named_parameters():
                if name .split('.')[-1] == 'bias' or (param.ndim == 2 and
                                                      param.shape):
                    biases += [param]
                else:
                    weights += [param]
            optimizer = Nero([
                dict(params=biases, constraints=False),
                dict(params=weights)],
                lr=self.lr,
                beta=self.betas[0])
        else:
            raise RuntimeError(
                f"[ERROR] Invalid Optimizer Param: {self.opt_name}")

        scheduler = th.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                               milestones=self.decay_epochs,
                                               gamma=0.1)
        lr_scheduler = {
            'scheduler': scheduler,
            'name': 'learning_rate',
            'monitor': 'training_loss',
            'interval': 'epoch',
            'frequency': 1
        }
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch
        batch_size = x.shape[0]

        loss = self.compute_loss(x, y, batch_size)

        self.log('training_loss', loss, on_step=True, on_epoch=True,
                 logger=True)
        return loss

    def compute_loss(self, x, y, batch_size):
        raise NotImplementedError('[ERROR] Abstract Method.')

    def validation_step(self, batch, batch_idxs):
        x, y = batch
        net_out = self(x)
        _, y_hat = th.max(net_out, dim=-1)
        error = (y_hat != y).float().mean()
        loss = self.criterion(net_out, y)
        self.log('validation_loss', loss, on_epoch=True, on_step=False, logger=True)
        self.log('validation_error', error, on_epoch=True, on_step=False, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        net_out = self(x)
        _, y_hat = th.max(net_out, dim=-1)
        error = (y_hat != y).float().mean()
        loss = self.criterion(net_out, y)
        self.log('test_loss', loss, on_epoch=True, on_step=False, logger=True)
        self.log('test_error', error, on_epoch=True, on_step=False, logger=True)
        return loss


class ClassicalLearning(GeneralLearning):

    def __init__(self, model: nn.Module, opt_name="SGD",
                 lr=1e-3,
                 momentum=0.9,
                 weight_decay=1e-4,
                 decay_epochs=[30, 60, 90],
                 beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(
            opt_name=opt_name, 
            lr=lr, 
            momentum=momentum,
            weight_decay=weight_decay, 
            decay_epochs=decay_epochs,
            beta1=beta1, beta2=beta2, eps=eps)
        self.model = model

    def forward(self, x):
        return self.model(x)

    def compute_loss(self, x, y, batch_size):
        return self.criterion(self(x), y)


class ODELearning(GeneralLearning):
    def __init__(self, dynamics: nn.Module,
                 output,
                 n_input,
                 n_output,
                 init_fun,
                 t_max=1.0,
                 train_ode_solver='dopri5',
                 train_ode_tol=1e-6,
                 val_ode_solver='dopri5',
                 val_ode_tol=1e-6,
                 opt_name="SGD",
                 lr=1e-3,
                 momentum=0.9,
                 weight_decay=1e-4,
                 decay_epochs=[30, 60, 90],
                 beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(
            opt_name=opt_name, lr=lr, momentum=momentum, weight_decay=weight_decay,
            decay_epochs=decay_epochs, beta1=beta1, beta2=beta2, eps=eps)
        self.t_max = t_max
        self.train_ode_solver = train_ode_solver
        self.train_ode_tol = train_ode_tol
        self.val_ode_solver = val_ode_solver
        self.val_ode_tol = val_ode_tol
        self.use_adjoint = False
        self.model = IVP(n_input=n_input,
                         n_output=n_output,
                         init_coordinates=init_fun,
                         # n_hidden=tuple(h_dims),
                         ts=th.linspace(0, t_max, 2),
                         ode_tol=train_ode_tol,
                         dyn_fun=dynamics,
                         output_fun=output)
    @property
    def train_solver_params(self):
        return make_solver_params(self.train_ode_solver,
                                  self.train_ode_tol)

    @property
    def val_solver_params(self):
        return make_solver_params(self.val_ode_solver,
                                  self.val_ode_tol)

    def forward(self, x):
        return self.model(x, ts=th.linspace(0., self.t_max, 2, device=x.device),
                          int_params=self.val_solver_params,
                          use_adjoint=self.use_adjoint)

    def compute_loss(self, x, y, batch_size):
        return self.criterion(
            self.model(x, ts=th.linspace(0., self.t_max, 2, device=x.device),
                          int_params=self.train_solver_params,
                          use_adjoint=self.use_adjoint), y)


class LyapunovLearning(ODELearning):

    def __init__(self, order, h_sample_size, h_dist_lim,
                 dynamics: nn.Module, output, n_input, n_output, init_fun,
                 t_max=1.0, train_ode_solver='dopri5', train_ode_tol=1e-6,
                 val_ode_solver='dopri5', val_ode_tol=1e-6, opt_name="SGD",
                 lr=1e-3, momentum=0.9, weight_decay=1e-4,
                 decay_epochs=[30, 60, 90], beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(dynamics, output, n_input, n_output, init_fun, t_max,
                         train_ode_solver, train_ode_tol, val_ode_solver,
                         val_ode_tol, opt_name, lr, momentum, weight_decay,
                         decay_epochs, beta1, beta2, eps)
        self.order = order
        self.h_sample_size = h_sample_size
        self.h_dims = self.model.init_coordinates.h_dims
        self.h_dist_lim = h_dist_lim
        self.h_dist = None
        self.t_dist = None
        self.h_dist_init = lambda: th.distributions.Uniform(
            th.tensor(0., device=self.device),
            th.tensor(sqrt(n_output * h_dist_lim**2), device=self.device))
        self.t_dist_init = lambda: th.distributions.Uniform(
            th.tensor(0., device=self.device),
            th.tensor(float(self.t_max), device=self.device))

    def make_samples(self, x, y, x_in, y_in, batch_size):
        if self.h_dist is None:
            self.h_dist = self.h_dist_init()
        if self.t_dist is None:
            self.t_dist = self.t_dist_init()
        h_sample_in = []
        for i, h_dim in enumerate(self.h_dims):
            h_radius = self.h_dist.sample((self.h_sample_size, 1))
            h_vec = th.randn(self.h_sample_size, h_dim, device=self.device)
            F.normalize(h_vec, p=2.0, dim=1, out=h_vec)
            h_sample = h_vec *h_radius
            h_sample.requires_grad = True
            h_sample = h_sample[None].repeat(batch_size, 1, 1).flatten(
                0, 1)
            h_sample_in += [h_sample]
        t_sample = self.t_dist.sample((1,)).squeeze()
        # t_sample = self.t_dist((batch_size, self.h_sample_size)).flatten(0,1)
        return t_sample, h_sample_in

    def compute_loss(self, x, y, batch_size):
        static_state, _ = self.model.init_coordinates(x, self.model.dyn_fun)
        x_in = static_state[:, None].expand(-1, self.h_sample_size, *((-1,)*(static_state.ndim-1))).flatten(0, 1)
        y_in = y[:, None].expand(-1, self.h_sample_size).flatten(0, 1)

        def v_ndot(order: int, t_sample, *oc_in):
            assert isinstance(order, int) and order >= 0, \
                f"[ERROR] Order({order}) must be non-negative integer."
            if order == 0:
                return F.cross_entropy(self.model.output_fun(oc_in), y_in, reduction='none')
            elif order == 1:
                return jvp(func=lambda *x: v_ndot(0, t_sample, *x),
                           inputs=tuple(oc_in),
                           v=self.model.dyn_fun.eval_dot(t_sample, tuple(oc_in), x_in),
                           create_graph=True)
            else:
                returns = tuple()
                for i in range(1, order):
                    returns += v_ndot(i, t_sample, *oc_in)
                returns += (jvp(func=lambda *x: v_ndot(order-1,t_sample, *x)[-1],
                               inputs=tuple(oc_in),
                               v=self.model.dyn_fun.eval_dot(t_sample, tuple(oc_in), x_in),
                               create_graph=True)[-1],)
                return returns

        t_samples, h_sample_in = self.make_samples(x, y, x_in, y_in, batch_size)
        if self.order == 0:
            raise NotImplementedError('[TODO] Implement this.')
        elif self.order == 1:
            v, vdot = v_ndot(1, t_samples, *h_sample_in)
            violations = th.relu(vdot + self.model.dyn_fun.max_factor * v.detach())
        elif self.order == 2:
            v, vdot, vddot = v_ndot(2, t_samples, *h_sample_in)
            violations = th.relu(vddot + 20 * vdot + 100 * v.detach())
        elif self.order == 3:
            v, vdot, vddot, vdddot = v_ndot(3, t_samples, *h_sample_in)
            violations = th.relu(vdddot + 1000 * vddot + 300 * vdot + 30 * v)
        else:
            raise NotImplementedError("[ERROR] Invalid lyapunov order.")
        violation_mask = violations > 0
        effective_batch_size = (violation_mask).sum()
        nominal_batch_size = y_in.shape[0]
        if effective_batch_size / nominal_batch_size < 0.7 and \
            self.model.dyn_fun.gain.item() > 1.0 and self.current_epoch <= 60:
            self.model.dyn_fun.gain = self.model.dyn_fun.gain.clone() * 0.9999
        elif effective_batch_size / nominal_batch_size > 0.1 and \
            self.model.dyn_fun.gain.item() < 100.0 and self.current_epoch > 60:
            self.model.dyn_fun.gain = self.model.dyn_fun.gain.clone() * 1.0001

        loss = violations.mean()
        self.log("model_gain", self.model.dyn_fun.gain, on_step=True, logger=True)
        self.log('effective_batch_size', effective_batch_size, on_step=True, logger=True)
        h_sample_in = None
        x_in = None
        y_in = None
        return loss


class PILyapunovLearning(LyapunovLearning):

    def __init__(self, t_upper, t_delta, patience, minimum_effective_batch_size,
                 order, h_sample_size, h_dist_lim,
                 dynamics: nn.Module,
                 output, n_input, n_output, init_fun,  t_max=1.0,
                 train_ode_solver='dopri5', train_ode_tol=1e-6,
                 val_ode_solver='dopri5', val_ode_tol=1e-6, opt_name="SGD",
                 lr=1e-3, momentum=0.9, weight_decay=1e-4,
                 decay_epochs=[30, 60, 90], beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(order, h_sample_size, h_dist_lim, dynamics,
                         output, n_input, n_output, init_fun, t_max,
                         train_ode_solver, train_ode_tol, val_ode_solver,
                         val_ode_tol, opt_name, lr, momentum, weight_decay,
                         decay_epochs, beta1, beta2, eps)
        self.t_upper = t_upper
        self.t_delta = t_delta
        self.patience = patience
        assert self.h_sample_size >= 3, "[ERROR] Path Integral Mode requires " \
                                        "at least three h samples in batch."
        self.minimum_effective_batch_size = minimum_effective_batch_size
        self.previous_batch_sizes = np.zeros((self.patience,)) + np.inf
        self.reached_t_max = False
        self.t_dist = th.distributions.Uniform(
            th.tensor(0., device=self.device),
            th.tensor(self.t_upper, device=self.device))

    def on_train_batch_end(self,
                           outputs: Any,
                           batch: Any,
                           batch_idx: int,
                           dataloader_idx: int) -> None:
        super().on_train_batch_end(outputs, batch, batch_idx, dataloader_idx)
        if self.reached_t_max:
            return
        logs = self.logger.callback_metrics
        eb_size = logs.get('effective_batch_size')
        self.previous_batch_sizes = np.roll(self.previous_batch_sizes, -1)
        self.previous_batch_sizes[-1] = eb_size
        has_patience = (
                self.previous_batch_sizes > self.minimum_effective_batch_size).any()
        if has_patience:
            return
        new_t_upper = self.t_upper + self.t_delta
        if new_t_upper >= self.t_max:
            new_t_upper = self.t_max
            self.reached_t_max = True
        self.t_upper = new_t_upper
        self.t_dist = th.distributions.Uniform(
            th.tensor(0., device=self.t_dist.low.device),
            th.tensor(new_t_upper, device=self.t_dist.high.device)
        )

    def make_samples(self, x, y, x_in, y_in, batch_size):
        #TODO: Fix this for const state x
        # because zero probability events indeed happen
        while True:
            ts = th.cat([th.tensor([th.finfo(x.dtype).tiny],
                                   device=self.device),
                         self.t_dist.sample(
                             (self.h_sample_size - 2,)).sort()[0],
                         th.tensor([self.t_upper], device=self.device)],
                        dim=0)
            # time hast o be strictly increasing or strictly decreasing.
            if (ts[1:] > ts[:-1]).all():
                break
        with torch.no_grad():
            state_sample = self.model.integrate(x, ts,
                                                self.train_solver_params)
        h_sample_in = []
        for i, state_sample in enumerate(state_sample):
            h_sample_in.append(
                state_sample.transpose(1, 0).flatten(0, 1))
            # h_sample_in[-1] += self.h_dist.sample(h_sample_in[-1].shape)
            h_sample_in[-1].requires_grad = True
        return h_sample_in

    def compute_loss(self, x, y, batch_size):
        self.log('t_upper', self.t_upper, on_step=True, logger=True)
        return super().compute_loss(x, y, batch_size)

from libs.ContinuousNet.continuous_net.continuous_net import ContinuousNet

class ContinuousNetLyapunovLearning(GeneralLearning):
    def __init__(self, model: ContinuousNet,
                 order = 0,
                 opt_name="SGD", lr=1e-3,
                 momentum=0.9, weight_decay=1e-4,
                 decay_epochs=[30, 60, 90], beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(opt_name, lr, momentum, weight_decay, decay_epochs,
                         beta1, beta2, eps)
        assert order in [0, 1], f"[ERROR] Invalid order: {order}."
        self.model = model
        self.order = order
        self.delta_t = self.model.dyns[0].ts[1] - self.model.dyns[0].ts[0]
        self.disc_alpha = 1 - th.exp((-10/3) * self.delta_t)

    def forward(self, x):
        return self.model(x)

    def compute_loss(self, x, y, batch_size):
        self.model(x)

        if self.order == 0:
            yts = self.model.state_traj_output()
            lya_truth = y.expand(yts.shape[0], -1).flatten(0, 1)
            lya_state = yts.flatten(0, 1)
            v = self.criterion_per_el(lya_state, lya_truth).unflatten(0, yts.shape[:2])
            v_delta = v[1:] - v[:-1].detach()
            violation = torch.relu(v_delta + self.disc_alpha * v[:-1].detach()).mean()
        else:
            assert self.order == 1, f"[ERROR] Invalid order: {self.order}."
            yts, yts_dot = self.model.full_traj_output()
            lya_truth = y.expand(yts.shape[0], -1).flatten(0, 1)
            lya_state = yts.flatten(0, 1)
            v, vdot = torch.autograd.functional.jvp(
                func=lambda x: self.criterion_per_el(x, lya_truth),
                inputs=(lya_state,),
                v=yts_dot.flatten(0,1),
                create_graph=True)
            v_delta = v[1:] - v[:-1].detach()
            violation0 = torch.relu(v_delta + self.disc_alpha * v[:-1].detach())
            violation1 = torch.relu(vdot + (10/3) * v.detach())
            violation = violation0.mean() * violation1.mean()
            self.log('Mean v_dot', vdot.mean(), on_step=True, on_epoch=True)
        self.log('Mean v', v.mean(), on_step=True, on_epoch=True)
        self.log('Mean v_delta', v_delta.mean(), on_step=True, on_epoch=True)
        return violation