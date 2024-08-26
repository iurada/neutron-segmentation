import random
import numpy as np

import torch
import torch.nn as nn

class Injector:
    def __init__(self, error_model: str = 'random', p: float = 0.3, train_aware: bool = True):
        self.error_model = error_model
        self.p = p
        self.train_aware = train_aware

        self.injection_hooks = []
        self.total_ops = 0
        self.inject = True
        self.mask_types = ['line', 'block', 'random']

        self.first_hook = None


    @property
    def random_relative_error(self) -> float:
        r"""Generator for relative errors to be injected on the training
        We have seen in the past relative error distributions that follow a Power Law PDF
        so we will use the approach proposed at https://arxiv.org/abs/1208.3524
        We will implement the function based on https://stats.stackexchange.com/a/406705
        Example:
        x_min, alpha, r = 5, 2.5, random()
        relative_error = x_min * (1 - r) ** (-1 / (alpha - 1))
        :return: the calculated relative_error
        """
        # TODO: Generalize the random generation to the values observed on GEMM output
        # Power Law parameters for the Functional Units
        power_law_fus = [
            (1.0728769e-07, 1.0868737), (2.0230031, 1.0568325), (8.1847715e-08, 1.082071), (136027.72, 27.1194),
            (3.0, 1.0678725), (0.03517608, 1.189603), (3.4028237e+38, 443107.0), (2.0, 1.4543958),
            (0.010238367, 1.1181921), (1.396856e-09, 1.0846596), (2.6865074e-10, 1.0769672), (1.3970158e-09, 1.085144),
            (0.66699225, 23.798765), (0.66699225, 23.798765), (0.66699225, 23.922783), (0.75000001, 121435080.0),
            (0.61141304, 3.4316596), (0.75000001, 121435080.0), (0.0, 1.08212), (7.0958774e-08, 1.082116),
            (0.0, 1.08212)
        ]

        alpha, x_min = random.choice(power_law_fus)
        r = random.random()
        relative_error = x_min * (1 - r) ** (-1 / (alpha - 1))
        return relative_error


    def training_error(self, x):
        error = torch.rand(size=(1,), device=x.device, dtype=x.dtype) * 6 + 1e-6  #* max(1, epoch, 1)
        if random.randint(0, 1):
            return error
        return - error


    def generate_injection(self, x, p, training):
        # We can inject the relative errors using only Torch built-in functions
        # Otherwise it is necessary to use AutoGrads

        if training:
            error = self.training_error(x)
            error_model = self.error_model
        else:
            error = self.random_relative_error
            error_model = np.random.choice(self.mask_types)

        mask = generate_mask(x, error_model, p).to(x.device)
        x = (x * mask).mul(error) + (x * ~mask)

        return x


    def injector_hook(self, module: nn.Module, input: torch.Tensor, output: torch.Tensor):

        if self.idx_counter == self.total_ops:
            self.idx_to_inject = np.random.randint(0, self.total_ops)
            self.idx_counter = 0

        if self.inject and self.idx_counter == self.idx_to_inject:

            # Train-Aware Injections
            if self.train_aware and module.training:
                output = self.generate_injection(output, self.p, training=True)

            # Evaluation Injections
            elif not module.training:
                output = self.generate_injection(output, 1.0, training=False)

        self.idx_counter += 1
        return output


    def hook_model(self, model: nn.Module, modules=(nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.ReLU)):
        for m in model.modules():
            if isinstance(m, modules):
                self.injection_hooks.append(m.register_forward_hook(self.injector_hook))
                self.total_ops += 1
        self.idx_counter = self.total_ops


    def remove_hooks(self):
        for h in self.injection_hooks:
            h.remove()


def generate_mask(x, error_type='block', p=1.0):
    shape = x.shape

    if len(shape) == 4:
        b, c, h, w = shape
        mask = generate_4D_mask(b, c, h, w, error_type, p)
    elif len(shape) == 3:
        b, h, w = shape
        mask = generate_3D_mask(b, h, w, p)
    else:
        b, c = shape
        mask = generate_2D_mask(b, c, p)

    return torch.tensor(mask > 0)


def generate_4D_mask(b, c, h, w, error_type, p):
    mask4D = torch.zeros((b, c, h, w))
    mask3D = torch.zeros((c, h, w))

    if error_type == 'line':
        mask2D = generate_line_mask(h, w)
        channel_p = 0.75
    elif error_type == 'block':
        mask2D = generate_block_mask(h, w)
        channel_p = 0.3
    elif error_type == 'random':
        mask2D = generate_random_mask(h, w)
        channel_p = 0.1

    samples = torch.bernoulli(torch.ones(b) * p) > 0 if p != 1 else torch.ones(b) > 0
    channels = torch.bernoulli(torch.ones(c) * channel_p) > 0

    mask3D[channels] += mask2D
    mask4D[samples] += mask3D

    return mask4D


def generate_2D_mask(b, c, p):
    mask2D = torch.zeros((b, c))
    mask2D[torch.bernoulli(torch.ones(b) * p) > 0, torch.bernoulli(torch.ones(c) * 0.3) > 0] = 1
    return mask2D

def generate_3D_mask(b, h, w, p):
    #mask3D = torch.zeros((b, h, w))
    #idx_b = torch.bernoulli(torch.ones(b) * p) > 0
    #idx_h = torch.bernoulli(torch.ones(h) * 0.3) > 0
    #idx_w = torch.bernoulli(torch.ones(w) * 0.3) > 0

    mask3D = torch.bernoulli(torch.ones(b, h, w) * 0.3)
    mask3D[mask3D > 0] = 1
    mask3D[mask3D != 1] = 0

    return mask3D


def generate_line_mask(h, w):
    mask = torch.zeros((h, w))
    if torch.bernoulli(torch.ones(1) * .5):
        # columns
        column = torch.randint(0, h, (1,))
        mask[column, :] = 1
    else:
        # rows
        row = torch.randint(0, w, (1,))
        mask[:, row] = 1
    return mask


def generate_block_mask(h, w):
    mask = torch.zeros((h, w))
    h0, h1, w0, w1 = torch.randint(0, h, (1,)), torch.randint(0, h, (1,)), \
                     torch.randint(0, w, (1,)), torch.randint(0, w, (1,))
    h0, h1 = min(h0, h1), max(h0, h1)
    w0, w1 = min(w0, w1), max(w0, w1)

    mask[h0:h1, w0:w1] = 1
    return mask


def generate_random_mask(h, w):
    mask = torch.rand((h, w)) > 0.5
    return mask
