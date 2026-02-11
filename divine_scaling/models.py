import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def maso_init_1d(
    model: nn.Module, b_min=-1.001, b_max=1.001, flip_first_neuron=False,
    alternating_gates=False, flip_first_neuron_2=False
):
    """Initialize the model using the spline basis interpretation."""
    model.G.weight.data.fill_(1.0)
    model.G.bias.data.copy_(torch.linspace(b_min, b_max, model.G.bias.size(0)))
    if flip_first_neuron:
        model.G.weight.data[1] = -1.0
        model.G.bias.data[1] *= -1.0
    if flip_first_neuron_2:
        model.G.weight.data[1] = -1.0
        model.G.bias.data *= -1.0
        model.G.bias.data[1] *= -1.0
    if alternating_gates:
        alternating = (torch.arange(model.G.bias.size(0)) % 2 == 0).float() * 2 - 1
        model.G.weight.data *= alternating.reshape(-1, 1)
        model.G.bias.data *= - alternating


class MLP(nn.Module):
    def __init__(self, n_x: int, n_h: int, n_y: int, act=F.relu, 
                 apply_maso_init=False, maso_init_kwargs=None):
        super().__init__()
        self.G = nn.Linear(n_x, n_h)
        self.D = nn.Linear(n_h, n_y)
        self.act = act
        if apply_maso_init: maso_init_1d(self, **maso_init_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.D(self.act(self.G(x)))


class GLU(nn.Module):
    def __init__(self, n_x: int, n_h: int, n_y: int, act=F.relu, 
                 apply_maso_init=False, maso_init_kwargs=None):
        super().__init__()
        self.G = nn.Linear(n_x, n_h)
        self.U = nn.Linear(n_x, n_h)
        self.D = nn.Linear(n_h, n_y)
        self.act = act
        if apply_maso_init: maso_init_1d(self, **maso_init_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.D(self.act(self.G(x)) * self.U(x))
        
