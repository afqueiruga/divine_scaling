from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def maso_init_1d(
    model: nn.Module, b_min=-1.001, b_max=1.001, flip_first_neuron=False,
    alternating_gates=False, g_scale=1.0
):
    """Initialize the model using the spline basis interpretation."""
    model.G.weight.data.fill_(1.0)
    model.G.bias.data.copy_(torch.linspace(b_min, b_max, model.G.bias.size(0)))
    if flip_first_neuron:
        model.G.weight.data[1] = -1.0
        model.G.bias.data *= -1.0
        model.G.bias.data[1] *= -1.0
    if alternating_gates:
        alternating = (torch.arange(model.G.bias.size(0)) % 2 == 0).float() * 2 - 1
        model.G.weight.data *= alternating.reshape(-1, 1)
        model.G.bias.data *= - alternating
    if g_scale != 1.0:
        model.G.weight.data *= g_scale
        model.G.bias.data *= g_scale


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


def build_model(
    model_arch: str,
    n_hidden: int,
    activation: str,
    apply_maso_init: bool,
    maso_init_kwargs: dict[str, Any],
) -> nn.Module:
    activation = activation.lower()
    activation_map = {
        "relu": F.relu,
        "sigmoid": torch.sigmoid,
    }
    if activation not in activation_map:
        raise ValueError(
            f"Unknown activation='{activation}'. Use one of {list(activation_map)}"
        )
    model_map = {
        "mlp": MLP,
        "glu": GLU,
    }
    if model_arch not in model_map:
        raise ValueError(f"Unknown model_arch='{model_arch}'. Use one of {list(model_map)}")
    return model_map[model_arch](
        n_x=1,
        n_h=int(n_hidden),
        n_y=1,
        act=activation_map[activation],
        apply_maso_init=apply_maso_init,
        maso_init_kwargs=maso_init_kwargs,
    )
