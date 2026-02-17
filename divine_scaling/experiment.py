from typing import Any

from dataclasses import dataclass, field
from pathlib import Path
import json

import numpy as np
import torch
import torch.nn as nn
import hydra
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
import tqdm


try:
    from .newton_optimizer import Newton, set_grad
    from .models import build_model
    from .problems import problem_factory_1d
except ImportError:
    from models import build_model
    from newton_optimizer import Newton, set_grad
    from problems import problem_factory_1d

device = "cpu"
dtype = torch.float64


@dataclass
class ExperimentConfig:
    model_arch: str = "mlp"  # string parameter to sweep
    n_hidden: int = 64       # int parameter to sweep
    activation: str = "relu"
    n_data: int = 2_000
    problem: str = "one_over_1_p_x2"
    seed: int = 0
    apply_maso_init: bool = False
    maso_init_kwargs: dict[str, Any] = field(
        default_factory=lambda: {"alternating_gates": True}
    )
    optimizer: str = "adam"
    newton_kwargs: dict[str, Any] = field(
        default_factory=lambda: {"line_search_fn": "strong_wolfe", "damping": 0.0}
    )
    hydra: Any = field(
        default_factory=lambda: {
            "run": {
                "dir": "outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}_${problem}_${model_arch}_${n_hidden}",
            },
            "sweep": {
                "dir": "multirun/${now:%Y-%m-%d}/${now:%H-%M-%S}",
                "subdir": "${hydra.job.num}__${problem}_${model_arch}_${n_hidden}",
            },
        }
    )


cs = ConfigStore.instance()
cs.store(name="experiment_config", node=ExperimentConfig)


@torch.no_grad()
def evaluate_model_on_linspace(model: nn.Module, n_points: int = 1_000) -> tuple[list[float], list[float]]:
    """Evaluate model predictions on a fixed linspace in [-1, 1]."""
    x = torch.linspace(-1.0, 1.0, n_points, device=device, dtype=dtype).reshape(-1, 1)
    y = model(x)
    return x.squeeze(-1).tolist(), y.squeeze(-1).tolist()


def optimize_adam(model, criterion, X_train, Y_train):
    opt_adam = torch.optim.AdamW(model.parameters(), lr=1.0e-4, weight_decay=0.0)
    def train_step():
        opt_adam.zero_grad()
        loss = criterion(model(X_train), Y_train)
        loss.backward()
        return loss.item()
    for i in (t:=tqdm.trange(1_000)):
        loss = opt_adam.step(train_step)
        t.set_postfix(loss=loss)
    return loss


def optimize_newton(model, criterion, X_train, Y_train, newton_kwargs, n_steps: int = 10):
    opt_newton = Newton(model, **newton_kwargs)
    for i in (t:=tqdm.trange(n_steps)):
        loss, grad_norm = opt_newton.step(criterion, X_train, Y_train)
        t.set_postfix(loss=loss)
    return loss


def optimize_newton_cascaded(
    model,
    criterion,
    X_train,
    Y_train,
    newton_kwargs,
    stages: list[str],
):
    loss = None
    for stage in stages:
        any_enabled = set_grad(model, stage, enable_all=(stage == "ALL"))
        if not any_enabled: continue
        loss = optimize_newton(model, criterion, X_train, Y_train, newton_kwargs, n_steps=100 if stage == "ALL" else 3)
    return loss


def optimize_multiplexed(model, criterion, X_train, Y_train, cfg: ExperimentConfig) -> float:
    if cfg.optimizer == "adam":
        return optimize_adam(model, criterion, X_train, Y_train)
    elif cfg.optimizer == "newton":
        return optimize_newton(model, criterion, X_train, Y_train, cfg.newton_kwargs)
    elif cfg.optimizer == "newton_only_splines":
        return optimize_newton_cascaded(model, criterion, X_train, Y_train, cfg.newton_kwargs, stages=["D", "U", "Q", "Q2"])
    elif cfg.optimizer == "newton_splines_and_gates":
        return optimize_newton_cascaded(model, criterion, X_train, Y_train, cfg.newton_kwargs, stages=["D", "U", "Q", "Q2", "G"])
    elif cfg.optimizer == "newton_layer_cascade":
        return optimize_newton_cascaded(model, criterion, X_train, Y_train, cfg.newton_kwargs, stages=["D", "U", "Q", "Q2", "G", "ALL"])
    else:
        raise ValueError(f"Optimizer {cfg.optimizer} not supported")


@hydra.main(version_base=None, config_name="experiment_config")
def main(cfg: ExperimentConfig) -> None:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    X_train, Y_train, X_test, Y_test = problem_factory_1d(cfg.problem, cfg.n_data)
    model = build_model(
        cfg.model_arch,
        cfg.n_hidden,
        cfg.activation,
        cfg.apply_maso_init,
        cfg.maso_init_kwargs,
    )
    model.to(device, dtype=dtype)
    criterion = nn.MSELoss()

    @torch.no_grad()
    def get_test_loss(X, Y):
        return criterion(model(X), Y).item()

    final_loss = optimize_multiplexed(model, criterion, X_train, Y_train, cfg)
    test_mse = get_test_loss(X_test, Y_test)
    eval_x, eval_y = evaluate_model_on_linspace(model, n_points=1_000)

    output_dir = Path(HydraConfig.get().runtime.output_dir)
    config_fields = OmegaConf.to_container(cfg, resolve=True)
    config_fields.pop("hydra", None)
    metrics = {
        **config_fields,
        "test_mse": float(test_mse),
        "train_mse": float(final_loss),
        "test_rmse": float(np.sqrt(test_mse)),
        "train_rmse": float(np.sqrt(final_loss)),
        "eval_x": [float(v) for v in eval_x],
        "eval_y": [float(v) for v in eval_y],
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    print(f"Wrote results to {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
