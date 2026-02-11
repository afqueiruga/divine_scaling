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
import tqdm


try:
    from .models import build_model
except ImportError:
    from models import build_model

device = "cpu"
dtype = torch.float32


f_one_over_1_p_x2 = lambda x: 1.0 / (1.0 + x**2)


@dataclass
class ExperimentConfig:
    model_arch: str = "mlp"  # string parameter to sweep
    n_hidden: int = 64       # int parameter to sweep
    n_data: int = 2_000
    seed: int = 0
    hydra: Any = field(default_factory=lambda: {
        "run": {
            "dir": "outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}_${model_arch}_${n_hidden}",
        },
        "sweep": {
            "dir": "multirun/${now:%Y-%m-%d}/${now:%H-%M-%S}",
            "subdir": "${hydra.job.num}_${model_arch}_${n_hidden}",
        },
    })


cs = ConfigStore.instance()
cs.store(name="experiment_config", node=ExperimentConfig)

def make_1d_problem(f, N_data=2_000):
    """Generate 1D train/test data from a scalar function f."""
    X_train = 2 * np.random.rand(N_data, 1) - 1
    X_test = np.linspace(-1, 1, N_data).reshape(-1, 1)
    Y_train, Y_test = f(X_train), f(X_test)
    return (torch.from_numpy(X_train).to(device, dtype=dtype),
            torch.from_numpy(Y_train).to(device, dtype=dtype),
            torch.from_numpy(X_test).to(device, dtype=dtype),
            torch.from_numpy(Y_test).to(device, dtype=dtype))


@hydra.main(version_base=None, config_name="experiment_config")
def main(cfg: ExperimentConfig) -> None:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    X_train, Y_train, X_test, Y_test = make_1d_problem(f_one_over_1_p_x2, cfg.n_data)
    model = build_model(cfg.model_arch, cfg.n_hidden)
    criterion = nn.MSELoss()
    opt_adam = torch.optim.AdamW(model.parameters(), lr=1.0e-4, weight_decay=0.0)

    def train_step():
        opt_adam.zero_grad()
        loss = criterion(model(X_train), Y_train)
        loss.backward()
        return loss.item()

    @torch.no_grad()
    def get_test_loss(X, Y):
        return criterion(model(X), Y).item()
    
    for i in (t:=tqdm.trange(1_000)):
        loss = opt_adam.step(train_step)
        t.set_postfix(loss=loss)
    test_mse = get_test_loss(X_test, Y_test)

    output_dir = Path(HydraConfig.get().runtime.output_dir)
    metrics = {
        "model_arch": cfg.model_arch,
        "n_hidden": int(cfg.n_hidden),
        "seed": int(cfg.seed),
        "test_mse": float(test_mse),
        "train_mse": float(loss),
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    print(f"Wrote results to {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()