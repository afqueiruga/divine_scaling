import hydra
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
import tqdm

try:
    from .models import GLU, MLP
except ImportError:
    from models import GLU, MLP

device = "cpu"
dtype = torch.float32


f_one_over_1_p_x2 = lambda x: 1.0 / (1.0 + x**2)


@dataclass
class ExperimentConfig:
    model_arch: str = "mlp"  # string parameter to sweep
    n_hidden: int = 64       # int parameter to sweep
    n_data: int = 2_000
    seed: int = 0


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


def build_model(model_arch: str, n_hidden: int) -> nn.Module:
    model_map = {
        "mlp": MLP,
        "glu": GLU,
    }
    if model_arch not in model_map:
        raise ValueError(f"Unknown model_arch='{model_arch}'. Use one of {list(model_map)}")
    return model_map[model_arch](n_x=1, n_h=int(n_hidden), n_y=1)


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
    mse = get_test_loss(X_test, Y_test)
    print(OmegaConf.to_yaml(cfg))
    print(f"Built {cfg.model_arch} with n_hidden={cfg.n_hidden}; test_mse={mse:.6f}")

if __name__ == "__main__":
    main()