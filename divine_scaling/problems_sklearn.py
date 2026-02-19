"""Sklearn-backed dataset loaders used by `problems.problem_factory`."""

import torch
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

def load_california_housing(
    n_data: int = -1,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load California Housing into standardized tensors (no train/test split)."""
    data = fetch_california_housing()
    X = data.data
    Y = data.target.reshape(-1, 1)
    if n_data > 0:
        X = X[:n_data, ...]
        Y = Y[:n_data, ...]

    scaler_X = StandardScaler().fit(X)
    scaler_Y = StandardScaler().fit(Y)
    X = torch.from_numpy(scaler_X.transform(X)).to(device, dtype=dtype)
    Y = torch.from_numpy(scaler_Y.transform(Y)).to(device, dtype=dtype)

    return X, Y, X, Y


SKLEARN_PROBLEMS = {
    "california_housing": load_california_housing,
}
