"""Sklearn-backed dataset loaders used by `problems.problem_factory`."""

import os

import pandas as pd
import torch
from sklearn.datasets import (
    fetch_california_housing,
    make_friedman1,
    make_friedman2,
    make_friedman3,
)
from sklearn.preprocessing import StandardScaler


def _to_standardized_tensors(
    X,
    Y,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    scaler_X = StandardScaler().fit(X)
    scaler_Y = StandardScaler().fit(Y)
    X = torch.from_numpy(scaler_X.transform(X)).to(device, dtype=dtype)
    Y = torch.from_numpy(scaler_Y.transform(Y)).to(device, dtype=dtype)
    return X, Y, X, Y


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

    return _to_standardized_tensors(X, Y, device=device, dtype=dtype)


def load_airfoil(n_data: int = -1,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Airfoil Self-Noise dataset: predict sound pressure level from wind tunnel params.
    https://archive.ics.uci.edu/dataset/291/airfoil+self+noise
    """
    CACHE_DIR = os.path.expanduser("data/cache/")
    os.makedirs(CACHE_DIR, exist_ok=True)
    CACHE_PATH = os.path.join(CACHE_DIR, "airfoil_self_noise.csv")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat"
    cols = ["Frequency", "Angle_of_attack", "Chord_length",
            "Free_stream_velocity", "Suction_side_thickness", "Sound_pressure"]
    if os.path.isfile(CACHE_PATH):
        df = pd.read_csv(CACHE_PATH)
    else:
        df = pd.read_csv(url, sep="\t", header=None, names=cols)
        df.to_csv(CACHE_PATH, index=False)

    Y = df.pop("Sound_pressure").values.reshape(-1, 1)
    X = df.values

    X, Y, _, _ = _to_standardized_tensors(X, Y, device=device, dtype=dtype)
    print(f"Airfoil — X: {X.shape}, y: {Y.shape}")
    print(f"Features: {df.columns.tolist()}")
    return X, Y, X, Y


def load_friedman1(
    n_data: int = -1,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n_samples = n_data if n_data > 0 else 2_000
    X, Y = make_friedman1(n_samples=n_samples, n_features=5, random_state=0)
    return _to_standardized_tensors(X, Y.reshape(-1, 1), device=device, dtype=dtype)


def load_friedman2(
    n_data: int = -1,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n_samples = n_data if n_data > 0 else 2_000
    X, Y = make_friedman2(n_samples=n_samples, random_state=0)
    return _to_standardized_tensors(X, Y.reshape(-1, 1), device=device, dtype=dtype)


def load_friedman3(
    n_data: int = -1,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n_samples = n_data if n_data > 0 else 2_000
    X, Y = make_friedman3(n_samples=n_samples, random_state=0)
    return _to_standardized_tensors(X, Y.reshape(-1, 1), device=device, dtype=dtype)


SKLEARN_PROBLEMS = {
    "california_housing": load_california_housing,
    "airfoil": load_airfoil,
    "friedman1": load_friedman1,
    "friedman2": load_friedman2,
    "friedman3": load_friedman3,
}
