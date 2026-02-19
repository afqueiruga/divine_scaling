from typing import Callable

import numpy as np
import torch

from .problems_activations import load_activations_plain


# 1D problems
f_x2 = lambda x: x**2
f_x3 = lambda x: x**3
f_cos = lambda x: np.cos(4 * x)
f_sin = lambda x: np.sin(4 * x)
f_sin_x2 = lambda x: np.sin(4 * (x - 1) ** 2)
f_one_over_1_p_x2 = lambda x: 1.0 / (1.0 + x**2)
f_one_over_1_p_9x2 = lambda x: 1.0 / (1.0 + 9.0 * x**2)
f_atan = lambda x: np.arctan(x)


# 2D problems
f_x2_y2 = lambda x, y: x**2 + y**2
f_x3_y3 = lambda x, y: x**3 + y**3
f_cos_x_cos_y = lambda x, y: np.cos(4 * x) * np.cos(4 * y)
f_sin_x_sin_y = lambda x, y: np.sin(4 * x) * np.sin(4 * y)
f_sin_x_sin_y_x2_y2 = lambda x, y: np.sin(4 * x) * np.sin(4 * y) * (x**2 + y**2)
f_one_over_1_p_x2_y2 = lambda x, y: 1.0 / (1.0 + x**2 + y**2)
f_one_over_1_p_9x2_9y2 = lambda x, y: 1.0 / (1.0 + 9.0 * x**2 + 9.0 * y**2)
f_atan_x_atan_y = lambda x, y: np.arctan(x) + np.arctan(y)


def make_1d_problem(
    f: Callable[[float], float],
    N_data: int = 2_000,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate 1D train/test data from a scalar function f."""
    X_train = 2 * np.random.rand(N_data, 1) - 1
    X_test = np.linspace(-1, 1, N_data).reshape(-1, 1)
    Y_train, Y_test = f(X_train), f(X_test)
    return (
        torch.from_numpy(X_train).to(device, dtype=dtype),
        torch.from_numpy(Y_train).to(device, dtype=dtype),
        torch.from_numpy(X_test).to(device, dtype=dtype),
        torch.from_numpy(Y_test).to(device, dtype=dtype),
    )

def make_2d_problem(
    f: Callable[[float, float], float],
    N_data: int = 2_000,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    X_train = 2 * np.random.rand(N_data, 2) - 1
    # Create a square grid of test points in [-1, 1] x [-1, 1]
    side = int(np.sqrt(N_data))
    x = np.linspace(-1, 1, side)
    y = np.linspace(-1, 1, side)
    xx, yy = np.meshgrid(x, y)
    X_test = np.stack([xx.ravel(), yy.ravel()], axis=1)
    Y_train, Y_test = f(X_train[:, 0], X_train[:, 1]), f(X_test[:, 0], X_test[:, 1])
    return (
        torch.from_numpy(X_train.reshape(-1, 2)).to(device, dtype=dtype),
        torch.from_numpy(Y_train.reshape(-1, 1)).to(device, dtype=dtype),
        torch.from_numpy(X_test.reshape(-1, 2)).to(device, dtype=dtype),
        torch.from_numpy(Y_test.reshape(-1, 1)).to(device, dtype=dtype),
    )


# Public mapping from string -> numpy scalar function.
# Uses the variable names, stripping the `f_` prefix (e.g. `f_x2` -> "x2").
PROBLEM_1D = {
    "x2": f_x2,
    "x3": f_x3,
    "cos": f_cos,
    "sin": f_sin,
    "sin_x2": f_sin_x2,
    "one_over_1_p_x2": f_one_over_1_p_x2,
    "one_over_1_p_9x2": f_one_over_1_p_9x2,
    "atan": f_atan,
}


PROBLEM_2D = {
    "x2_y2": f_x2_y2,
    "x3_y3": f_x3_y3,
    "cos_x_cos_y": f_cos_x_cos_y,
    "sin_x_sin_y": f_sin_x_sin_y,
    "sin_x_sin_y_x2_y2": f_sin_x_sin_y_x2_y2,
    "one_over_1_p_x2_y2": f_one_over_1_p_x2_y2,
    "one_over_1_p_9x2_9y2": f_one_over_1_p_9x2_9y2,
    "atan_x_atan_y": f_atan_x_atan_y,
}


def problem_factory(
    problem: str, n_data: int, device: str = "cpu", dtype: torch.dtype = torch.float64
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if problem.startswith("activations_"):
        layer = int(problem.split("_")[1])
        return load_activations_plain(path="data/activations.h5", layer=layer, dtype=dtype, n_data=n_data)
    elif problem.startswith("real_"):
        dataset_name = problem[len("real_") :]
        # Lazy import so sklearn is only required for sklearn-backed problems.
        from .problems_sklearn import SKLEARN_PROBLEMS
        loader = SKLEARN_PROBLEMS[dataset_name]
        return loader(n_data=n_data, device=device, dtype=dtype)
    elif problem in PROBLEM_1D:
        f = PROBLEM_1D[problem]
        return make_1d_problem(f, n_data, device=device, dtype=dtype)
    elif problem in PROBLEM_2D:
        f = PROBLEM_2D[problem]
        return make_2d_problem(f, n_data, device=device, dtype=dtype)
    else:
        raise ValueError(f"Unknown problem '{problem}'.")
