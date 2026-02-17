import numpy as np
import torch


f_x2 = lambda x: x**2
f_x3 = lambda x: x**3
f_cos = lambda x: np.cos(4 * x)
f_sin = lambda x: np.sin(4 * x)
f_sin_x2 = lambda x: np.sin(4 * (x - 1)**2)
f_one_over_1_p_x2 = lambda x: 1.0 / (1.0 + x**2)
f_one_over_1_p_9x2 = lambda x: 1.0 / (1.0 + 9.0*x**2)
f_atan = lambda x: np.arctan(x)


def make_1d_problem(f, N_data=2_000, device="cpu", dtype=torch.float64):
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


# Public mapping from string -> numpy scalar function.
# Uses the variable names, stripping the `f_` prefix (e.g. `f_x2` -> "x2").
PROBLEM_1D = {name[2:]: fn for name, fn in globals().items() if name.startswith("f_") and callable(fn)}
