import torch
from torch.optim.optimizer import Optimizer
from torch.func import jacrev, hessian, functional_call
from torch.optim.lbfgs import _strong_wolfe
import scipy
import torch.nn as nn
import numpy as np
import warnings


def set_grad(
    model: nn.Module, modules: list[str] | None, enable_all: bool = False
) -> bool:
    """Freeze all parameters except for the selected modules."""
    enabled = set(modules or [])
    any_enabled = False
    for name, p in model.named_parameters():
        if enable_all:
            p.requires_grad_(True)
            any_enabled = True
            continue
        top_level = name.split(".", 1)[0]
        is_enabled = top_level in enabled or name in enabled
        p.requires_grad_(is_enabled)
        if is_enabled:
            any_enabled = True
    return any_enabled


def get_trainable_params(model: nn.Module):
    return {k: v for k, v in model.named_parameters() if v.requires_grad}


def _backtracking_line_search(
    obj_func, x_init, t, d, f, g, gtd, c1=1e-4, rho=0.5, max_iter=20
):
    """Simple backtracking line search for Newton's method."""
    alpha = t
    for i in range(max_iter):
        f_new, g_new = obj_func(x_init, alpha, d)   # Evaluate new objective and gradient
        if f_new <= f + c1 * alpha * gtd:           # Check Armijo condition
            return f_new, g_new, alpha, i + 1
        alpha *= rho                                # Reduce step size
    f_final, g_final = obj_func(x_init, alpha, d)   # If no good step found, return very small step
    return f_final, g_final, alpha, max_iter


def compute_hessian_and_grad(model, loss_fn, x, y):
    """
    Compute Hessian matrix and gradient vector for a model's loss
    """
    # Extract parameters and buffers
    params = get_trainable_params(model)
    param0 = next(iter(params.values()))
    dtype, device = param0.dtype, param0.device
    buffers = {k: v for k, v in model.named_buffers()}

    # Define loss as function of parameters
    def loss(params):
        y_pred = functional_call(model, (params, buffers), x)
        return loss_fn(y_pred, y)

    # Compute gradient and Hessian
    grad_dict = jacrev(loss)(params)
    hess_dict = hessian(loss)(params)
    param_sizes = [params[k].numel() for k in params]
    n = sum(param_sizes)

    grad_vec = torch.cat([grad_dict[k].flatten() for k in params])
    # Build Hessian matrix, flattening the Hessian into a single vector.
    H = torch.zeros(n, n, dtype=dtype, device=device)
    i = 0
    for k1, s1 in zip(params, param_sizes):
        j = 0
        for k2, s2 in zip(params, param_sizes):
            H[i : i + s1, j : j + s2] = hess_dict[k1][k2].reshape(s1, s2)
            j += s2
        i += s1
    return H, grad_vec


@torch.no_grad()
def solve_with_preconditioning(H, g, diag_clamp=1e-12, t_reg=0.0, return_info=True):
    n = H.shape[0]
    device, dtype = H.device, H.dtype
    # Step 1: Identify and remove zero rows/columns
    rows_all_zero = (H == 0).all(dim=1)
    non_zero_mask = ~rows_all_zero
    n_reduced = non_zero_mask.sum().item()
    # Extract the reduced system
    idx = torch.where(non_zero_mask)[0]
    H_reduced = H[idx][:, idx]
    g_reduced = g[idx]
    # Step 2: Symmetric preconditioning:
    # P @ H @ P where P = diag(1/sqrt(|d_ii|))
    diag_H = H_reduced.diag()
    diag_abs = diag_H.abs().clamp(min=diag_clamp)
    # Apply Tikhonov regularization to the reduced Hessian
    H_reduced = H_reduced + t_reg * torch.eye(n_reduced, dtype=dtype, device=device) * diag_abs.mean()
    P = 1.0 / torch.sqrt(diag_abs)
    # Apply preconditioning (element-wise for efficiency)
    H_precond = H_reduced * P.unsqueeze(1) * P.unsqueeze(0)
    g_precond = P * g_reduced
    # Step 3: Solve with scipy
    H_np = H_precond.detach().cpu().numpy()
    g_np = g_precond.detach().cpu().numpy()
    y_precond = scipy.linalg.solve(H_np, -g_np, assume_a='sym')
    # Step 4: Recover solution and reverse preconditioning
    # Undo preconditioning: x = P @ y
    x_reduced = P * torch.from_numpy(y_precond).to(device=device, dtype=dtype)
    # Project back to original size (zero rows get zero solution)
    x_full = torch.zeros(n, dtype=dtype, device=device)
    x_full[idx] = x_reduced
    return x_full


@torch.no_grad()
def solve_with_preconditioning_robust(
    H,
    g,
    diag_clamp=1e-12,
    t_reg=0.0,
    return_info=False,
    use_double_precision=True,
    enforce_symmetry=True,
    drop_near_zero_rows=True,
    adaptive_damping=True,
    enable_fallback=True,
):
    """
    Robustly solve H x = -g using symmetric diagonal preconditioning.

    The routine:
    1) enforces symmetry,
    2) drops numerically-null rows/cols using a tolerance,
    3) adaptively increases Tikhonov damping if the solve is unstable,
    4) falls back to least-squares / pseudo-inverse when needed.
    """
    n = H.shape[0]
    device, out_dtype = H.device, H.dtype
    solve_dtype = torch.float64 if use_double_precision else H.dtype
    H_solve = H.to(dtype=solve_dtype)
    g_solve = g.to(dtype=solve_dtype)

    # Numerical Hessians can be slightly asymmetric; optionally symmetrize before solving.
    H_sym = 0.5 * (H_solve + H_solve.T) if enforce_symmetry else H_solve
    asymmetry_num = torch.linalg.norm(H_solve - H_solve.T)
    asymmetry_den = torch.linalg.norm(H_solve).clamp_min(torch.finfo(solve_dtype).eps)
    symmetry_residual = (asymmetry_num / asymmetry_den).item()

    max_abs = H_sym.abs().max().item()
    tiny = torch.finfo(solve_dtype).eps
    zero_tol = max(diag_clamp, tiny) * max(1.0, max_abs)
    if drop_near_zero_rows:
        row_inf = H_sym.abs().amax(dim=1)
        col_inf = H_sym.abs().amax(dim=0)
        g_abs = g_solve.abs()
        drop_mask = (row_inf <= zero_tol) & (col_inf <= zero_tol) & (g_abs <= zero_tol)
    else:
        drop_mask = torch.zeros(n, dtype=torch.bool, device=H_sym.device)
    keep_mask = ~drop_mask
    idx = torch.where(keep_mask)[0]
    n_reduced = idx.numel()

    info = {
        "n": n,
        "n_reduced": int(n_reduced),
        "dropped_rows": int(drop_mask.sum().item()),
        "symmetry_residual": symmetry_residual,
        "stabilization": {
            "use_double_precision": bool(use_double_precision),
            "enforce_symmetry": bool(enforce_symmetry),
            "drop_near_zero_rows": bool(drop_near_zero_rows),
            "adaptive_damping": bool(adaptive_damping),
            "enable_fallback": bool(enable_fallback),
        },
        "solver": None,
        "attempts": 0,
        "used_fallback": False,
        "lambda_used": 0.0,
        "condition_estimate": float("inf"),
        "warning_messages": [],
    }

    x_full = torch.zeros(n, dtype=out_dtype, device=device)
    if n_reduced == 0:
        info["solver"] = "empty_reduced_system"
        return (x_full, info) if return_info else x_full

    H_reduced = H_sym[idx][:, idx]
    g_reduced = g_solve[idx]
    diag_abs = H_reduced.diag().abs().clamp(min=diag_clamp)
    base_scale = torch.maximum(diag_abs.mean(), H_reduced.abs().mean()).clamp_min(diag_clamp)
    base_lambda = float(t_reg) * float(base_scale.item())

    max_attempts = 6 if adaptive_damping else 1
    damping_growth = 10.0
    cond_threshold = 1e12

    y_precond = None
    P_used = None
    lambda_used = None
    cond_est = float("inf")
    warning_messages = []

    eye = torch.eye(n_reduced, dtype=solve_dtype, device=H_reduced.device)
    for k in range(max_attempts):
        lam = base_lambda * (damping_growth ** k)
        H_reg = H_reduced + lam * eye
        diag_abs_reg = H_reg.diag().abs().clamp(min=diag_clamp)
        P = 1.0 / torch.sqrt(diag_abs_reg)
        H_precond = H_reg * P.unsqueeze(1) * P.unsqueeze(0)
        g_precond = P * g_reduced
        H_np = H_precond.detach().cpu().numpy()
        g_np = g_precond.detach().cpu().numpy()

        try:
            cond_est = float(np.linalg.cond(H_np))
        except Exception:
            cond_est = float("inf")

        local_messages = []
        try:
            with warnings.catch_warnings(record=True) as ws:
                warnings.simplefilter("always")
                y_try = scipy.linalg.solve(
                    H_np, -g_np, assume_a="sym", check_finite=False
                )
            local_messages = [str(w.message) for w in ws]
            ill_warn = any("ill-conditioned" in msg.lower() for msg in local_messages)
            if np.all(np.isfinite(y_try)) and not ill_warn and cond_est <= cond_threshold:
                y_precond = y_try
                P_used = P
                lambda_used = lam
                warning_messages = local_messages
                info["solver"] = "solve"
                info["attempts"] = k + 1
                break
        except Exception as e:
            local_messages.append(str(e))

        warning_messages = local_messages
        info["attempts"] = k + 1

    if y_precond is None and not enable_fallback:
        raise scipy.linalg.LinAlgError(
            f"Robust solve failed after {max_attempts} attempts; "
            "fallback disabled."
        )

    if y_precond is None:
        info["used_fallback"] = True
        H_np = H_precond.detach().cpu().numpy()
        g_np = g_precond.detach().cpu().numpy()
        try:
            y_precond = scipy.linalg.lstsq(H_np, -g_np, check_finite=False)[0]
            info["solver"] = "lstsq"
        except Exception:
            y_precond = scipy.linalg.pinv(H_np, check_finite=False) @ (-g_np)
            info["solver"] = "pinv"
        P_used = P
        lambda_used = lam

    x_reduced = P_used * torch.from_numpy(y_precond).to(device=device, dtype=solve_dtype)
    x_full[idx] = x_reduced.to(dtype=out_dtype)

    info["attempts"] = max(info["attempts"], 1 if lambda_used is not None else 0)
    info["lambda_used"] = float(0.0 if lambda_used is None else lambda_used)
    info["condition_estimate"] = cond_est
    info["warning_messages"] = warning_messages
    return (x_full, info) if return_info else x_full


class Newton(Optimizer):
    """Newton's method optimizer with Hessian computation."""

    def __init__(self, model, lr=1.0, max_iter=20, damping=1e-6, line_search_fn=None):
        defaults = dict(
            lr=lr, max_iter=max_iter, damping=damping, line_search_fn=line_search_fn
        )
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        super().__init__(trainable_params, defaults)
        self._model = model
        self._params = self.param_groups[0]["params"]
        self._n_iter = 0

    def _gather_flat_grad(self):
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.new(p.numel()).zero_()
            else:
                view = p.grad.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _add_grad(self, step_size, update):
        offset = 0
        for p in self._params:
            numel = p.numel()
            p.add_(update[offset : offset + numel].view_as(p), alpha=step_size)
            offset += numel

    def _clone_param(self):
        return [p.clone(memory_format=torch.contiguous_format) for p in self._params]

    def _set_param(self, params_data):
        for p, pdata in zip(self._params, params_data):
            p.copy_(pdata)

    def _directional_evaluate(self, loss_fn, x, y, x_params, t, d):
        """Evaluate loss and gradient in a given direction for line search."""
        # Move in direction d with step size t
        self._add_grad(t, d)
        # Enable gradients temporarily for loss and gradient computation
        with torch.enable_grad():
            # Forward pass and loss computation
            y_pred = self._model(x)
            loss = loss_fn(y_pred, y)
            loss_val = float(loss.detach())
            # Zero gradients and compute new gradients
            self._model.zero_grad()
            loss.backward()
            flat_grad = self._gather_flat_grad()
        # Restore original parameters
        self._set_param(x_params)
        return loss_val, flat_grad

    @torch.no_grad()
    def step(self, loss_fn, x, y):
        group = self.param_groups[0]
        lr = group["lr"]
        damping = group["damping"]
        line_search_fn = group["line_search_fn"]
        # Compute Hessian and gradient at current position
        hessian_matrix, grad_vec = compute_hessian_and_grad(self._model, loss_fn, x, y)
        # Solve to get the ideal Newton step, which we will line search along.
        newton_dir = solve_with_preconditioning_robust(
            hessian_matrix, grad_vec, t_reg=damping, return_info=False
        )
        # Compute initial loss and directional derivative
        orig_loss = loss_fn(self._model(x), y)
        loss_val = float(orig_loss)
        gtd = grad_vec.dot(newton_dir)  # directional derivative

        # Check if Newton direction is a descent direction
        if gtd >= 0:
            # Newton direction is not a descent direction (Hessian not positive definite)
            # Fall back to steepest descent direction (unnormalized)
            newton_dir = -grad_vec
            gtd = -grad_vec.dot(grad_vec)  # = -||g||^2, guaranteed negative

        # Apply line search or fixed step
        if line_search_fn is not None:
            # Store initial parameters
            x_init = self._clone_param()
            def obj_func(x_params, t, d):
                return self._directional_evaluate(loss_fn, x, y, x_params, t, d)
            step_size = lr
            # Choose line search method
            if line_search_fn == "strong_wolfe":
                _line_search = _strong_wolfe
            elif line_search_fn == "backtracking":
                _line_search = _backtracking_line_search
            else:
                raise RuntimeError(f"line search method '{line_search_fn}' not supported")
            loss_val, grad_vec_ls, step_size, ls_func_evals = _line_search(
                obj_func, x_init, step_size, newton_dir, loss_val, grad_vec, gtd)
            # Apply the optimal step found by line search
            self._add_grad(step_size, newton_dir)
        else:
            # Just vanilla Newton step with the fixed step size.
            self._add_grad(lr, newton_dir)

        # Return final loss and gradient magnitude at new position
        # Note: We always recompute the gradient at the final position
        with torch.enable_grad():
            y_pred = self._model(x)
            final_loss = loss_fn(y_pred, y)
            self._model.zero_grad()
            final_loss.backward()
            final_grad_vec = self._gather_flat_grad()

        return float(final_loss), final_grad_vec.abs().max().item()
