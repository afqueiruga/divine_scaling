import torch
from torch.optim.optimizer import Optimizer
from torch.func import jacrev, hessian, functional_call
from torch.optim.lbfgs import _strong_wolfe
import scipy


def get_trainable_params(model):
    return {k: v for k, v in model.named_parameters() if v.requires_grad}


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

    @torch.no_grad()
    def step(self, loss_fn, x, y):
        group = self.param_groups[0]
        lr = group["lr"]
        damping = group["damping"]
        # Compute Hessian and gradient at current position
        hessian_matrix, grad_vec = compute_hessian_and_grad(self._model, loss_fn, x, y)
        # Solve to get the ideal Newton step, which we will line search along.
        newton_dir = solve_with_preconditioning(
            hessian_matrix, grad_vec, t_reg=damping, return_info=False
        )
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
