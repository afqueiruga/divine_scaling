import unittest
import torch
import torch.nn as nn

from divine_scaling import newton_optimizer


class NewtonOptimizerTest(unittest.TestCase):
    class _BlockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.D = nn.Linear(3, 3)
            self.U = nn.Linear(3, 3)
            self.Q = nn.Linear(3, 3)
            self.G = nn.Linear(3, 3)
            self.extra = nn.Linear(3, 3)

    def _run_linear_regression_step(self, line_search_fn):
        torch.manual_seed(0)
        model = nn.Linear(1, 1, bias=False)
        X = torch.tensor([[1.0], [2.0], [3.0]])
        y = 3.0 * X
        loss_fn = nn.MSELoss()
        optimizer = newton_optimizer.Newton(model, line_search_fn=line_search_fn)

        with torch.no_grad():
            initial_loss = loss_fn(model(X), y).item()

        final_loss, final_grad_max = optimizer.step(loss_fn, X, y)

        self.assertLess(final_loss, initial_loss)
        self.assertLess(final_grad_max, 1e-4)
        self.assertTrue(torch.allclose(model.weight, torch.tensor([[3.0]]), atol=1e-5))

    def test_solve_linear_system(self):
        H = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        g = torch.tensor([1.0, 1.0])
        x = newton_optimizer.solve_with_preconditioning(H, g)
        # solve_with_preconditioning solves H x = -g.
        self.assertTrue(torch.allclose(x, torch.tensor([-1.0, -1.0])))

    def test_optimize_linear_model_line_search_off(self):
        self._run_linear_regression_step(line_search_fn=None)

    def test_optimize_linear_model_line_search_backtracking(self):
        self._run_linear_regression_step(line_search_fn="backtracking")

    def test_optimize_linear_model_line_search_strong_wolfe(self):
        self._run_linear_regression_step(line_search_fn="strong_wolfe")

    def test_set_grad_enables_selected_blocks_and_disables_others(self):
        model = self._BlockModel()
        any_enabled = newton_optimizer.set_grad(model, ["U", "extra"])
        self.assertTrue(any_enabled)

        for p in model.U.parameters():
            self.assertTrue(p.requires_grad)
        for p in model.extra.parameters():
            self.assertTrue(p.requires_grad)

        for p in model.D.parameters():
            self.assertFalse(p.requires_grad)
        for p in model.Q.parameters():
            self.assertFalse(p.requires_grad)
        for p in model.G.parameters():
            self.assertFalse(p.requires_grad)

    def test_set_grad_freezes_all_when_no_blocks_selected(self):
        model = self._BlockModel()
        any_enabled = newton_optimizer.set_grad(model, [])
        self.assertFalse(any_enabled)
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))

    def test_set_grad_enable_all_option_enables_all_parameters(self):
        model = self._BlockModel()
        any_enabled = newton_optimizer.set_grad(model, [], enable_all=True)
        self.assertTrue(any_enabled)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))

    def test_set_grad_returns_false_when_names_do_not_match_any_parameter(self):
        model = self._BlockModel()
        any_enabled = newton_optimizer.set_grad(model, ["does_not_exist"])
        self.assertFalse(any_enabled)
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
