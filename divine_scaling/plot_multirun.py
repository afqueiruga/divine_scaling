import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
import random
from statistics import mean

import matplotlib.pyplot as plt


def detect_latest_multirun_dirs(n_latest: int) -> list[Path]:
    """Detect the latest N multirun directories in the default 'multirun' folder."""
    if n_latest <= 0:
        raise ValueError("n_latest must be positive.")

    multirun_root = Path("multirun")
    if not multirun_root.is_dir():
        raise FileNotFoundError("No 'multirun' directory found.")
    all_time_dirs = [
        d
        for date_dir in multirun_root.iterdir() if date_dir.is_dir()
        for d in date_dir.iterdir() if d.is_dir()
    ]
    if not all_time_dirs:
        raise FileNotFoundError("Could not find any valid multirun directories inside 'multirun'.")
    sorted_dirs = sorted(all_time_dirs)
    return sorted_dirs[-n_latest:]


def detect_latest_multirun_dir() -> Path:
    """Detect the latest multirun directory in the default 'multirun' folder."""
    latest_dir = detect_latest_multirun_dirs(1)[0]
    print(f"Auto-detected latest multirun_dir: {latest_dir}")
    return latest_dir


def load_metrics(multirun_dir: Path) -> dict[tuple[str, str], dict[int, list[float]]]:
    """Load metrics.json files from one Hydra multirun directory."""
    grouped: dict[tuple[str, str], dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for child in sorted(multirun_dir.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        metrics_path = child / "metrics.json"
        if not metrics_path.exists():
            continue
        with metrics_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
        model_arch = str(metrics["model_arch"])
        activation = str(metrics.get("activation", "relu"))
        n_hidden = int(metrics["n_hidden"])
        test_rmse = float(metrics["test_rmse"])
        grouped[(model_arch, activation)][n_hidden].append(test_rmse)
    return grouped


def load_metrics_from_dirs(
    multirun_dirs: list[Path],
) -> dict[tuple[str, str], dict[int, list[float]]]:
    """Load and merge metrics.json data from one or more Hydra multirun directories."""
    grouped: dict[tuple[str, str], dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for multirun_dir in multirun_dirs:
        per_dir_grouped = load_metrics(multirun_dir)
        for key, hidden_to_rmse in per_dir_grouped.items():
            for n_hidden, rmse_values in hidden_to_rmse.items():
                grouped[key][n_hidden].extend(rmse_values)
    return grouped


def load_trial_functions(multirun_dir: Path) -> list[dict[str, object]]:
    """Load per-trial function samples from metrics.json files."""
    trials: list[dict[str, object]] = []
    for child in sorted(multirun_dir.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        metrics_path = child / "metrics.json"
        if not metrics_path.exists():
            continue
        with metrics_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
        eval_x = metrics.get("eval_x")
        eval_y = metrics.get("eval_y")
        if not isinstance(eval_x, list) or not isinstance(eval_y, list):
            continue
        if len(eval_x) != len(eval_y) or len(eval_x) == 0:
            continue
        trials.append(
            {
                "trial_name": child.name,
                "model_arch": str(metrics.get("model_arch", "unknown")),
                "activation": str(metrics.get("activation", "relu")),
                "n_hidden": int(metrics.get("n_hidden", -1)),
                "x": [float(v) for v in eval_x],
                "y": [float(v) for v in eval_y],
            }
        )
    return trials


def load_trial_functions_from_dirs(multirun_dirs: list[Path]) -> list[dict[str, object]]:
    """Load and merge per-trial function samples from one or more multirun directories."""
    merged_trials: list[dict[str, object]] = []
    for multirun_dir in multirun_dirs:
        merged_trials.extend(load_trial_functions(multirun_dir))
    return merged_trials


def fit_loglog_regression(x_vals: list[int], y_vals: list[float]) -> tuple[float, float, float]:
    """Fit log(y) = intercept + slope * log(x), return slope, intercept, r^2."""
    if len(x_vals) != len(y_vals):
        raise ValueError("x_vals and y_vals must have the same length.")
    if len(x_vals) < 2:
        raise ValueError("Need at least 2 points for regression.")
    if any(x <= 0 for x in x_vals) or any(y <= 0 for y in y_vals):
        raise ValueError("Log-log regression requires all x and y values to be positive.")

    log_x = [math.log(x) for x in x_vals]
    log_y = [math.log(y) for y in y_vals]

    n = len(log_x)
    mean_x = sum(log_x) / n
    mean_y = sum(log_y) / n
    ss_xx = sum((x - mean_x) ** 2 for x in log_x)
    if ss_xx == 0:
        raise ValueError("Cannot fit regression when all x values are identical.")

    ss_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(log_x, log_y))
    slope = ss_xy / ss_xx
    intercept = mean_y - slope * mean_x

    y_hat = [intercept + slope * x for x in log_x]
    ss_tot = sum((y - mean_y) ** 2 for y in log_y)
    ss_res = sum((y - y_pred) ** 2 for y, y_pred in zip(log_y, y_hat))
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0

    return slope, intercept, r_squared


def plot_metrics(
    grouped: dict[tuple[str, str], dict[int, list[float]]],
    min_n_hidden_regression: int = 0,
) -> None:
    plt.figure(figsize=(8, 5))
    ax = plt.gca()

    for idx, key in enumerate(sorted(grouped.keys())):
        model_arch, activation = key
        hidden_to_mse = grouped[key]
        x_vals = sorted(hidden_to_mse.keys())
        y_vals = [mean(hidden_to_mse[n_hidden]) for n_hidden in x_vals]
        label = f"{model_arch} ({activation})"
        (line,) = plt.loglog(x_vals, y_vals, label=label)

        reg_pairs = [(x, y) for x, y in zip(x_vals, y_vals) if x >= min_n_hidden_regression]
        reg_x_vals = [x for x, _ in reg_pairs]
        reg_y_vals = [y for _, y in reg_pairs]

        try:
            slope, intercept, r_squared = fit_loglog_regression(reg_x_vals, reg_y_vals)
            fit_y_vals = [math.exp(intercept) * (x ** slope) for x in reg_x_vals]
            plt.loglog(
                reg_x_vals,
                fit_y_vals,
                linestyle="--",
                color=line.get_color(),
                alpha=0.9,
                label=f"{label} fit",
            )
            anchor_idx = len(reg_x_vals) // 2
            anchor_x = reg_x_vals[anchor_idx]
            anchor_y = fit_y_vals[anchor_idx]
            x_offset = -18 if idx % 2 == 0 else -18
            y_offset = 18 if (idx // 2) % 2 == 0 else -22
            ax.annotate(
                f"slope={slope:.3f}",
                xy=(anchor_x, anchor_y),
                xytext=(x_offset, y_offset),
                textcoords="offset points",
                fontsize=9,
                color=line.get_color(),
                arrowprops={"arrowstyle": "->", "color": line.get_color(), "lw": 1.0},
            )
            print(
                f"log-log regression [{label}]: "
                f"log(test_rmse) = {intercept:.6f} + {slope:.6f} * log(n_hidden), "
                f"r^2={r_squared:.6f}"
            )
        except ValueError as exc:
            print(f"Skipping log-log regression for {label}: {exc}")

    plt.xlabel("n_hidden")
    plt.ylabel("test_rmse")
    plt.title("Hydra sweep: test_rmse vs n_hidden")
    plt.grid(True, alpha=0.3)
    plt.legend(title="model_arch (activation)")
    plt.tight_layout()


def plot_sample_functions(
    trials: list[dict[str, object]], max_functions: int, sample_seed: int
) -> None:
    if not trials:
        raise ValueError("No trial function samples found in metrics.json files.")

    rng = random.Random(sample_seed)
    count = min(max_functions, len(trials))
    sampled = rng.sample(trials, count) if len(trials) > count else list(trials)

    plt.figure(figsize=(8, 5))
    for trial in sampled:
        model_arch = str(trial["model_arch"])
        activation = str(trial["activation"])
        n_hidden = int(trial["n_hidden"])
        label = f"{model_arch} ({activation}), h={n_hidden}"
        plt.plot(trial["x"], trial["y"], alpha=0.85, linewidth=1.5, label=label)
    plt.xlabel("x")
    plt.ylabel("model(x)")
    plt.title(f"Sampled trial functions (n={count})")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot test_mse vs n_hidden from a Hydra multirun directory."
    )
    parser.add_argument(
        "--multirun_dir",
        type=Path,
        default=None,
        help="Path to a Hydra multirun directory, e.g. multirun/2026-02-11/01-41-59",
    )
    parser.add_argument(
        "--multirun_dirs",
        type=Path,
        nargs="+",
        default=None,
        help=(
            "One or more Hydra multirun directories to aggregate before plotting. "
            "Example: --multirun_dirs multirun/2026-02-11/01-41-59 multirun/2026-02-12/10-15-22"
        ),
    )
    parser.add_argument(
        "--latest_n",
        type=int,
        default=1,
        help=(
            "When --multirun_dir/--multirun_dirs are not set, auto-detect and join the latest N "
            "multirun directories from ./multirun (default: 1)."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path (default: <multirun_dir>/test_mse_vs_n_hidden.png)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the plot interactively in addition to saving.",
    )
    parser.add_argument(
        "--plot_sample_functions",
        action="store_true",
        help="Also plot sampled function curves from trial metrics eval_x/eval_y arrays.",
    )
    parser.add_argument(
        "--num_sample_functions",
        type=int,
        default=8,
        help="Maximum number of sampled trial functions to plot (used with --plot_sample_functions).",
    )
    parser.add_argument(
        "--sample_seed",
        type=int,
        default=0,
        help="Random seed for selecting sampled trial functions.",
    )
    parser.add_argument(
        "--min_n_hidden_regression",
        type=int,
        default=0,
        help=(
            "Minimum n_hidden value (inclusive) for points included in the log-log regression fit. "
            "Points below this threshold are still plotted but excluded from the regression."
        ),
    )
    args = parser.parse_args()

    if args.multirun_dir is not None and args.multirun_dirs is not None:
        raise ValueError("Use either --multirun_dir or --multirun_dirs, not both.")
    if args.latest_n <= 0:
        raise ValueError("--latest_n must be positive.")
    if (args.multirun_dir is not None or args.multirun_dirs is not None) and args.latest_n != 1:
        raise ValueError("--latest_n can only be used when --multirun_dir/--multirun_dirs are not set.")

    if args.multirun_dirs is not None:
        multirun_dirs = [d.resolve() for d in args.multirun_dirs]
    elif args.multirun_dir is None:
        multirun_dirs = [d.resolve() for d in detect_latest_multirun_dirs(args.latest_n)]
        if len(multirun_dirs) == 1:
            print(f"Auto-detected latest multirun_dir: {multirun_dirs[0]}")
        else:
            print(f"Auto-detected latest {len(multirun_dirs)} multirun_dirs:")
            for multirun_dir in multirun_dirs:
                print(f"  - {multirun_dir}")
    else:
        multirun_dirs = [args.multirun_dir.resolve()]

    for multirun_dir in multirun_dirs:
        if not multirun_dir.is_dir():
            raise FileNotFoundError(f"Not a directory: {multirun_dir}")

    grouped = load_metrics_from_dirs(multirun_dirs)
    plot_metrics(grouped, min_n_hidden_regression=args.min_n_hidden_regression)
    if args.output is not None:
        output_path = args.output.resolve()
    elif len(multirun_dirs) == 1:
        output_path = multirun_dirs[0] / "test_mse_vs_n_hidden.png"
    else:
        output_path = Path("joined_test_mse_vs_n_hidden.png").resolve()
    plt.savefig(output_path, dpi=150)

    print(f"Loaded {len(multirun_dirs)} multirun director{'y' if len(multirun_dirs) == 1 else 'ies'}.")
    print(f"Saved plot to {output_path}")

    if args.plot_sample_functions:
        if args.num_sample_functions <= 0:
            raise ValueError("--num_sample_functions must be positive.")
        trials = load_trial_functions_from_dirs(multirun_dirs)
        plot_sample_functions(trials, args.num_sample_functions, args.sample_seed)
        if args.output is not None:
            sample_output_path = output_path.with_name(
                f"{output_path.stem}_sampled_functions{output_path.suffix}"
            )
        elif len(multirun_dirs) == 1:
            sample_output_path = multirun_dirs[0] / "sampled_trial_functions.png"
        else:
            sample_output_path = Path("joined_sampled_trial_functions.png").resolve()
        plt.savefig(sample_output_path, dpi=150)
        print(f"Saved sampled function plot to {sample_output_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
