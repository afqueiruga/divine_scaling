import argparse
from datetime import datetime
import json
import math
from collections import defaultdict
from pathlib import Path
import random
import re
from statistics import mean

import matplotlib.pyplot as plt
from adjustText import adjust_text

# Pretty labels for known problem keys from `problems.py` and `problems_sklearn.py`.
PROBLEM_DISPLAY_NAMES: dict[str, str] = {
    # problems.py (1D)
    "x2": "x^2",
    "x3": "x^3",
    "cos": "cos(4x)",
    "sin": "sin(4x)",
    "sin_x2": "sin(4(x-1)^2)",
    "one_over_1_p_x2": "1 / (1 + x^2)",
    "one_over_1_p_9x2": "1 / (1 + 9x^2)",
    "atan": "atan(x)",
    # problems.py (2D)
    "x2_y2": "x^2 + y^2",
    "x3_y3": "x^3 + y^3",
    "cos_x_cos_y": "cos(4x) cos(4y)",
    "sin_x_sin_y": "sin(4x) sin(4y)",
    "sin_x_sin_y_x2_y2": "sin(4x) sin(4y) (x^2 + y^2)",
    "one_over_1_p_x2_y2": "1 / (1 + x^2 + y^2)",
    "one_over_1_p_9x2_9y2": "1 / (1 + 9x^2 + 9y^2)",
    "atan_x_atan_y": "atan(x) + atan(y)",
    # problems_sklearn.py (raw names)
    "california_housing": "California Housing",
    "airfoil": "Airfoil Self-Noise",
    "friedman1": "Friedman #1",
    "friedman2": "Friedman #2",
    "friedman3": "Friedman #3",
    # problems.py real_* external names used by problem_factory
    "real_california_housing": "California Housing",
    "real_airfoil": "Airfoil Self-Noise",
    "real_friedman1": "Friedman #1",
    "real_friedman2": "Friedman #2",
    "real_friedman3": "Friedman #3",
}

# Optional per-problem/per-architecture bounds for log-log regression x-range.
# If a pair is absent, regression falls back to --min_n_hidden_regression and no xmax cap.
REGRESSION_X_RANGES: dict[str, dict[str, tuple[int, int | None]]] = {
    "real_airfoil": {
        "mlp": (10, 500),
        "glu": (10, 100),
    },
}


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


def _get_problem_key(metrics: dict[str, object]) -> str:
    """Return a normalized problem key from metrics."""
    return str(metrics.get("problem", "unknown"))


def _sanitize_for_filename(name: str) -> str:
    """Sanitize arbitrary string to a filesystem-friendly slug."""
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    return sanitized.strip("_") or "unknown"


def _apply_paper_style(ax: plt.Axes) -> None:
    """Apply a paper-friendly Times-like style to an existing axes."""
    times_family = "Times New Roman"
    ax.title.set_fontfamily(times_family)
    ax.xaxis.label.set_fontfamily(times_family)
    ax.yaxis.label.set_fontfamily(times_family)
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontfamily(times_family)
    legend = ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontfamily(times_family)
        legend_title = legend.get_title()
        if legend_title is not None:
            legend_title.set_fontfamily(times_family)
    for text in ax.texts:
        text.set_fontfamily(times_family)


def _adjust_slope_annotations(ax: plt.Axes) -> None:
    """Auto-place slope annotation texts while keeping arrows in-bounds."""
    slope_texts = [text for text in ax.texts if text.get_text().startswith("slope=")]
    if slope_texts:
        # Repel labels from both other labels and plotted curves.
        adjust_text(
            slope_texts,
            ax=ax,
            objects=list(ax.lines),
            ensure_inside_axes=True,
            expand=(1.15, 1.25),
            force_text=(0.2, 0.3),
            force_static=(0.2, 0.3),
        )
        # Keep labels close to anchors to reduce edge escapes on small figures.
        max_abs_offset_points = 26.0
        for text in slope_texts:
            x_off, y_off = text.get_position()
            text.set_position(
                (
                    max(-max_abs_offset_points, min(max_abs_offset_points, float(x_off))),
                    max(-max_abs_offset_points, min(max_abs_offset_points, float(y_off))),
                )
            )


def _inward_offset(ax: plt.Axes, x: float, y: float, magnitude: float = 12.0) -> tuple[float, float]:
    """Pick an initial text offset that points toward the center of the axes."""
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    if x_min <= 0 or y_min <= 0:
        return (magnitude, magnitude)
    x_frac = (math.log(x) - math.log(x_min)) / (math.log(x_max) - math.log(x_min))
    y_frac = (math.log(y) - math.log(y_min)) / (math.log(y_max) - math.log(y_min))
    x_off = -magnitude if x_frac > 0.5 else magnitude
    y_off = -magnitude if y_frac > 0.5 else magnitude
    return (x_off, y_off)


def _apply_paper_layout(fig: plt.Figure) -> None:
    """Use fixed margins so the 3x3 PDF keeps a wide plotting region."""
    fig.subplots_adjust(left=0.19, right=0.98, bottom=0.17, top=0.90)


def load_metrics(multirun_dir: Path) -> dict[str, dict[tuple[str, str], dict[int, list[float]]]]:
    """Load metrics.json files from one Hydra multirun directory."""
    grouped: dict[str, dict[tuple[str, str], dict[int, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
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
        problem = _get_problem_key(metrics)
        n_hidden = int(metrics["n_hidden"])
        test_rmse = float(metrics["test_rmse"])
        grouped[problem][(model_arch, activation)][n_hidden].append(test_rmse)
    return grouped


def load_metrics_from_dirs(
    multirun_dirs: list[Path],
) -> dict[str, dict[tuple[str, str], dict[int, list[float]]]]:
    """Load and merge metrics.json data from one or more Hydra multirun directories."""
    grouped: dict[str, dict[tuple[str, str], dict[int, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for multirun_dir in multirun_dirs:
        per_dir_grouped = load_metrics(multirun_dir)
        for problem, per_problem_grouped in per_dir_grouped.items():
            for key, hidden_to_rmse in per_problem_grouped.items():
                for n_hidden, rmse_values in hidden_to_rmse.items():
                    grouped[problem][key][n_hidden].extend(rmse_values)
    return grouped


def load_trial_functions(multirun_dir: Path) -> dict[str, list[dict[str, object]]]:
    """Load per-trial function samples from metrics.json files."""
    trials: dict[str, list[dict[str, object]]] = defaultdict(list)
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
        problem = _get_problem_key(metrics)
        trials[problem].append(
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


def load_trial_functions_from_dirs(multirun_dirs: list[Path]) -> dict[str, list[dict[str, object]]]:
    """Load and merge per-trial function samples from one or more multirun directories."""
    merged_trials: dict[str, list[dict[str, object]]] = defaultdict(list)
    for multirun_dir in multirun_dirs:
        per_dir_trials = load_trial_functions(multirun_dir)
        for problem, problem_trials in per_dir_trials.items():
            merged_trials[problem].extend(problem_trials)
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
    problem: str,
    min_n_hidden_regression: int = 0,
) -> None:
    plt.figure(figsize=(8, 5))
    ax = plt.gca()
    slope_texts: list[plt.Annotation] = []

    for key in sorted(grouped.keys()):
        model_arch, activation = key
        hidden_to_mse = grouped[key]
        x_vals = sorted(hidden_to_mse.keys())
        y_vals = [mean(hidden_to_mse[n_hidden]) for n_hidden in x_vals]
        label = f"{model_arch} ({activation})"
        (line,) = plt.loglog(x_vals, y_vals, label=label)

        arch_bounds = REGRESSION_X_RANGES.get(problem, {}).get(model_arch)
        if arch_bounds is None:
            xmin = min_n_hidden_regression
            xmax = None
        else:
            xmin, xmax = arch_bounds

        reg_pairs = [
            (x, y)
            for x, y in zip(x_vals, y_vals)
            if x >= xmin and (xmax is None or x <= xmax)
        ]
        reg_x_vals = [x for x, _ in reg_pairs]
        reg_y_vals = [y for _, y in reg_pairs]

        try:
            slope, intercept, r_squared = fit_loglog_regression(reg_x_vals, reg_y_vals)
            fit_y_vals = [math.exp(intercept) * (x ** slope) for x in x_vals]
            plt.loglog(
                x_vals,
                fit_y_vals,
                linestyle="--",
                color=line.get_color(),
                alpha=0.9,
                label="_nolegend_",
            )
            anchor_idx = len(reg_x_vals) // 2
            anchor_x = reg_x_vals[anchor_idx]
            anchor_y = math.exp(intercept) * (anchor_x ** slope)
            x_offset, y_offset = _inward_offset(ax, float(anchor_x), float(anchor_y))
            slope_texts.append(
                ax.annotate(
                    f"slope={slope:.2f}",
                    xy=(anchor_x, anchor_y),
                    xytext=(x_offset, y_offset),
                    textcoords="offset points",
                    fontsize=10,
                    color="black",
                    arrowprops={"arrowstyle": "->", "color": "black", "lw": 1.0},
                )
            )
            print(
                f"log-log regression [{label}]: "
                f"log(test_rmse) = {intercept:.6f} + {slope:.6f} * log(n_hidden), "
                f"r^2={r_squared:.6f}, "
                f"x-range=[{xmin}, {xmax if xmax is not None else 'inf'}]"
            )
        except ValueError as exc:
            print(f"Skipping log-log regression for {label}: {exc}")

    plt.xlabel("n_hidden")
    plt.ylabel("RMSE")
    plt.title(PROBLEM_DISPLAY_NAMES.get(problem, problem), pad=4)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    _adjust_slope_annotations(ax)


def plot_sample_functions(
    trials: list[dict[str, object]], problem: str, max_functions: int, sample_seed: int
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
    plt.title(f"Sampled trial functions ({problem}, n={count})")
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
        help=(
            "Output image filename/path. When omitted, files are written to "
            "plot_outputs/ with a run timestamp in the filename."
        ),
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

    grouped_by_problem = load_metrics_from_dirs(multirun_dirs)
    problems = sorted(grouped_by_problem.keys())
    if not problems:
        raise ValueError("No metrics.json files with required fields were found.")

    print(f"Loaded {len(multirun_dirs)} multirun director{'y' if len(multirun_dirs) == 1 else 'ies'}.")
    print(f"Detected {len(problems)} problem key{'s' if len(problems) != 1 else ''}: {', '.join(problems)}")

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("plot_outputs").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.output is not None:
        output_name = args.output.name
        output_base = (output_dir / output_name).resolve()
        if output_base.suffix == "":
            output_base = output_base.with_suffix(".png")
        output_base = output_base.with_name(
            f"{output_base.stem}_{run_timestamp}{output_base.suffix}"
        )
    elif len(multirun_dirs) == 1:
        output_base = (output_dir / f"test_mse_vs_n_hidden_{run_timestamp}.png").resolve()
    else:
        output_base = (output_dir / f"joined_test_mse_vs_n_hidden_{run_timestamp}.png").resolve()

    output_paths: list[Path] = []
    multiple_problems = len(problems) > 1
    for problem in problems:
        plot_metrics(
            grouped_by_problem[problem],
            problem=problem,
            min_n_hidden_regression=args.min_n_hidden_regression,
        )
        if multiple_problems:
            output_path = output_base.with_name(
                f"{output_base.stem}_{_sanitize_for_filename(problem)}{output_base.suffix}"
            )
        else:
            output_path = output_base
        plt.savefig(output_path, dpi=150)
        ax = plt.gca()
        fig = plt.gcf()
        _apply_paper_style(ax)
        fig.set_size_inches(3, 3)
        _apply_paper_layout(fig)
        _adjust_slope_annotations(ax)
        paper_output_path = output_path.with_suffix(".pdf")
        fig.savefig(paper_output_path, format="pdf")
        output_paths.append(output_path)
        print(f"Saved plot for problem '{problem}' to {output_path}")
        print(f"Saved paper plot for problem '{problem}' to {paper_output_path}")

    if args.plot_sample_functions:
        if args.num_sample_functions <= 0:
            raise ValueError("--num_sample_functions must be positive.")
        trials_by_problem = load_trial_functions_from_dirs(multirun_dirs)
        if args.output is not None:
            sample_output_base = output_base.with_name(
                f"{output_base.stem}_sampled_functions{output_base.suffix}"
            )
        elif len(multirun_dirs) == 1:
            sample_output_base = (
                output_dir / f"sampled_trial_functions_{run_timestamp}.png"
            ).resolve()
        else:
            sample_output_base = (
                output_dir / f"joined_sampled_trial_functions_{run_timestamp}.png"
            ).resolve()

        for problem in problems:
            problem_trials = trials_by_problem.get(problem, [])
            if not problem_trials:
                print(
                    f"No trial function samples for problem '{problem}'; "
                    "skipping sampled function plot."
                )
                continue
            plot_sample_functions(problem_trials, problem, args.num_sample_functions, args.sample_seed)
            if multiple_problems:
                sample_output_path = sample_output_base.with_name(
                    f"{sample_output_base.stem}_{_sanitize_for_filename(problem)}"
                    f"{sample_output_base.suffix}"
                )
            else:
                sample_output_path = sample_output_base
            plt.savefig(sample_output_path, dpi=150)
            print(f"Saved sampled function plot for problem '{problem}' to {sample_output_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
