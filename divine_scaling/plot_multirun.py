import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt


def detect_latest_multirun_dir() -> Path:
    """Detect the latest multirun directory in the default 'multirun' folder."""
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
    latest_dir = max(all_time_dirs)
    print(f"Auto-detected latest multirun_dir: {latest_dir}")
    return latest_dir


def load_metrics(multirun_dir: Path) -> dict[str, dict[int, list[float]]]:
    """Load metrics.json files from one Hydra multirun directory."""
    grouped: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))

    for child in sorted(multirun_dir.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue

        metrics_path = child / "metrics.json"
        if not metrics_path.exists():
            continue

        with metrics_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)

        model_arch = str(metrics["model_arch"])
        n_hidden = int(metrics["n_hidden"])
        test_mse = float(metrics["test_mse"])
        grouped[model_arch][n_hidden].append(test_mse)

    return grouped


def plot_metrics(grouped: dict[str, dict[int, list[float]]]) -> None:
    plt.figure(figsize=(8, 5))

    for model_arch in sorted(grouped.keys()):
        hidden_to_mse = grouped[model_arch]
        x_vals = sorted(hidden_to_mse.keys())
        y_vals = [mean(hidden_to_mse[n_hidden]) for n_hidden in x_vals]
        plt.loglog(x_vals, y_vals, label=model_arch)

    plt.xlabel("n_hidden")
    plt.ylabel("test_mse")
    plt.title("Hydra sweep: test_mse vs n_hidden")
    plt.grid(True, alpha=0.3)
    plt.legend(title="model_arch")
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
    args = parser.parse_args()

    if args.multirun_dir is None:
        multirun_dir = detect_latest_multirun_dir()
    else:
       multirun_dir = args.multirun_dir.resolve()
    if not multirun_dir.is_dir():
        raise FileNotFoundError(f"Not a directory: {multirun_dir}")

    grouped = load_metrics(multirun_dir)
    plot_metrics(grouped)
    if args.output is not None:
        output_path = args.output.resolve()
    else:
        output_path = multirun_dir / "test_mse_vs_n_hidden.png"
    plt.savefig(output_path, dpi=150)

    print(f"Saved plot to {output_path}")
    if args.show: plt.show()


if __name__ == "__main__":
    main()
