"""
Standalone script to plot training loss curves.

Run this after training completes to regenerate plots, or to compare multiple training runs.

Usage:
  # Plot single training run
  python plot_training_loss.py

  # Plot with custom metrics file
  python plot_training_loss.py --metrics-file checkpoints/logs/training_metrics.json

  # Compare multiple runs
  python plot_training_loss.py --compare run1 checkpoints1/logs/training_metrics.json run2 checkpoints2/logs/training_metrics.json
"""

import argparse
from pathlib import Path

try:
    from . import plot_utils
except ImportError:
    import plot_utils


def main():
    parser = argparse.ArgumentParser(description="Plot training loss curves")

    parser.add_argument(
        "--metrics-file",
        type=str,
        default="checkpoints/logs/training_metrics_2.json",
        help="Path to training_metrics.json file"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints",
        help="Directory to save plots"
    )

    parser.add_argument(
        "--compare",
        nargs="+",
        help="Compare multiple runs: --compare run1_name run1_path run2_name run2_path ..."
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs (for epoch boundary markers)"
    )

    parser.add_argument(
        "--samples-per-epoch",
        type=int,
        default=None,
        help="Samples per epoch (for epoch boundary markers)"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Single run plot
    if not args.compare:
        metrics_file = args.metrics_file
        if not Path(metrics_file).exists():
            print(f"Error: Metrics file not found: {metrics_file}")
            return

        print(f"Plotting training loss from {metrics_file}")
        plot_utils.plot_training_loss(metrics_file, output_dir=str(output_dir))

        # Also plot training vs validation if data available
        print(f"Plotting training vs validation loss...")
        plot_utils.plot_train_vs_validation(metrics_file, output_dir=str(output_dir))

        # Also plot with epoch boundaries if provided
        if args.epochs and args.samples_per_epoch:
            plot_utils.plot_loss_vs_epochs(
                metrics_file,
                num_epochs=args.epochs,
                samples_per_epoch=args.samples_per_epoch,
                output_dir=str(output_dir)
            )

    # Compare multiple runs
    else:
        if len(args.compare) % 2 != 0:
            print("Error: --compare requires pairs of (run_name, metrics_file)")
            return

        runs = {}
        for i in range(0, len(args.compare), 2):
            run_name = args.compare[i]
            metrics_path = args.compare[i + 1]

            if not Path(metrics_path).exists():
                print(f"Warning: Metrics file not found: {metrics_path}")
                continue

            runs[run_name] = metrics_path

        if not runs:
            print("Error: No valid metrics files found")
            return

        print(f"Comparing {len(runs)} training runs:")
        for run_name in runs:
            print(f"  - {run_name}")

        plot_utils.compare_training_runs(runs, output_dir=str(output_dir))

    print(f"\n✓ Plots saved to {output_dir}")


if __name__ == "__main__":
    main()
