"""
Plotting utilities for training visualization.

Provides functions to plot loss curves and other training metrics.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path


def load_metrics(metrics_file):
    """
    Load training metrics from JSON file.

    Args:
        metrics_file (str): Path to training_metrics.json

    Returns:
        dict: Metrics dictionary with keys: step, loss, learning_rate, reward
    """
    with open(metrics_file, "r") as f:
        metrics = json.load(f)
    return metrics


def plot_training_loss(metrics_file, output_dir="."):
    """
    Plot training loss and related metrics.

    Args:
        metrics_file (str): Path to training_metrics.json
        output_dir (str): Directory to save plot
    """
    metrics = load_metrics(metrics_file)

    if not metrics.get("step"):
        print("No metrics to plot")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create figure with subplots
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

    # Plot 1: Loss over steps
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(metrics["step"], metrics["loss"], linewidth=2, color="#1f77b4", label="Training Loss")
    ax1.set_xlabel("Training Step", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("Training Loss Over Time", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)

    # Plot 2: Smoothed loss
    ax2 = fig.add_subplot(gs[1, 0])
    smoothed_loss = _smooth_curve(metrics["loss"], window=5)
    ax2.plot(metrics["step"], metrics["loss"], alpha=0.3, color="#1f77b4", label="Raw")
    ax2.plot(metrics["step"], smoothed_loss, linewidth=2, color="#ff7f0e", label="Smoothed (window=5)")
    ax2.set_xlabel("Training Step", fontsize=11)
    ax2.set_ylabel("Loss", fontsize=11)
    ax2.set_title("Smoothed Loss", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)

    # Plot 3: Learning rate
    ax3 = fig.add_subplot(gs[1, 1])
    if metrics.get("learning_rate") and any(lr > 0 for lr in metrics["learning_rate"]):
        ax3.plot(metrics["step"], metrics["learning_rate"], linewidth=2, color="#2ca02c")
        ax3.set_xlabel("Training Step", fontsize=11)
        ax3.set_ylabel("Learning Rate", fontsize=11)
        ax3.set_title("Learning Rate Schedule", fontsize=12, fontweight="bold")
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, "No learning rate data", ha="center", va="center", fontsize=11)
        ax3.set_xticks([])
        ax3.set_yticks([])

    # Plot 4: Reward distribution (if available)
    # ax4 = fig.add_subplot(gs[2, 0])
    # if metrics.get("reward") and any(r is not None for r in metrics["reward"]):
    #     ax4.plot(metrics["step"][:len(metrics["reward"])], metrics["reward"], linewidth=2, color="#d62728", label="Reward")
    #     ax4.set_xlabel("Training Step", fontsize=11)
    #     ax4.set_ylabel("Reward", fontsize=11)
    #     ax4.set_title("Training Reward Over Time", fontsize=12, fontweight="bold")
    #     ax4.grid(True, alpha=0.3)
    #     ax4.legend(fontsize=10)
    # else:
    #     ax4.text(0.5, 0.5, "No reward data", ha="center", va="center", fontsize=11)
    #     ax4.set_xticks([])
    #     ax4.set_yticks([])

    # Plot 5: Statistics
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis("off")

    # Calculate statistics
    loss_values = metrics["loss"]
    stats_text = f"""
Training Statistics:

Total Steps: {len(metrics['step'])}
Final Loss: {loss_values[-1]:.4f}
Initial Loss: {loss_values[0]:.4f}
Min Loss: {min(loss_values):.4f}
Max Loss: {max(loss_values):.4f}
Avg Loss: {sum(loss_values) / len(loss_values):.4f}
Loss Change: {loss_values[-1] - loss_values[0]:.4f}
Improvement: {100 * (1 - loss_values[-1] / (loss_values[0] + 1e-8)):.2f}%
"""

    # ax5.text(0.05, 0.95, stats_text, transform=ax5.transAxes, fontsize=11,
    #          verticalalignment="top", fontfamily="monospace",
    #          bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # Overall title
    fig.suptitle("GRPO Training: Verifiable-Reward RL", fontsize=16, fontweight="bold", y=0.995)

    # Save plot
    output_path = output_dir / "training_loss.png"
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    print(f"✓ Loss plot saved to {output_path}")

    # Also save as PDF
    output_path_pdf = output_dir / "training_loss.pdf"
    plt.savefig(str(output_path_pdf), dpi=150, bbox_inches="tight")
    print(f"✓ Loss plot saved to {output_path_pdf}")

    plt.close()


def plot_loss_vs_epochs(metrics_file, num_epochs=None, samples_per_epoch=None, output_dir="."):
    """
    Plot loss organized by epochs.

    Args:
        metrics_file (str): Path to training_metrics.json
        num_epochs (int): Number of epochs (optional)
        samples_per_epoch (int): Samples per epoch (optional)
        output_dir (str): Directory to save plot
    """
    metrics = load_metrics(metrics_file)

    if not metrics.get("step"):
        print("No metrics to plot")
        return

    loss_values = metrics["loss"]
    steps = metrics["step"]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot loss with epoch boundaries
    ax.plot(steps, loss_values, linewidth=2.5, color="#1f77b4", label="Loss")

    # Add epoch boundaries if provided
    if num_epochs and samples_per_epoch:
        epoch_steps = [i * samples_per_epoch for i in range(1, num_epochs)]
        for epoch_step in epoch_steps:
            ax.axvline(x=epoch_step, color="red", linestyle="--", alpha=0.5, linewidth=1)
        ax.text(0, 1.02, "Red dashed lines: Epoch boundaries", transform=ax.transAxes,
                fontsize=10, style="italic", color="red")

    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Training Loss with Epoch Markers", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    output_path = output_dir / "training_loss_epochs.png"
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    print(f"✓ Loss plot (by epochs) saved to {output_path}")

    plt.close()


def _smooth_curve(values, window=5):
    """
    Smooth a curve using a simple moving average.

    Args:
        values (list): Values to smooth
        window (int): Window size for moving average

    Returns:
        list: Smoothed values
    """
    if window < 1:
        return values

    smoothed = []
    for i in range(len(values)):
        start = max(0, i - window // 2)
        end = min(len(values), i + window // 2 + 1)
        smoothed.append(sum(values[start:end]) / (end - start))

    return smoothed


def plot_train_vs_validation(metrics_file, output_dir="."):
    """
    Plot training loss vs validation loss.

    Args:
        metrics_file (str): Path to training_metrics.json
        output_dir (str): Directory to save plot
    """
    metrics = load_metrics(metrics_file)

    if not metrics.get("step"):
        print("No metrics to plot")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot training loss
    ax.plot(metrics["step"], metrics["training_loss"], linewidth=2.5, color="#1f77b4", label="Training Loss", marker='o', markersize=3, alpha=0.7)

    # Plot validation loss if available
    if metrics.get("eval_loss") and any(v is not None for v in metrics["eval_loss"]):
        # eval_loss might be shorter than training_loss, so we need to align steps
        eval_steps = metrics["step"][:len(metrics["eval_loss"])]
        ax.plot(eval_steps, metrics["eval_loss"], linewidth=2.5, color="#ff7f0e", label="Validation Loss", marker='s', markersize=3, alpha=0.7)

    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Training vs Validation Loss", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc="best")

    output_path = output_dir / "train_vs_validation_loss.png"
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    print(f"✓ Train vs validation plot saved to {output_path}")

    plt.close()


def compare_training_runs(metrics_files_dict, output_dir="."):
    """
    Compare multiple training runs.

    Args:
        metrics_files_dict (dict): Dict mapping run names to metrics file paths
        output_dir (str): Directory to save plot
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for idx, (run_name, metrics_file) in enumerate(metrics_files_dict.items()):
        metrics = load_metrics(metrics_file)
        if not metrics.get("step"):
            continue

        color = colors[idx % len(colors)]
        ax.plot(metrics["step"], metrics.get("training_loss", metrics.get("loss")), linewidth=2, label=run_name, color=color)

    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Training Loss Comparison", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    output_path = Path(output_dir) / "training_loss_comparison.png"
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    print(f"✓ Comparison plot saved to {output_path}")

    plt.close()
