"""Shared training-loss plotting, used by every training script in this
directory (models_bank.py, erm_baselines.py, utils.py's train_method,
difficulty_shift/diagnostic_experiment.py) instead of each keeping its own
near-identical copy of plot_training_curves.

Every training loop in utils.py now returns (avg_loss, num_steps, num_tokens)
per epoch, so callers can plot loss against epoch index, optimizer step
count, or tokens processed -- not just epoch, which is the only axis the
duplicated per-file versions of this used to support.
"""
import os

import matplotlib.pyplot as plt

_X_AXES = {
    "epoch": ("Epoch", None),  # x-values are just 1..len(train_losses)
    "step": ("Cumulative optimizer steps", "cumulative_steps"),
    "token": ("Cumulative tokens processed", "cumulative_tokens"),
}


def plot_loss_curves(curves, xlabel, output_path, ylabel="Loss", title=None):
    """Draw one or more (x, y) lines on a single figure and save it.

    `curves` is a list of {"label": str, "x": sequence, "y": sequence} dicts.
    """
    plt.figure(figsize=(10, 6))

    for curve in curves:
        plt.plot(curve["x"], curve["y"], marker="o", label=curve["label"])

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.grid()
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path


def plot_training_loss_curves(results_by_model, output_dir, filename_prefix="training_curves", title_prefix=""):
    """Plot train/validation loss vs. epoch, optimizer step, and tokens
    processed, for one or more named models on the same axes.

    `results_by_model`: {model_name: {"train_losses": [...], "val_losses": [...],
    "cumulative_steps": [...], "cumulative_tokens": [...]}}, one entry per
    epoch in every list, all the same length within a given model.

    For a single-model plot, pass a single-entry dict, e.g.
    {"ERM": results}.

    Returns {"epoch": path, "step": path, "token": path}.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_paths = {}

    for x_axis, (xlabel, x_key) in _X_AXES.items():
        curves = []

        for model_name, results in results_by_model.items():
            train_losses = results["train_losses"]
            val_losses = results["val_losses"]

            if x_key is None:
                x_values = range(1, len(train_losses) + 1)
            else:
                x_values = results[x_key]

            curves.append({"label": f"{model_name} Train", "x": x_values, "y": train_losses})
            curves.append({"label": f"{model_name} Validation", "x": x_values, "y": val_losses})

        output_path = os.path.join(output_dir, f"{filename_prefix}_by_{x_axis}.png")
        title = f"{title_prefix}Training and Validation Loss vs. {xlabel}".strip()

        output_paths[x_axis] = plot_loss_curves(
            curves,
            xlabel=xlabel,
            output_path=output_path,
            title=title,
        )

    return output_paths
