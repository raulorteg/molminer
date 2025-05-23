import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.append("..")
from molminer.scalers import PropertyScaler
import pathlib
import argparse


def _parse_args() -> argparse.Namespace:
    """Command-line interface."""
    p = argparse.ArgumentParser(description="")
    p.add_argument(
        "--calibration_dir",
        type=pathlib.Path,
        help="Folder that contains the calibration logs",
    )
    p.add_argument("--stats_path", required=True, type=pathlib.Path)
    p.add_argument("--figure_savepath", required=True, type=pathlib.Path)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    calibration_dir: pathlib.Path = args.calibration_dir.expanduser().resolve()

    # Global font size parameters
    TICK_LABEL_SIZE = 9
    AXIS_LABEL_SIZE = 11
    LEGEND_SIZE = 10
    TITLE_SIZE = 12
    CONFUSION_NUMBER_SIZE = 8

    continuous = ["logP", "qed", "SAS", "FractionCSP3", "molWt", "TPSA", "MR"]
    discrete = ["hbd", "hba", "num_rings", "num_rotable_bonds", "num_quiral_centers"]

    scaler = PropertyScaler(args.stats_path)

    fig, axs = plt.subplots(4, 3, figsize=(7.5, 9.5))
    plt.subplots_adjust(
        left=0.08, right=0.95, top=0.99, bottom=0.08, wspace=0.35, hspace=0.45
    )
    axs_flat = axs.ravel()

    reference_labels = continuous + discrete

    for i, mode in enumerate(reference_labels):
        filename = calibration_dir / f"{mode}_calibration.txt"

        if os.path.exists(filename):
            data = pd.read_csv(filename)

            ax = axs_flat[i]
            ax.set_title(f"{mode}", fontsize=TITLE_SIZE)

            if mode in continuous:
                # Scatter plot (to show individual samples)
                scatter = ax.scatter(
                    data[f"prompted_{mode}"],
                    data[f"pred_{mode}"],
                    alpha=0.3,
                    s=15,
                    label="Predictions",
                )

                # Calculate mean and std for each unique prompted value
                stats = (
                    data.groupby(f"prompted_{mode}")[f"pred_{mode}"]
                    .agg(["mean", "std"])
                    .reset_index()
                )

                # Plot mean line
                (mean_line,) = ax.plot(
                    stats[f"prompted_{mode}"],
                    stats["mean"],
                    color="blue",
                    label="Mean prediction",
                    linewidth=3,
                )

                # Plot \pm 1 std shaded area
                ax.fill_between(
                    stats[f"prompted_{mode}"],
                    stats["mean"] - stats["std"],
                    stats["mean"] + stats["std"],
                    color="blue",
                    alpha=0.2,
                    label="±1 std dev",
                )

                # Plot ideal line
                min_val = scaler.get("mean", mode) - 2 * scaler.get("std", mode)
                max_val = scaler.get("mean", mode) + 2 * scaler.get("std", mode)
                (ideal_line,) = ax.plot(
                    [min_val, max_val],
                    [min_val, max_val],
                    "k--",
                    label="Ideal",
                    linewidth=3,
                )

                ax.set_xlabel("Prompted", fontsize=AXIS_LABEL_SIZE)
                ax.set_ylabel("Predicted", fontsize=AXIS_LABEL_SIZE)
                ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)

            else:  # discrete properties
                # Filter out negative prompted values
                valid_data = data[data[f"prompted_{mode}"] >= 0]

                # Cast to int (both prompted and predicted)
                valid_data[f"prompted_{mode}"] = valid_data[f"prompted_{mode}"].astype(
                    int
                )
                valid_data[f"pred_{mode}"] = valid_data[f"pred_{mode}"].astype(int)

                # Get all possible values from both prompted and predicted (after filtering)
                all_values = sorted(
                    set(valid_data[f"prompted_{mode}"]).union(
                        set(valid_data[f"pred_{mode}"])
                    )
                )

                # Create confusion matrix with all possible values for both axes
                confusion = pd.crosstab(
                    valid_data[f"pred_{mode}"], valid_data[f"prompted_{mode}"]
                ).reindex(index=all_values, columns=all_values, fill_value=0)

                # Plot heatmap
                sns.heatmap(
                    confusion,
                    annot=False,
                    fmt="d",
                    cmap="Blues",
                    ax=ax,
                )
                ax.set_ylabel("Predicted", fontsize=AXIS_LABEL_SIZE)
                ax.set_xlabel("Prompted", fontsize=AXIS_LABEL_SIZE)
                ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
                ax.set_title(f"{mode}", fontsize=TITLE_SIZE)

        fig.legend(
            [scatter, mean_line, ideal_line],
            ["Predictions", "Mean prediction", "Ideal"],
            loc="upper center",
            fontsize=LEGEND_SIZE,
            ncol=3,
            frameon=False,
            bbox_to_anchor=(0.5, 1.05),
        )

    plt.savefig(
        args.figure_savepath,
        bbox_inches="tight",
        dpi=800,
    )
