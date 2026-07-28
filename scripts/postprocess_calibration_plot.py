import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score

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
    p.add_argument("--median", action="store_true", help="Use median and 25-75%% quartile bands instead of mean and 1 std")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    calibration_dir: pathlib.Path = args.calibration_dir.expanduser().resolve()

    # Global font size parameters
    TICK_LABEL_SIZE = 8
    AXIS_LABEL_SIZE = 10
    LEGEND_SIZE = 10
    TITLE_SIZE = 11
    CONFUSION_NUMBER_SIZE = 8

    continuous = ["logP", "qed", "SAS", "FractionCSP3", "molWt", "TPSA", "MR"]
    discrete = ["hbd", "hba", "num_rings", "num_rotable_bonds", "num_chiral_centers"]

    scaler = PropertyScaler(args.stats_path)

    fig, axs = plt.subplots(
        3,
        4,
        figsize=(7.5, 5.1),
        gridspec_kw={"height_ratios": [1, 1, 0.88]},
    )
    plt.subplots_adjust(
        left=0.08, right=0.95, top=0.99, bottom=0.08, wspace=0.35, hspace=0.35
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
                    s=8,
                    label="Predictions",
                )

                if args.median:
                    # Median and 25-75% quartile bands
                    stats = (
                        data.groupby(f"prompted_{mode}")[f"pred_{mode}"]
                        .agg([
                            ("center", "median"),
                            ("lo", lambda x: x.quantile(0.25)),
                            ("hi", lambda x: x.quantile(0.75)),
                        ])
                        .reset_index()
                    )
                else:
                    # Mean and 1 std
                    stats = (
                        data.groupby(f"prompted_{mode}")[f"pred_{mode}"]
                        .agg(["mean", "std"])
                        .reset_index()
                    )
                    stats["center"] = stats["mean"]
                    stats["lo"] = stats["mean"] - stats["std"]
                    stats["hi"] = stats["mean"] + stats["std"]

                # Plot center line
                ax.plot(
                    stats[f"prompted_{mode}"],
                    stats["center"],
                    color="blue",
                    linewidth=3,
                )

                # Plot shaded band
                ax.fill_between(
                    stats[f"prompted_{mode}"],
                    stats["lo"],
                    stats["hi"],
                    color="blue",
                    alpha=0.2,
                )

                # Plot ideal line
                min_val = scaler.get("mean", mode) - 2 * scaler.get("std", mode)
                max_val = scaler.get("mean", mode) + 2 * scaler.get("std", mode)
                ax.plot(
                    [min_val, max_val],
                    [min_val, max_val],
                    "k--",
                    linewidth=3,
                )

                ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)

                r2 = r2_score(stats[f"prompted_{mode}"], stats["center"])
                mae = mean_absolute_error(data[f"prompted_{mode}"], data[f"pred_{mode}"])
                ax.text(0.98, 0.02, f"R²={r2:.2g}  MAE={mae:.2g}", transform=ax.transAxes, fontsize=9, va="bottom", ha="right",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="lightgray", edgecolor="none", alpha=0.9))

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

                print(confusion)

                # Plot heatmap
                sns.heatmap(
                    confusion,
                    annot=False,
                    fmt="d",
                    cmap="Blues",
                    ax=ax,
                )
                ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
                ax.set_title(f"{mode}", fontsize=TITLE_SIZE)

                mae = mean_absolute_error(valid_data[f"prompted_{mode}"], valid_data[f"pred_{mode}"])
                ax.text(0.98, 0.02, f"MAE={mae:.2g}", transform=ax.transAxes, fontsize=9, va="bottom", ha="right",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="lightgray", edgecolor="none", alpha=0.9))

            # Label y on the first column, x on the last row
            ax.set_ylabel("Predicted" if i % 4 == 0 else "", fontsize=AXIS_LABEL_SIZE)
            ax.set_xlabel("Prompted" if i >= 8 else "", fontsize=AXIS_LABEL_SIZE)

    plt.savefig(
        args.figure_savepath,
        bbox_inches="tight",
        pad_inches=0.02,
        dpi=800,
    )
