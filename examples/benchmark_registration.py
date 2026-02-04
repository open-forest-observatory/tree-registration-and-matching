from pathlib import Path
import json

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tree_registration_and_matching.benchmark import score_approach
from tree_registration_and_matching.constants import DATA_DIR
from tree_registration_and_matching.register_CHM import find_best_shift

DETECTED_TREES = Path(DATA_DIR, "ofo-tree-registration", "detected-trees.gpkg")
CHMS = Path(DATA_DIR, "ofo-tree-registration", "CHMs")
FIELD_TREES = Path(DATA_DIR, "ofo-tree-registration", "field_trees.gpkg")
PLOTS_FILE = Path(DATA_DIR, "ofo-tree-registration", "plot_bounds.gpkg")
QUALITY_FILE = Path(DATA_DIR, "ofo-tree-registration", "shift_quality.json")
SHIFT_FILE = Path(DATA_DIR, "ofo-tree-registration", "shifts_per_dataset.json")
SHIFT_CRS = 26910

OUTPUT_FOLDER = Path(DATA_DIR, "benchmarking_results")
OUTPUT_CHM = Path(OUTPUT_FOLDER, "shifts_CHM.npy")
OUTPUT_MEE = Path(OUTPUT_FOLDER, "shifts_MEE.npy")

OUTPUT_FOLDER.mkdir(exist_ok=True, parents=True)

RANGES = (-12, 12, 0.25)

RUN_MEE = True
RUN_CHM = True
VIS = True

# An m3.2xl node has 64 CPU cores
N_WORKERS = 60

with open(SHIFT_FILE) as infile:
    shifts = json.load(infile)
    # Negate the shift because the convention is it should be what is applied to the field trees
    shifts = {k: -np.array(v) for k, v in shifts.items()}
    # This is what we hope the algorithm reports, the opposite of the applied shift, ordered by
    # dataset ID
    target_shifts = -1 * np.concat([shifts[k] for k in sorted(shifts.keys())], axis=0)

if RUN_MEE:
    shifts_MEE = score_approach(
        DETECTED_TREES,
        FIELD_TREES,
        PLOTS_FILE,
        shifts=shifts,
        shift_CRS=SHIFT_CRS,
        vis_results=False,
        crop_to_plot_bounds=True,
        plot_buffer_distance=10,
        CHM_approach=False,
        n_workers=N_WORKERS,
    )
    np.save(
        OUTPUT_MEE,
        shifts_MEE,
    )

if RUN_CHM:

    def find_best_shift_specialized(tree_points, CHM, plot_bounds):
        return find_best_shift(
            tree_points=tree_points,
            CHM=CHM,
            plot_bounds=plot_bounds,
            x_range=RANGES,
            y_range=RANGES,
        )

    shifts_CHM = score_approach(
        field_trees_file=FIELD_TREES,
        CHMs_or_detected_trees=CHMS,
        plots_file=PLOTS_FILE,
        shifts=shifts,
        shift_CRS=SHIFT_CRS,
        alignment_algorithm=find_best_shift_specialized,
        CHM_approach=True,
        crop_to_plot_bounds=True,
        plot_buffer_distance=10,
        vis_plots=False,
        n_workers=N_WORKERS,
    )
    np.save(
        OUTPUT_CHM,
        shifts_CHM,
    )

if VIS:
    field_trees = gpd.read_file(FIELD_TREES)
    tree_counts = field_trees.dataset_id.value_counts()
    tree_counts = pd.DataFrame(
        {"dataset_id": tree_counts.index, "counts": tree_counts.values}
    )
    tree_counts = tree_counts.sort_values(by="dataset_id", axis=0)

    dataset_ids = field_trees.dataset_id.unique()

    with open(QUALITY_FILE, "r") as infile:
        quality = json.load(infile)
        quality = pd.DataFrame(
            {"dataset": list(quality.keys()), "quality": list(quality.values())}
        )

    quality = quality[quality.dataset.isin(tree_counts.dataset_id)]
    high_quality = (quality.quality >= 3).values
    high_counts = np.array(tree_counts.counts) > 10
    high_counts_and_quality = np.logical_and(high_counts, high_quality)

    CHM_shifts = np.load(OUTPUT_CHM)
    MEE_shifts = np.load(OUTPUT_MEE)

    # Subset to the high quality and high count plots
    CHM_shifts = CHM_shifts[high_counts_and_quality]
    MEE_shifts = MEE_shifts[high_counts_and_quality]
    target_shifts = target_shifts[high_counts_and_quality]
    dataset_ids = dataset_ids[high_counts_and_quality]

    # Set error values to 0
    failed_MEE_shifts = np.logical_and(MEE_shifts[:, 0] == -12, MEE_shifts[:, 1] == -12)
    failed_CHM_shifts = np.logical_not(np.isfinite(CHM_shifts[:, 1]))
    failed_shifts_both = np.logical_and(failed_MEE_shifts, failed_CHM_shifts)

    n_good_plots = CHM_shifts.shape[0]

    print(f"{failed_CHM_shifts.sum()} / {n_good_plots} CHM registrations failed")
    print(f"{failed_MEE_shifts.sum()} / {n_good_plots} MEE registrations failed")
    print(f"{failed_shifts_both.sum()} / {n_good_plots} plots failed for both")

    print(f"The following datasets failed: {dataset_ids[failed_CHM_shifts]}")

    CHM_shifts = np.nan_to_num(CHM_shifts, 0)
    MEE_shifts[failed_MEE_shifts, :] = 0

    # Compute the difference between the computed shift and that which would have actually aligned
    # it to the manually selected location
    CHM_diffs = CHM_shifts - target_shifts
    MEE_diffs = MEE_shifts - target_shifts

    # Show the magtides of the error for indivdual plots with the two approaches
    CHM_errors = np.linalg.norm(CHM_diffs, axis=1)
    MEE_errors = np.linalg.norm(MEE_diffs, axis=1)
    # Compute the errors for leaving the plot at the initial location
    no_shift_errors = np.linalg.norm(target_shifts, axis=1)

    errors = np.array([MEE_errors, CHM_errors, no_shift_errors]).T
    colors = ["red", "blue", "green"]
    # Show histograms
    plt.hist(
        errors,
        bins=np.arange(0, 20 * np.sqrt(2), 1),
        histtype="bar",
        label=["MEE", "CHM", "no shift"],
    )
    plt.legend()
    plt.title("Paired histogram")
    plt.xlabel("Errors from true shift (m)")
    plt.show()

    plt.title("MEE error magnitudes vs. CHM error magnitudes")
    plt.scatter(MEE_errors, CHM_errors, alpha=0.5)

    # Compute trend line between MEE and CHM errors
    z = np.polyfit(MEE_errors, CHM_errors, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(MEE_errors.min(), MEE_errors.max(), 100)
    plt.plot(x_trend, p(x_trend), "r--", label=f"Trend line: y={z[0]:.2f}x+{z[1]:.2f}")

    plt.xlabel("MEE error magnitudes")
    plt.ylabel("CHM error magnitudes")
    plt.xlim(-0.5, np.sqrt(2) * 20)
    plt.ylim(-0.5, np.sqrt(2) * 20)
    plt.legend()
    plt.show()

    # Show the x, y coordinates of the errors for the two approaches on separate subplots
    f, ax = plt.subplots(1, 2)
    ax[0].scatter(MEE_diffs[:, 0], MEE_diffs[:, 1], alpha=0.2)
    ax[1].scatter(CHM_diffs[:, 0], CHM_diffs[:, 1], alpha=0.2)

    ax[0].set_xlim((-20.0, 20.0))
    ax[0].set_ylim((-20.0, 20.0))
    ax[1].set_xlim((-20.0, 20.0))
    ax[1].set_ylim((-20.0, 20.0))

    ax[0].set_title("MEE displacement errors")
    ax[1].set_title("CHM displacement errors")
    plt.show()

    # Show the x, y coordinates of the errors for the two approaches on separate subplots
    f, ax = plt.subplots(1, 2)
    ax[0].scatter(MEE_shifts[:, 0], MEE_shifts[:, 1], alpha=0.2)
    ax[1].scatter(CHM_shifts[:, 0], CHM_shifts[:, 1], alpha=0.2)

    ax[0].set_xlim((-20.0, 20.0))
    ax[0].set_ylim((-20.0, 20.0))
    ax[1].set_xlim((-20.0, 20.0))
    ax[1].set_ylim((-20.0, 20.0))

    ax[0].set_title("MEE shifts")
    ax[1].set_title("CHM shifts")
    plt.suptitle("Shifts")
    plt.show()
