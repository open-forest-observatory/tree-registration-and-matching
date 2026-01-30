from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tree_registration_and_matching.benchmark import score_approach
from tree_registration_and_matching.constants import DATA_DIR
from tree_registration_and_matching.register_CHM import find_best_shift

DETECTED_TREES = Path(DATA_DIR, "ofo-example-2", "detected-trees")
CHMS = Path(DATA_DIR, "ofo-example-2", "CHMs")
FIELD_TREES = Path(DATA_DIR, "ofo-example-2", "field_trees")
PLOTS_FILE = Path(DATA_DIR, "ofo-example-2", "ofo_ground-reference_plots.gpkg")
QUALITY_FILE = Path(DATA_DIR, "ofo-example-2", "shift_quality.csv")

OUTPUT_FOLDER = Path(DATA_DIR, "benchmarking_results")
OUTPUT_CHM = Path(OUTPUT_FOLDER, "shifts_CHM.npy")
OUTPUT_MEE = Path(OUTPUT_FOLDER, "shifts_MEE.npy")

OUTPUT_FOLDER.mkdir(exist_ok=True, parents=True)

RUN_MEE = False
RUN_CHM = False
VIS = True

if RUN_MEE:
    shifts_MEE = score_approach(
        DETECTED_TREES,
        FIELD_TREES,
        PLOTS_FILE,
        vis_results=False,
        crop_to_plot_bounds=True,
        plot_buffer_distance=10,
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
            x_range=(-12, 12, 0.25),
            y_range=(-12, 12, 0.25),
        )

    shifts_CHM = score_approach(
        field_trees_folder=FIELD_TREES,
        CHMs_or_detected_trees_folder=CHMS,
        plots_file=PLOTS_FILE,
        alignment_algorithm=find_best_shift_specialized,
        CHM_approach=True,
        crop_to_plot_bounds=True,
        plot_buffer_distance=10,
        vis_plots=False,
    )
    np.save(
        OUTPUT_CHM,
        shifts_CHM,
    )

if VIS:
    field_trees = sorted(FIELD_TREES.glob("*"))
    datasets = [str(f.stem) + ".tif" for f in field_trees]

    quality = pd.read_csv(QUALITY_FILE)
    quality = quality[quality.Dataset.isin(datasets)]
    high_quality = (quality.Quality >= 3).values

    CHM_shifts = np.load(OUTPUT_CHM)
    MEE_shifts = np.load(OUTPUT_MEE)

    CHM_shifts = CHM_shifts[high_quality]
    MEE_shifts = MEE_shifts[high_quality]

    # Replace nans with the worst value
    CHM_shifts = np.nan_to_num(CHM_shifts, nan=12)

    # Show the magtides of the error for indivdual plots with the two approaches
    CHM_errors = np.linalg.norm(CHM_shifts, axis=1)
    MEE_errors = np.linalg.norm(MEE_shifts, axis=1)

    errors = np.array([MEE_errors, CHM_errors]).T
    colors = ["red", "blue"]
    # Show histograms
    plt.hist(
        errors,
        bins=np.linspace(0, 12 * np.sqrt(2), 20),
        histtype="bar",
        label=["MEE", "CHM"],
    )
    plt.legend()
    plt.title("Paired histogram")
    plt.xlabel("Errors from true shift (m)")
    plt.show()

    plt.title("MEE error magnitudes vs. CHM error magnitudes")
    plt.scatter(MEE_errors, CHM_errors)

    # Compute trend line between MEE and CHM errors
    z = np.polyfit(MEE_errors, CHM_errors, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(MEE_errors.min(), MEE_errors.max(), 100)
    print(x_trend)
    print(p(x_trend))
    plt.plot(x_trend, p(x_trend), "r--", label=f"Trend line: y={z[0]:.2f}x+{z[1]:.2f}")

    plt.xlabel("MEE error magnitudes")
    plt.ylabel("CHM error magnitudes")
    plt.xlim(-0.5, np.sqrt(2) * 12.5)
    plt.ylim(-0.5, np.sqrt(2) * 12.5)
    plt.legend()
    plt.show()

    # Show the x, y coordinates of the errors for the two approaches on separate subplots
    f, ax = plt.subplots(1, 2)
    ax[0].scatter(MEE_shifts[:, 0], MEE_shifts[:, 1])
    ax[1].scatter(CHM_shifts[:, 0], CHM_shifts[:, 1])

    ax[0].set_xlim((-12.5, 12.5))
    ax[0].set_ylim((-12.5, 12.5))
    ax[1].set_xlim((-12.5, 12.5))
    ax[1].set_ylim((-12.5, 12.5))

    ax[0].set_title("MEE displacements")
    ax[1].set_title("CHM displacements")
    plt.show()
