from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tree_registration_and_matching.benchmark import score_approach
from tree_registration_and_matching.constants import DATA_DIR
from tree_registration_and_matching.register_CHM import find_best_shift

DETECTED_TREES = Path(DATA_DIR, "ofo-example-2", "detected-trees")
CHMS = Path(DATA_DIR, "ofo-example-2", "CHMs")
FIELD_TREES = Path(DATA_DIR, "ofo-example-2", "field_trees")
PLOTS_FILE = Path(DATA_DIR, "ofo-example-2", "ofo_ground-reference_plots.gpkg")

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
    shifts_CHM = score_approach(
        field_trees_folder=FIELD_TREES,
        CHMs_or_detected_trees_folder=CHMS,
        plots_file=PLOTS_FILE,
        alignment_algorithm=find_best_shift,
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
    CHM_shifts = np.load(OUTPUT_CHM)
    MEE_shifts = np.load(OUTPUT_MEE)

    # Show the magtides of the error for indivdual plots with the two approaches
    CHM_errors = np.linalg.norm(CHM_shifts, axis=1)
    MEE_errors = np.linalg.norm(MEE_shifts, axis=1)

    plt.title("MEE error magnitudes vs. CHM error magnitudes")
    plt.scatter(MEE_errors, CHM_errors[:3])
    plt.xlabel("MEE error magnitudes")
    plt.ylabel("CHM error magnitudes")
    plt.xlim(0, np.sqrt(2) * 12)
    plt.ylim(0, np.sqrt(2) * 12)
    plt.show()

    # Show the x, y coordinates of the errors for the two approaches on separate subplots
    f, ax = plt.subplots(1, 2)
    ax[0].scatter(MEE_shifts[:, 0], MEE_shifts[:, 1])
    ax[1].scatter(CHM_shifts[:, 0], CHM_shifts[:, 1])

    ax[0].set_xlim((-12, 12))
    ax[0].set_ylim((-12, 12))
    ax[1].set_xlim((-12, 12))
    ax[1].set_ylim((-12, 12))

    ax[0].set_title("MEE displacements")
    ax[1].set_title("CHM displacements")
    plt.show()
