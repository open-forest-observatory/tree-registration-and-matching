from pathlib import Path
import json

import geopandas as gpd
import pandas as pd
import numpy as np

from tree_registration_and_matching.utils import is_overstory

FIELD_TREES = Path(
    "/ofo-share/project-data/species-prediction-project/raw/ground-reference/ofo_ground-reference_trees.gpkg"
)
PLOT_BOUNDS = Path(
    "/ofo-share/project-data/species-prediction-project/raw/ground-reference/ofo_ground-reference_plots.gpkg"
)
SHIFTS = Path(
    "/ofo-share/repos/david/tree-registration-and-matching/data/ofo-example-2/shifts_per_dataset.json"
)

OUTPUT_PLOT_BOUNDS = "/ofo-share/repos/david/tree-registration-and-matching/data/benchmarking-data/plot_bounds.gpkg"
OUTPUT_FIELD_TREES = "/ofo-share/repos/david/tree-registration-and-matching/data/benchmarking-data/field_trees.gpkg"

if False:
    ## Prep shift quality
    shift_quality = pd.read_csv(
        "/ofo-share/repos/david/tree-registration-and-matching/data/ofo-example-2/shift_quality.csv"
    )
    shift_quality.Dataset = shift_quality.Dataset.str.replace(".tif", "")
    shift_quality_dict = {
        k: v
        for k, v in zip(
            shift_quality.Dataset.to_list(), shift_quality.Quality.to_list()
        )
    }
    with open(
        "/ofo-share/repos/david/tree-registration-and-matching/data/benchmarking-data/shift_quality.json",
        "w",
    ) as infile:
        json.dump(shift_quality_dict, infile)

if False:
    ## Merge detected trees
    detected_tree_files = list(
        Path(
            "/ofo-share/repos/david/tree-registration-and-matching/data/ofo-example-2/detected-trees"
        ).glob("*")
    )

    all_detected_trees = []
    for detected_tree_file in detected_tree_files:
        detected_trees = gpd.read_file(detected_tree_file)
        detected_trees["dataset_id"] = detected_tree_file.stem
        all_detected_trees.append(detected_trees)

    all_detected_trees = gpd.GeoDataFrame(
        pd.concat(all_detected_trees), crs=all_detected_trees[0].crs
    )
    all_detected_trees.to_file(
        "/ofo-share/repos/david/tree-registration-and-matching/data/ofo-tree-registration/detected-trees.gpkg"
    )

all_field_trees = gpd.read_file(FIELD_TREES).to_crs(crs=26910)
all_plot_bounds = gpd.read_file(PLOT_BOUNDS).to_crs(crs=26910)

with open(SHIFTS) as infile:
    shifts = json.load(infile)

nan_height = all_field_trees.height.isna()
all_field_trees[nan_height].height = all_field_trees[nan_height].height_allometric

# For any remaining missing height values that have DBH, use an allometric equation to compute
# the height
nan_height = all_field_trees.height.isna()
# These parameters were fit on paired height, DBH data from this dataset.
allometric_height_func = lambda x: 1.3 + np.exp(
    -0.3136489123372108 + 0.84623571 * np.log(x)
)
# Compute the allometric height and assign it
allometric_height = allometric_height_func(all_field_trees[nan_height].dbh.to_numpy())
all_field_trees.loc[nan_height, "height"] = allometric_height

# Filter out any trees that still don't have height (just 1 in current experiments)
ground_reference_trees = all_field_trees[~all_field_trees.height.isna()]

# Drop dead trees
live_trees = all_field_trees.live_dead != "D"
all_field_trees = all_field_trees[live_trees]
# Drop short trees
tall_enough_trees = all_field_trees.height > 10
all_field_trees = all_field_trees[tall_enough_trees]

output_plot_bounds = []
output_field_trees = []

for dataset_id, v in shifts.items():
    shift = v[0]

    plot_id = dataset_id[:4]

    field_trees = all_field_trees.query("plot_id == @plot_id")
    plot_bounds = all_plot_bounds.query("plot_id == @plot_id")

    field_trees.geometry = field_trees.translate(xoff=shift[0], yoff=shift[1])
    plot_bounds.geometry = plot_bounds.translate(xoff=shift[0], yoff=shift[1])

    overstory = is_overstory(field_trees)

    field_trees = field_trees[overstory]

    plot_bounds["dataset_id"] = dataset_id
    field_trees["dataset_id"] = dataset_id

    output_plot_bounds.append(plot_bounds)
    output_field_trees.append(field_trees)


output_plot_bounds = gpd.GeoDataFrame(pd.concat(output_plot_bounds), crs=26910)
output_field_trees = gpd.GeoDataFrame(pd.concat(output_field_trees), crs=26910)

output_plot_bounds.to_file(OUTPUT_PLOT_BOUNDS)
output_field_trees.to_file(OUTPUT_FIELD_TREES)
