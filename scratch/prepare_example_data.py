from pathlib import Path
import geopandas as gpd
import json
import pandas as pd

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

all_field_trees = gpd.read_file(FIELD_TREES).to_crs(crs=26910)
all_plot_bounds = gpd.read_file(PLOT_BOUNDS).to_crs(crs=26910)

with open(SHIFTS) as infile:
    shifts = json.load(infile)

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
