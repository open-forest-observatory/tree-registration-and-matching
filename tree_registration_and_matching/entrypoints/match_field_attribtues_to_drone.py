from pathlib import Path
import argparse

import geopandas as gpd
import pandas as pd
from tree_registration_and_matching.add_attributes import match_field_and_drone_trees
from tree_registration_and_matching.utils import ensure_height_is_present, is_overstory

# Input variables
pair_name = "{{inputs.parameters.drone-ground-pair-name}}"
ground_plot_id = "{{inputs.parameters.ground-plot-id}}"
field_trees_file = "{{inputs.parameters.field-trees-file}}"
field_bounds_file = "{{inputs.parameters.field-bounds-file}}"
drone_trees_file = "{{inputs.parameters.drone-trees-file}}"
drone_crowns_file = "{{inputs.parameters.drone-crowns-file}}"
registration_file = "{{inputs.parameters.registration-file}}"
registration_quality_threshold = float(
    "{{inputs.parameters.registration-quality-threshold}}"
)

output_file = Path("{{inputs.parameters.output-file}}")
# Read all the products
field_trees = gpd.read_file(field_trees_file)
field_bounds = gpd.read_file(field_bounds_file)

drone_trees = gpd.read_file(drone_trees_file)
drone_crowns = gpd.read_file(drone_crowns_file)

# Subset to the specified plot
field_trees = field_trees.query("plot_id==@ground_plot_id")
field_bounds = field_bounds.query("plot_id==@ground_plot_id")

registration = pd.read_csv(registration_file)

# See if it passes the quality threshold
ratio_quality_metric = registration["ratio_quality_metric"].iloc[0]
if ratio_quality_metric < registration_quality_threshold:
    raise ValueError(
        f"The quality metric ({ratio_quality_metric}) is less than the threshold ({registration_quality_threshold})"
    )

# Determine the shift that the CRS needs to be interpreted in
working_CRS = registration["shift_CRS"].iloc[0]
drone_crown_crs = drone_crowns.crs

# Apply the shift
x_shift = registration["estimated_shift_x"].iloc[0]
y_shift = registration["estimated_shift_y"].iloc[0]

field_trees.to_crs(working_CRS, inplace=True)
field_bounds.to_crs(working_CRS, inplace=True)

field_trees.geometry = field_trees.geometry.translate(xoff=x_shift, yoff=y_shift)
field_bounds.geometry = field_bounds.geometry.translate(xoff=x_shift, yoff=y_shift)


def match_field_attributes_to_drone(
    field_trees: gpd.GeoDataFrame,
    drone_trees: gpd.GeoDataFrame,
    drone_crowns: gpd.GeoDataFrame,
    field_bounds=gpd.GeoDataFrame,
    keep_only_matched_crowns: bool = True,
):
    print(f"A total of {len(field_trees)} were present")
    # The decay class specifies how severely a dead trees is decaying. At values above decay class 2,
    # it is expected that the stem may be broken. This would cause issues estimating the height from
    # DBH, and likely suggests a tree that will overall not be reconstructed well. Therefore, these
    # trees are dropped prior to matching.
    decay_mask = field_trees.decay_class > 2
    print(f"Removing {decay_mask.sum()} trees due to decay")
    field_trees = field_trees[~decay_mask]
    # Impute height for as many trees as possible, using other attributes
    field_trees = ensure_height_is_present(field_trees)
    # Remove understory trees
    overstory_mask = is_overstory(field_trees)
    print(f"Removing {(~overstory_mask).sum()} trees due to being understory")
    field_trees = field_trees[overstory_mask]

    print(f"Matching {len(field_trees)} field trees to {len(drone_trees)} drone trees")
    # Perform matching
    drone_crowns_with_additional_attributes = match_field_and_drone_trees(
        field_trees=field_trees,
        drone_trees=drone_trees,
        drone_crowns=drone_crowns,
        field_perim=field_bounds,
        keep_only_matched_crowns=keep_only_matched_crowns,
    )

    # Convert back to the original drone CRS
    drone_crowns_with_additional_attributes.to_crs(drone_crown_crs, inplace=True)
    print(f"Matched {len(drone_crowns_with_additional_attributes)} trees")

    # Drop crowns matched to field trees with no species code
    drone_crowns_with_additional_attributes = (
        drone_crowns_with_additional_attributes.dropna(subset=["species_code"])
    )
    # Drop any dead trees. Note that there may be classes other than "L" (live) in the output
    # but these are assumed to be live as well.
    drone_crowns_with_additional_attributes = drone_crowns_with_additional_attributes[
        drone_crowns_with_additional_attributes.live_dead != "D"
    ]
    # Drop any crowns that were less than 10m tall
    drone_crowns_with_additional_attributes = drone_crowns_with_additional_attributes[
        drone_crowns_with_additional_attributes.height_field > 10
    ]
    final_n_matched = len(drone_crowns_with_additional_attributes)
    print(f"After filtering all matched trees, {final_n_matched} trees remain")

    print(f"Matched {final_n_matched} trees")
    if final_n_matched >= 10:
        # Save the drone crowns with additional field attributes to the file
        output_file.parent.mkdir(exist_ok=True, parents=True)
        drone_crowns_with_additional_attributes.to_file(output_file)
    else:
        raise ValueError(f"Not enough matched trees (only {final_n_matched})")


def parse_args():
    parser = argparse.ArgumentParser()
