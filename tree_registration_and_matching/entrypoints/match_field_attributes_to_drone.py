import argparse
import logging
from pathlib import Path

import geopandas as gpd

from tree_registration_and_matching.add_attributes import match_field_and_drone_trees
from tree_registration_and_matching.utils import ensure_height_is_present, is_overstory

DEFAULT_MIN_FIELD_HEIGHT = 10.0
DEFAULT_MIN_MATCHED_TREES = 10
DEFAULT_MAX_DECAY_CLASS = 2


def match_field_attributes_to_drone(
    field_trees_file: Path | str,
    drone_trees_file: Path | str,
    drone_crowns_file: Path | str,
    field_bounds_file: Path | str,
    output_file: Path | str,
    min_field_height: float = DEFAULT_MIN_FIELD_HEIGHT,
    min_matched_trees: int = DEFAULT_MIN_MATCHED_TREES,
    max_decay_class: int = DEFAULT_MAX_DECAY_CLASS,
):
    """Apply pre- and post-matching business logic which is specific to attributes in the OFO catalog

    Args:
        field_trees_file (Path | str): Path to the field surveyed trees, represented as points. These data should have the following attributes: decay_class, live_dead, height, height_allometric, and dbh.
        drone_trees_file (Path | str): Path to the drone detected tree tops, represented as points
        drone_crowns_file (Path | str): Path to the drone detected tree crowns, represented as polygons. These are linked to the tree tops by the tree_top_unique_ID which corresponds to the tree top unique ID field
        field_bounds_file (Path | str): Path to the extent of what was surveyed in the field survey.
        output_file (Path | str): Path to write the drone crowns with additional field attributes to.
        min_field_height (float, optional): The minimum height of trees to be retained after matching. Defaults to 10.0.
        min_matched_trees (int, optional): The minimum number of trees that must be matched to write anything out. Defaults to 10.
        max_decay_class (int, optional): The maximum decay class of trees to be retained. Defaults to 2.

    Raises:
        ValueError: If no trees meet the criteria for matching
        ValueError: If not enough trees are matched
    """
    # Read the files
    field_trees = gpd.read_file(field_trees_file)
    drone_trees = gpd.read_file(drone_trees_file)
    drone_crowns = gpd.read_file(drone_crowns_file)
    field_bounds = gpd.read_file(field_bounds_file)

    logging.info(f"A total of {len(field_trees)} were present")
    # The decay class specifies how severely a dead trees is decaying. At values above decay class 2,
    # it is expected that the stem may be broken. This would cause issues estimating the height from
    # DBH, and likely suggests a tree that will overall not be reconstructed well. Therefore, these
    # trees are dropped prior to matching.
    decay_mask = field_trees.decay_class > max_decay_class
    logging.info(f"Removing {decay_mask.sum()} trees due to decay")
    field_trees = field_trees[~decay_mask].copy()
    # Impute height for as many trees as possible, using other attributes. Any trees for which it is
    # impossible to compute height are dropped.
    field_trees = ensure_height_is_present(field_trees)
    # Remove understory trees
    overstory_mask = is_overstory(field_trees)
    logging.info(f"Removing {(~overstory_mask).sum()} trees due to being understory")
    field_trees = field_trees[overstory_mask]

    if len(field_trees) == 0:
        raise ValueError("No field trees retained for matching after all checks.")

    logging.info(
        f"Matching {len(field_trees)} field trees to {len(drone_trees)} drone trees"
    )

    # Perform matching
    # The strategy is to include both short trees and dead trees in matching, and then drop them later.
    # This approach could be reconsidered in the future, in favor of pre-dropping these trees.
    drone_crowns_with_additional_attributes = match_field_and_drone_trees(
        field_trees=field_trees,
        drone_trees=drone_trees,
        drone_crowns=drone_crowns,
        field_perim=field_bounds,
        keep_only_matched_crowns=True,
    )
    logging.info(f"Matched {len(drone_crowns_with_additional_attributes)} trees")

    # Drop any dead trees. Note that there may be classes other than "L" (live) in the output
    # but these are assumed to be live as well.
    drone_crowns_with_additional_attributes = drone_crowns_with_additional_attributes[
        drone_crowns_with_additional_attributes.live_dead != "D"
    ]
    # Drop any crowns that were less than min_field_height tall
    drone_crowns_with_additional_attributes = drone_crowns_with_additional_attributes[
        drone_crowns_with_additional_attributes.height_field > min_field_height
    ]
    final_n_matched = len(drone_crowns_with_additional_attributes)
    logging.info(f"After filtering all matched trees, {final_n_matched} trees remain")

    if final_n_matched >= min_matched_trees:
        # Save the drone crowns with additional field attributes to the file
        output_file = Path(output_file)
        output_file.parent.mkdir(exist_ok=True, parents=True)
        drone_crowns_with_additional_attributes.to_file(output_file)
    else:
        raise ValueError(f"Not enough matched trees (only {final_n_matched})")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply pre- and post-matching business logic which is specific to attributes in the OFO catalog"
    )
    parser.add_argument(
        "field_trees_file",
        type=Path,
        help="Path to the field surveyed trees, represented as points",
    )
    parser.add_argument(
        "drone_trees_file",
        type=Path,
        help="Path to the drone detected tree tops, represented as points",
    )
    parser.add_argument(
        "drone_crowns_file",
        type=Path,
        help="Path to the drone detected tree crowns, represented as polygons. These are linked to the tree tops by the tree_top_unique_ID",
    )
    parser.add_argument(
        "field_bounds_file",
        type=Path,
        help="Path to the extent of what was surveyed in the field survey",
    )
    parser.add_argument(
        "output_file",
        type=Path,
        help="Path to write the drone crowns with additional field attributes to",
    )
    parser.add_argument(
        "--min-field-height",
        type=float,
        default=DEFAULT_MIN_FIELD_HEIGHT,
        help=f"The minimum height of trees to be considered. Defaults to {DEFAULT_MIN_FIELD_HEIGHT}.",
    )
    parser.add_argument(
        "--min-matched-trees",
        type=int,
        default=DEFAULT_MIN_MATCHED_TREES,
        help=f"The minimum number of trees that must be matched to write anything out. Defaults to {DEFAULT_MIN_MATCHED_TREES}.",
    )
    parser.add_argument(
        "--max-decay-class",
        type=int,
        default=DEFAULT_MAX_DECAY_CLASS,
        help=f"The maximum decay class of trees to be retained. Defaults to {DEFAULT_MAX_DECAY_CLASS}.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    match_field_attributes_to_drone(**vars(args))
