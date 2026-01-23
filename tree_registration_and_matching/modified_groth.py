"""
Whale Shark Spot Pattern Matching using Modified Groth Algorithm

Based on: Arzoumanian et al. (2005) "An astronomical pattern-matching algorithm
for computer-aided identification of whale sharks Rhincodon typus"
Journal of Applied Ecology 42:999-1011

This implementation adapts the Groth (1986) star-matching algorithm for
whale shark spot pattern identification.
"""

import numpy as np
from itertools import combinations
from typing import List, Tuple, Dict
from dataclasses import dataclass
import matplotlib.pyplot as plt


@dataclass
class Triangle:
    """Represents a triangle formed by three spots"""

    vertices: Tuple[int, int, int]  # Indices of the three spots
    sides: np.ndarray  # Lengths of the three sides
    R: float  # Ratio of shortest to longest side
    C: float  # Cosine of angle at vertex 1
    orientation: int  # +1 for CCW, -1 for CW
    t2C: float  # The uncertainty (squared) in the cosine
    t2R: float  # The uncertainty (squared) in the ratio
    log_p: float  # The log of the perimiter

    def score_match(self, other: "Triangle", unmatched_value=100):
        """Return the score and whether or not the threshold is satisifed"""
        R_error = (self.R - other.R) ** 2
        C_error = (self.C - other.C) ** 2

        t_R = self.t2R + other.t2R
        t_C = self.t2C + other.t2C

        matches = (
            (R_error < t_R)
            and (C_error < t_C)
            and (self.orientation == other.orientation)
        )
        return (R_error + C_error) if matches else unmatched_value


class SpotPattern:
    """Represents a spot pattern from a whale shark image"""

    def __init__(self, spots: np.ndarray, name: str = ""):
        """
        Initialize with spot coordinates

        Args:
            spots: Nx2 array of (x, y) coordinates
            name: Optional identifier for this pattern
        """
        self.spots = np.array(spots, dtype=float)
        self.name = name
        self.triangles = []
        self._normalize_units()
        self._build_triangles()

    def _normalize_units(self):
        # Keep the aspect the same while normalizing the coordinates to the range (0, 1)
        min_coords = np.min(self.spots, axis=0, keepdims=True)
        max_coords = np.max(self.spots, axis=0, keepdims=True)

        biggest_diff = (max_coords - min_coords).max()
        self.spots = (self.spots - min_coords) / biggest_diff

    def _build_triangles(self, epsilon=5e-3):
        """Build all possible triangles from spot triplets"""
        n_spots = len(self.spots)

        # Generate all combinations of 3 spots
        for indices in combinations(range(n_spots), 3):
            a, b, c = indices

            pa, pb, pc = self.spots[a], self.spots[b], self.spots[c]

            # Calculate side lengths
            side_ab = np.linalg.norm(pb - pa)
            side_bc = np.linalg.norm(pc - pb)
            side_ca = np.linalg.norm(pa - pc)
            # If it's index 0, that means vertex c was not included, etc.
            not_included_dict = {0: c, 1: a, 2: b}

            sorting_inds = tuple(np.argsort([side_ab, side_bc, side_ca]).squeeze())

            # Vertex 1 is not included in the intermediate side.
            i1 = not_included_dict[sorting_inds[1]]
            # Vertex 2 is not included in the longest side
            i2 = not_included_dict[sorting_inds[2]]
            # Vertex 3 is not included in the shortest side
            i3 = not_included_dict[sorting_inds[0]]

            # Rerun with the correct order
            p1, p2, p3 = self.spots[i1], self.spots[i2], self.spots[i3]

            # Calculate side lengths
            side_1 = float(np.linalg.norm(p2 - p3))
            side_2 = float(np.linalg.norm(p1 - p2))
            side_3 = float(np.linalg.norm(p1 - p3))

            sides = [side_1, side_2, side_3]

            sorting_inds = np.argsort([side_1, side_2, side_3]).squeeze()

            # Calculate length ratio (shortest/longest)
            R = side_3 / side_2

            # Calculate cosine at vertex 1
            C = (
                (p3[0] - p1[0]) * (p2[0] - p1[0]) + (p3[1] - p1[1]) * (p2[1] - p1[1])
            ) / (side_3 * side_2)

            # Calculate orientation (CCW = +1, CW = -1)
            # TODO validate if this is correct
            cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (
                p3[0] - p1[0]
            )
            orientation = 1 if cross > 0 else -1

            # Log of perimiter
            log_p = np.log(np.sum(sides))

            ## Compute the tolerences
            # F
            F = epsilon**2 * (1 / side_3**2 - C / (side_3 * side_2) + 1 / side_2**2)
            # S is the sine of the angle at vertex 1
            S = np.sin(np.arccos(C))
            # t^2_R
            t2R = 2 * R**2 * F
            t2C = 2 * S**2 * F + 3 * C**2 * F**2

            triangle = Triangle(
                vertices=indices,
                sides=sides,
                R=R,
                C=C,
                orientation=orientation,
                t2R=t2R,
                t2C=t2C,
                log_p=log_p,
            )
            self.triangles.append(triangle)

    def plot(self):
        plt.scatter(self.spots[:, 0], self.spots[:, 1])
        plt.show()

        R_values = [t.R for t in self.triangles]
        plt.hist(R_values)
        plt.title("R values distribution")
        plt.show()

        C_values = [t.C for t in self.triangles]
        plt.hist(C_values)
        plt.title("C values distribution")
        plt.show()


class Matcher:
    def compare_patterns(self, pattern_1: SpotPattern, pattern_2: SpotPattern):
        comparisons = [
            [t1.score_match(t2) for t2 in pattern_2.triangles]
            for t1 in pattern_1.triangles
        ]
        # The i dimension should be pattern1, the j should be pattern2
        # The third dimension has length 2. If the triangles match, the value  will be the matching
        # score. Else, it will be the fill value of 100
        comparisons = np.array(comparisons)

        matches = []
        # Iterate over the first dimension
        for index_1 in range(comparisons.shape[0]):
            index_2_match = int(np.argmin(comparisons[index_1, :]))
            # This indicates it was not a background value
            if comparisons[index_1, index_2_match] < 100:
                matches.append((index_1, index_2_match))
        print(matches)
        print(len(matches))
