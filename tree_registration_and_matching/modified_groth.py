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
    length_ratio: float  # Ratio of shortest to longest side
    cosine_v1: float  # Cosine of angle at vertex 1
    orientation: int  # +1 for CCW, -1 for CW


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
        self._build_triangles()

    def _build_triangles(self):
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

            sorting_inds = tuple(np.argsort([side_ab, side_bc, side_ca]).squeeze())

            # Remap according to the "Formation of triangles" section
            i1, i2, i3 = {
                (0, 1, 2): (a, b, c),
                (0, 2, 1): (b, a, c),
                (1, 0, 2): (c, b, a),
                (1, 2, 0): (c, a, b),
                (2, 0, 1): (b, c, a),
                (2, 1, 0): (a, c, b),
            }[sorting_inds]

            # Rerun with the correct order
            p1, p2, p3 = self.spots[i1], self.spots[i2], self.spots[i3]

            # Calculate side lengths
            side_1 = float(np.linalg.norm(p2 - p3))
            side_2 = float(np.linalg.norm(p1 - p2))
            side_3 = float(np.linalg.norm(p1 - p3))

            sorting_inds = np.argsort([side_1, side_2, side_3]).squeeze()
            # print(sorting_inds, sorted([side_1, side_2, side_3]))
            if np.any(sorting_inds != np.array([1, 0, 2])):
                print(sorting_inds)
            continue
            sides = (side_1, side_2, side_3)

            # Skip degenerate triangles
            if np.min(sides) < 1e-6:
                continue

            # Calculate length ratio (shortest/longest)
            length_ratio = np.min(sides) / np.max(sides)

            # Calculate cosine at vertex 1
            # Using law of cosines: cos(A) = (b² + c² - a²) / (2bc)
            cosine_v1 = (side_12**2 + side_31**2 - side_23**2) / (2 * side_12 * side_31)
            cosine_v1 = np.clip(cosine_v1, -1.0, 1.0)  # Handle numerical errors

            # Calculate orientation (CCW = +1, CW = -1)
            cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (
                p3[0] - p1[0]
            )
            orientation = 1 if cross > 0 else -1

            triangle = Triangle(
                vertices=indices,
                sides=sides,
                length_ratio=length_ratio,
                cosine_v1=cosine_v1,
                orientation=orientation,
            )
            self.triangles.append(triangle)

    def plot(self):
        plt.scatter(self.spots[:, 0], self.spots[:, 1])
        plt.show()


class WhaleSharkMatcher:
    """
    Modified Groth algorithm for matching whale shark spot patterns
    """

    def __init__(
        self,
        length_ratio_tolerance: float = 0.15,
        cosine_tolerance: float = 0.15,
        min_triangle_matches: int = 3,
    ):
        """
        Initialize matcher with tolerances

        Args:
            length_ratio_tolerance: Maximum difference in length ratios
            cosine_tolerance: Maximum difference in cosines
            min_triangle_matches: Minimum number of triangle matches required
        """
        self.length_ratio_tolerance = length_ratio_tolerance
        self.cosine_tolerance = cosine_tolerance
        self.min_triangle_matches = min_triangle_matches

    def match_triangles(
        self, pattern1: SpotPattern, pattern2: SpotPattern
    ) -> List[Tuple[Triangle, Triangle]]:
        """
        Find matching triangles between two patterns

        Returns list of (triangle1, triangle2) pairs that match
        """
        matches = []

        for t1 in pattern1.triangles:
            for t2 in pattern2.triangles:
                if self._triangles_match(t1, t2):
                    matches.append((t1, t2))

        return matches

    def _triangles_match(self, t1: Triangle, t2: Triangle) -> bool:
        """
        Check if two triangles match within tolerances
        """
        # Check length ratio
        if abs(t1.length_ratio - t2.length_ratio) > self.length_ratio_tolerance:
            return False

        # Check cosine at vertex 1
        if abs(t1.cosine_v1 - t2.cosine_v1) > self.cosine_tolerance:
            return False

        # Check orientation (must be same)
        if t1.orientation != t2.orientation:
            return False

        return True

    def compute_match_score(
        self, pattern1: SpotPattern, pattern2: SpotPattern
    ) -> float:
        """
        Compute overall match score between two patterns

        Returns:
            Score between 0 and 1, where 1 is perfect match
        """
        triangle_matches = self.match_triangles(pattern1, pattern2)

        # No matches
        if len(triangle_matches) == 0:
            return 0.0

        # Vote for spot correspondences
        spot_votes = self._vote_for_correspondences(triangle_matches)

        # Calculate score based on number of consistent spot matches
        max_possible = min(len(pattern1.spots), len(pattern2.spots))
        n_matched = len(spot_votes)

        # Weight by triangle match quality
        triangle_score = len(triangle_matches) / max(
            len(pattern1.triangles), len(pattern2.triangles)
        )
        spot_score = n_matched / max_possible if max_possible > 0 else 0

        # Combined score (weighted average)
        score = 0.6 * spot_score + 0.4 * triangle_score

        return score

    def _vote_for_correspondences(
        self, triangle_matches: List[Tuple[Triangle, Triangle]]
    ) -> Dict[Tuple[int, int], int]:
        """
        Vote for spot correspondences based on triangle matches

        Returns dictionary of (spot1_idx, spot2_idx) -> vote_count
        """
        votes = {}

        for t1, t2 in triangle_matches:
            # Each triangle proposes 3 spot correspondences
            for i in range(3):
                correspondence = (t1.vertices[i], t2.vertices[i])
                votes[correspondence] = votes.get(correspondence, 0) + 1

        # Filter to keep only highly voted correspondences
        threshold = max(votes.values()) * 0.5 if votes else 0
        return {k: v for k, v in votes.items() if v >= threshold}

    def find_best_match(
        self, query: SpotPattern, library: List[SpotPattern], threshold: float = 0.3
    ) -> List[Tuple[str, float]]:
        """
        Find best matches for query pattern in a library

        Args:
            query: Query spot pattern
            library: List of spot patterns to search
            threshold: Minimum score to consider a match

        Returns:
            List of (pattern_name, score) sorted by score (best first)
        """
        results = []

        for pattern in library:
            score = self.compute_match_score(query, pattern)
            if score >= threshold:
                results.append((pattern.name, score))

        # Sort by score (descending)
        results.sort(key=lambda x: x[1], reverse=True)

        return results
