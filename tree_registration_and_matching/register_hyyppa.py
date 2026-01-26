import numpy as np
from scipy.optimize import fmin
import warnings
import geopandas as gpd
import matplotlib.pyplot as plt

from tree_registration_and_matching.utils import ensure_projected_CRS


def fit_euclidean_transformation(xy_mat1, xy_mat2, parameters=None):
    """
    This function finds the 2D Euclidean transformation between two point
    clouds using the locations of some distinct objects detected from the point clouds.
    The Euclidean transformation is defined as xy_2 = R_mat*xy_1 + t_vect,
    where xy_1 is a 2-by-1 vector containing the (x,y)-coordinates of an object
    in the first coordinate frame, xy_2 is a 2-by-1 vector containing the
    (x,y)-coordinates of the same object in the second coordinate frame,
    R_mat is a 2-by-2 rotation matrix, and t_vect is a 2-by-1 translation vector.

    The function is based on the algorithm proposed in Hyyppä and Muhojoki et
    al (2021), Efficient coarse registration method using translation- and
    rotation-invariant local descriptors towards fully automated forest
    inventory, ISPRS Open Journal of Photogrammetry and Remote Sensing.

    Args:
        xy_mat1: (x,y)-coordinates of objects (e.g. trees) found from point
                 cloud 1 as a N1-by-2 numpy array.
        xy_mat2: (x,y)-coordinates of objects (e.g. trees) found from point
                 cloud 2 as N2-by-2 numpy array. If one of the datasets has
                 significantly less objects, it should be chosen as dataset 2
                 (i.e., N2 < N1).
        parameters: Dictionary with optional keys:
            'R_local': radius of the local neighborhood used in constructing
                      the feature descriptor of each object. Default is 10.
            'k': number of promising matching pairs to use for finding the
                best Euclidean transformation. Default is 20.
            'r_thres': distance threshold for considering an object pair as a
                      match. Default is 1.0.
            'max_iter': maximum number of iterations in the refinement of
                       the parameter values. Default is 200.

    Returns:
        theta: rotation angle (by which xy_1 should be rotated to obtain xy_2)
        R_mat: 2D rotation matrix corresponding to theta
        t_vect: translation vector as a 2-by-1 column vector
        n_of_matches: number of matching object pairs for the best transformation
        feat_desc_cell_array: list with two elements containing the
                             feature descriptors for all objects as N-by-8 arrays
    """

    # Set default parameter values
    R_local = 10
    k = 20
    r_thres = 1.0
    max_iter = 200

    # Get parameters from input
    if parameters is not None:
        R_local = parameters.get("R_local", R_local)
        k = parameters.get("k", k)
        r_thres = parameters.get("r_thres", r_thres)
        max_iter = parameters.get("max_iter", max_iter)

    # Convert to numpy arrays
    xy_mat1 = np.array(xy_mat1, dtype=float)
    xy_mat2 = np.array(xy_mat2, dtype=float)

    # Centering the coordinates before matching to improve numerical stability
    xy_mat1_mean = np.mean(xy_mat1, axis=0)  # 1-by-2
    xy_mat2_mean = np.mean(xy_mat2, axis=0)  # 1-by-2
    xy_mat1 = xy_mat1 - xy_mat1_mean
    xy_mat2 = xy_mat2 - xy_mat2_mean

    # Number of objects detected from each point cloud
    N_objects_vect = [len(xy_mat1), len(xy_mat2)]

    if N_objects_vect[1] < k:
        k = N_objects_vect[1]
        msg = (
            f"Number of objects in dataset 2 (and thus the potential number of matches) "
            f"is smaller than given parameter k. k was set to {N_objects_vect[1]}"
        )
        warnings.warn(msg)

    # Construct the feature descriptor for each object in each of the point clouds
    feat_desc_mat1, char_dirs1 = compute_feature_descriptors(xy_mat1, R_local)
    feat_desc_mat2, char_dirs2 = compute_feature_descriptors(xy_mat2, R_local)
    feat_desc_cell_array = [feat_desc_mat1, feat_desc_mat2]

    # Preallocating matrix for storing pairwise distances between feature
    # descriptors of the two point clouds
    feat_desc_pdist = np.zeros((N_objects_vect[1], N_objects_vect[0]))
    for i in range(feat_desc_mat1.shape[1]):
        feat_desc_pdist = (
            feat_desc_pdist
            + (feat_desc_mat2[:, i : i + 1] - feat_desc_mat1[:, i].T) ** 2
        )
    feat_desc_pdist = np.sqrt(feat_desc_pdist)

    # For each object in point cloud 2, find the most similar feature
    # descriptor vector in point cloud 1
    min_desc_dists = np.min(feat_desc_pdist, axis=1)
    nn_indices = np.argmin(feat_desc_pdist, axis=1)

    # For each object in point cloud 2, find the 2nd nearest neighbor feature
    # descriptor in point cloud 1
    feat_desc_pdist_copy = feat_desc_pdist.copy()
    for j in range(N_objects_vect[1]):
        feat_desc_pdist_copy[j, nn_indices[j]] = np.inf
    min2_desc_dists = np.min(feat_desc_pdist_copy, axis=1)

    # Computing the nearest neighbor distance ratio
    NNDR_vect = min_desc_dists / min2_desc_dists

    # Sorting the nearest neighbor distance ratios
    sorted_indices = np.argsort(NNDR_vect)

    # Consider the k most promising tentative matches
    max_n_of_matches = 0
    t_best = np.zeros(2)
    theta_best = 0
    idx_matches1_best = []
    idx_matches2_best = []

    for i_iter in range(k):
        # Indices of objects corresponding to the current tentative match
        idx_object_2 = sorted_indices[i_iter]
        idx_object_1 = nn_indices[idx_object_2]

        # Characteristic directions corresponding to these objects
        char_dir1 = char_dirs1[idx_object_1, :]
        char_dir2 = char_dirs2[idx_object_2, :]

        # Computing the rotation angle based on the characteristic directions
        theta_magn = np.arccos(np.sum(char_dir1 * char_dir2))
        cross_prod = np.cross(np.append(char_dir1, 0), np.append(char_dir2, 0))
        theta_sign = np.sign(np.sum(cross_prod * np.array([0, 0, 1])))
        theta = theta_magn * theta_sign

        # The corresponding rotation matrix
        R_mat = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )

        # Computing the translation vector
        xy_1 = xy_mat1[idx_object_1, :]
        xy_2 = xy_mat2[idx_object_2, :]
        t_vect = xy_2 - R_mat @ xy_1

        # Find closest pairs and compute number of matches
        from_idx2_to_closest_idx1, from_idx2_to_min_dist1, n_of_matches = (
            get_closest_pairs_after_transformation(
                xy_mat1, xy_mat2, R_mat, t_vect, r_thres
            )
        )

        if n_of_matches > max_n_of_matches:
            max_n_of_matches = n_of_matches
            indices2 = np.arange(N_objects_vect[1])
            idx_matches2_best = indices2[from_idx2_to_min_dist1 < r_thres]
            idx_matches1_best = from_idx2_to_closest_idx1[idx_matches2_best]
            t_best = t_vect
            theta_best = theta

    # Refine the transformation using the found matching pairs
    xy_mat1_matching = xy_mat1[idx_matches1_best, :]
    xy_mat2_matching = xy_mat2[idx_matches2_best, :]

    # Function to minimize: beta[0] = theta, beta[1] = t_x, beta[2] = t_y
    def obj_fun(beta):
        R = np.array(
            [[np.cos(beta[0]), -np.sin(beta[0])], [np.sin(beta[0]), np.cos(beta[0])]]
        )
        t = np.array([beta[1], beta[2]])
        transformed = (R @ xy_mat1_matching.T).T + t
        return np.sum((transformed - xy_mat2_matching) ** 2)

    beta0 = np.array([theta_best, t_best[0], t_best[1]])

    # Conducting the optimization
    beta = fmin(obj_fun, beta0, maxiter=max_iter, disp=False)

    # Optimized parameters for the Euclidean transformation
    theta = beta[0]
    R_mat = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    t_vect = np.array([[beta[1]], [beta[2]]])  # column vector

    # Computing the number of matches based on the refined transformation
    _, _, n_of_matches = get_closest_pairs_after_transformation(
        xy_mat1, xy_mat2, R_mat, t_vect.flatten(), r_thres
    )

    # Undoing the centering to get the correct translation vector
    t_vect = (
        t_vect - (R_mat @ xy_mat1_mean.reshape(-1, 1)) + xy_mat2_mean.reshape(-1, 1)
    )

    return theta, R_mat, t_vect, n_of_matches, feat_desc_cell_array


def compute_feature_descriptors(xy_mat, R_local):
    """
    Helper function for constructing the feature descriptor vectors for the
    objects whose locations are listed in xy_mat.

    Args:
        xy_mat: (x,y)-coordinates of objects as a N-by-2 matrix
        R_local: radius of the local neighborhood

    Returns:
        feat_desc_mat: N-by-8 matrix containing the feature descriptor vectors
        char_dirs: N-by-2 matrix containing the characteristic directions
    """

    # Pre-allocating the feature descriptor matrix
    feat_desc_mat = np.zeros((len(xy_mat), 8))
    x_vect = xy_mat[:, 0]  # column vector
    y_vect = xy_mat[:, 1]

    # Pair-wise distances between all objects
    p_dist_mat = np.sqrt(
        (x_vect[:, np.newaxis] - x_vect) ** 2 + (y_vect[:, np.newaxis] - y_vect) ** 2
    )

    # Setting the "self-distance" to infinity
    np.fill_diagonal(p_dist_mat, np.inf)

    # Finding the closest neighboring object for each object
    idx_closest = np.argmin(p_dist_mat, axis=1)

    # Coordinates of the closest neighboring object
    x_closest = x_vect[idx_closest]
    y_closest = y_vect[idx_closest]

    # Compute the normalized characteristic directions
    char_dirs = np.column_stack([x_closest - x_vect, y_closest - y_vect])
    char_dirs = char_dirs / np.sqrt(np.sum(char_dirs**2, axis=1, keepdims=True))

    # Directions perpendicular to the characteristic directions
    perp_char_dirs = np.column_stack([-char_dirs[:, 1], char_dirs[:, 0]])

    # Transform coordinates into local coordinate frame
    v_mat = (x_vect[:, np.newaxis] - x_vect) * char_dirs[:, 0:1] + (
        y_vect[:, np.newaxis] - y_vect
    ) * char_dirs[:, 1:2]
    w_mat = (x_vect[:, np.newaxis] - x_vect) * perp_char_dirs[:, 0:1] + (
        y_vect[:, np.newaxis] - y_vect
    ) * perp_char_dirs[:, 1:2]

    # Compute angles with respect to the characteristic directions
    angle_mat = np.arctan2(w_mat, v_mat)

    eps = 1e-6

    for i_quadrant in range(1, 5):
        if i_quadrant == 1:
            quadrant_i_mat = (angle_mat > 0 + eps) & (angle_mat <= np.pi / 2)
        elif i_quadrant == 2:
            quadrant_i_mat = (angle_mat > np.pi / 2) & (angle_mat <= np.pi)
        elif i_quadrant == 3:
            quadrant_i_mat = (angle_mat > -np.pi) & (angle_mat <= -np.pi / 2)
        else:
            quadrant_i_mat = (angle_mat > -np.pi / 2) & (angle_mat < 0 - eps)

        # Construct a matrix marking objects not in current quadrant with Inf
        not_quadrant_i_mat = np.zeros(quadrant_i_mat.shape)
        not_quadrant_i_mat[~quadrant_i_mat] = np.inf

        # Find the closest object within the current quadrant
        min_dist_i = np.min(p_dist_mat + not_quadrant_i_mat, axis=1)
        idx_closest_i = np.argmin(p_dist_mat + not_quadrant_i_mat, axis=1)

        # Determine the angle of the closest object in the current quadrant
        angle_closest_quad_i = angle_mat[np.arange(len(angle_mat)), idx_closest_i]

        # Normalizing the distances and angles
        min_dist_i_norm = min_dist_i / R_local

        if i_quadrant == 1:
            angle_quad_i_norm = (angle_closest_quad_i - 0) / (np.pi / 2)
        elif i_quadrant == 2:
            angle_quad_i_norm = (angle_closest_quad_i - np.pi / 2) / (np.pi / 2)
        elif i_quadrant == 3:
            angle_quad_i_norm = (angle_closest_quad_i - (-np.pi)) / (np.pi / 2)
        else:
            angle_quad_i_norm = (angle_closest_quad_i - (-np.pi / 2)) / (np.pi / 2)

        # If min distance > R_local, set normalized values to -1
        min_dist_i_norm[min_dist_i > R_local] = -1
        angle_quad_i_norm[min_dist_i > R_local] = -1

        # Store in feature descriptor matrix
        feat_desc_mat[:, i_quadrant - 1] = min_dist_i_norm
        feat_desc_mat[:, i_quadrant + 3] = angle_quad_i_norm

    return feat_desc_mat, char_dirs


def get_closest_pairs_after_transformation(xy_mat1, xy_mat2, R_mat, t_vect, r_thres):
    """
    Helper function for obtaining a mapping from each object in dataset 2 to the
    index of the closest object in dataset 1 after transformation.

    Args:
        xy_mat1: (x,y)-coordinates of objects from point cloud 1 as N1-by-2 matrix
        xy_mat2: (x,y)-coordinates of objects from point cloud 2 as N2-by-2 matrix
        R_mat: 2-by-2 rotation matrix
        t_vect: 2-element array (translation vector)
        r_thres: distance threshold for deciding if closest object is a match

    Returns:
        from_idx2_to_closest_idx1: N2-by-1 array of closest object indices
        from_idx2_to_min_dist1: N2-by-1 array of distances to closest objects
        n_of_matches: number of matching object pairs
    """

    t_vect = np.array(t_vect).flatten()

    # Transform xy_mat1
    xy_mat1_transformed = (R_mat @ xy_mat1.T + t_vect.reshape(-1, 1)).T

    # Pair-wise distances
    transformed_loc_p_dist = np.sqrt(
        (xy_mat2[:, 0:1] - xy_mat1_transformed[:, 0].T) ** 2
        + (xy_mat2[:, 1:2] - xy_mat1_transformed[:, 1].T) ** 2
    )

    # Find closest object in transformed point cloud 1 for each object in point cloud 2
    from_idx2_to_min_dist1 = np.min(transformed_loc_p_dist, axis=1)
    from_idx2_to_closest_idx1 = np.argmin(transformed_loc_p_dist, axis=1)

    # Compute number of matches
    n_of_matches = np.sum(from_idx2_to_min_dist1 < r_thres)

    return from_idx2_to_closest_idx1, from_idx2_to_min_dist1, n_of_matches


def align_plot_hyyppa(field_trees, detected_trees):
    original_field_CRS = field_trees.crs
    # Transform the drone trees to a cartesian CRS if not already
    field_trees = ensure_projected_CRS(field_trees)

    # Ensure that drone trees are in the same CRS
    detected_trees.to_crs(field_trees.crs, inplace=True)

    field_trees_points = np.array(field_trees.geometry.get_coordinates())
    detected_trees_points = np.array(detected_trees.geometry.get_coordinates())

    # Note that the field trees should be second
    theta, R_mat, t_vect, n_of_matches, feat_desc = fit_euclidean_transformation(
        detected_trees_points, field_trees_points
    )

    transform = np.eye(3)
    transform[:2, :2] = R_mat
    transform[:2, 2:] = t_vect

    transform_inv = np.linalg.inv(transform)

    field_trees_shifted = field_trees.copy()
    field_trees_shifted.geometry = field_trees.geometry.affine_transform(
        [
            transform_inv[0, 0],
            transform_inv[0, 1],
            transform_inv[1, 0],
            transform_inv[1, 1],
            transform_inv[0, 2],
            transform_inv[1, 2],
        ]
    )
    f, ax = plt.subplots()

    field_trees.plot(ax=ax, c="b", label="field trees")
    field_trees_shifted.plot(ax=ax, c="r", label="field trees shifted")
    plt.show()

    field_trees_shifted.to_crs(original_field_CRS)
    return field_trees_shifted


# Example usage
if __name__ == "__main__":
    dataset = "0008_000256_000254"
    field_trees = gpd.read_file(
        f"/ofo-share/repos/david/tree-registration-and-matching/data/ofo-example-2/field_trees/{dataset}.gpkg"
    )
    detected_trees = gpd.read_file(
        f"/ofo-share/repos/david/tree-registration-and-matching/data/ofo-example-2/detected-trees/{dataset}.gpkg"
    )
    align_plot_hyyppa(field_trees, detected_trees)

    # Create synthetic test data
    np.random.seed(42)

    # Generate some random points for dataset 1
    xy_mat1 = np.random.rand(50, 2) * 100

    # Create dataset 2 by applying a known transformation to a subset of dataset 1
    true_theta = np.pi / 4  # 45 degrees
    true_R = np.array(
        [
            [np.cos(true_theta), -np.sin(true_theta)],
            [np.sin(true_theta), np.cos(true_theta)],
        ]
    )
    true_t = np.array([20, 30])

    # Select subset and transform
    indices = np.random.choice(len(xy_mat1), size=30, replace=False)
    xy_mat2 = (true_R @ xy_mat1[indices].T).T + true_t

    # Add some noise
    xy_mat2 += np.random.randn(*xy_mat2.shape) * 0.5

    # Add some outliers to both datasets
    xy_mat1 = np.vstack([xy_mat1, np.random.rand(10, 2) * 100])
    xy_mat2 = np.vstack([xy_mat2, np.random.rand(10, 2) * 100])

    # Fit the transformation
    parameters = {"R_local": 15, "k": 25, "r_thres": 2.0, "max_iter": 300}

    theta, R_mat, t_vect, n_of_matches, feat_desc = fit_euclidean_transformation(
        xy_mat1, xy_mat2, parameters
    )

    print("Fitted transformation:")
    print(f"Rotation angle: {np.degrees(theta):.2f} degrees")
    print(f"Rotation matrix:\n{R_mat}")
    print(f"Translation vector:\n{t_vect}")
    print(f"Number of matches: {n_of_matches}")
    print(f"\nTrue rotation angle: {np.degrees(true_theta):.2f} degrees")
    print(f"True translation: {true_t}")
