import numpy as np
import matplotlib.pyplot as plt


def load_pts_features(path):
    """ Load interest points and SIFT features.

    Args:
        path: path to the file pts_feats.npz
    
    Returns:
        pts: coordinate points for two images;
             an array (2,) of numpy arrays (N1, 2), (N2, 2)
        feats: SIFT descriptors for two images;
               an array (2,) of numpy arrays (N1, 128), (N2, 128)
    """
    # Load data
    data = np.load(path, allow_pickle=True)

    # Extract interest points and SIFT features
    interest_points = data['pts']
    features = data['feats']

    # Separate coordinate points and SIFT descriptors for two images
    pts = [interest_points[0], interest_points[1]]
    feats = [features[0], features[1]]

    return pts, feats


def min_num_pairs():
    return 4


def pickup_samples(pts1, pts2):
    """ Randomly select k corresponding point pairs.
    Note that here we assume that pts1 and pts2 have 
    been already aligned: pts1[k] corresponds to pts2[k].

    This function makes use of min_num_pairs()

    Args:
        pts1 and pts2: point coordinates from Image 1 and Image 2
    
    Returns:
        pts1_sub and pts2_sub: N_min randomly selected points 
                               from pts1 and pts2
    """
    # Determine the minimum number of point pairs needed
    N_min = min_num_pairs()

    # Randomly select N_min corresponding point pairs
    index = np.random.choice(len(pts1), size=N_min, replace=False)

    # Extract selected points from pts1 and pts2
    pts1_sub = pts1[index]
    pts2_sub = pts2[index]

    return pts1_sub, pts2_sub


def compute_homography(pts1, pts2):
    """ Construct homography matrix and solve it by SVD

    Args:
        pts1: the coordinates of interest points in img1, array (N, 2)
        pts2: the coordinates of interest points in img2, array (M, 2)
    
    Returns:
        H: homography matrix as array (3, 3)
    """
    assert len(pts1) == len(pts2)

    # Construct the homogeneous linear equation system
    A = []
    for i in range(pts1.shape[0]):
        A += [[0, 0, 0, pts1[i, 0], pts1[i, 1], 1, -pts1[i, 0] * pts2[i, 1], -pts1[i, 1] * pts2[i, 1], -pts2[i, 1]],
              [-pts1[i, 0], -pts1[i, 1], -1, 0, 0, 0, pts1[i, 0] * pts2[i, 0], pts1[i, 1] * pts2[i, 0], pts2[i, 0]]]

    A = np.array(A)

    # Solve the homogeneous linear equation system using SVD
    _, _, vh = np.linalg.svd(A)
    H = vh[-1, :].reshape((3, 3))

    return H


def transform_pts(pts, H):
    """ Transform pst1 through the homography matrix to compare pts2 to find inliners

    Args:
        pts: interest points in img1, array (N, 2)
        H: homography matrix as array (3, 3)
    
    Returns:
        transformed points, array (N, 2)
    """
    # Homogeneous coordinates (add a column of ones)
    p = np.hstack((pts, np.ones((pts.shape[0], 1))))

    # Transform the points
    pointsT = H @ p.T

    # Normalization
    pointsT = pointsT / pointsT[-1]

    return pointsT[:2, :].T


def count_inliers(H, pts1, pts2, threshold=5):
    """ Count inliers
        Tips: We provide the default threshold value, but you're free to test other values
    Args:
        H: homography matrix as array (3, 3)
        pts1: interest points in img1, array (N, 2)
        pts2: interest points in img2, array (N, 2)
        threshold: scale down threshold
    
    Returns:
        number of inliers
    """
    # Transformation
    pts1_H = transform_pts(pts1, H)

    # Calculate the L2 distance
    dist = np.linalg.norm(pts1_H - pts2, axis=1)

    # Count the number of inliers based on the threshold
    n_inliers = len(dist[dist < threshold])

    return n_inliers


def ransac_iters(w=0.5, d=min_num_pairs(), z=0.99):
    """ Computes the required number of iterations for RANSAC.

    Args:
        w: probability that any given correspondence is valid
        d: minimum number of pairs
        z: total probability of success after all iterations
    
    Returns:
        minimum number of required iterations
    """
    nume = np.log(1 - z)  # numerator
    deno = np.log(1 - w ** d)  # denominator
    return int(nume / deno) + 1  # return the ceiling


def ransac(pts1, pts2):
    """ RANSAC algorithm

    Args:
        pts1: matched points in img1, array (N, 2)
        pts2: matched points in img2, array (N, 2)
    
    Returns:
        best homography observed during RANSAC, array (3, 3)
    """
    # init
    best_H = np.empty((3, 3))
    max_inliers = 0
    max_iters = ransac_iters()
    assert max_iters <= len(pts1)

    # RANSAC algorithm
    for i in range(max_iters):
        print("iters", i, " of ", max_iters)
        s1, s2 = pickup_samples(pts1, pts2)  # random select 4 pairs
        H = compute_homography(s1, s2)
        n_inliers = count_inliers(H, pts1, pts2)
        if n_inliers > max_inliers:
            best_H = H
            max_inliers = n_inliers

    return best_H


def find_matches(feats1, feats2, rT=0.8):
    """ Find pairs of corresponding interest points with distance comparsion
        Tips: We provide the default ratio value, but you're free to test other values

    Args:
        feats1: SIFT descriptors of interest points in img1, array (N, 128)
        feats2: SIFT descriptors of interest points in img1, array (M, 128)
        rT: Ratio of similar distances
    
    Returns:
        idx1: list of indices of matching points in img1
        idx2: list of indices of matching points in img2
    """
    idx1 = []
    idx2 = []

    for i, feat1 in enumerate(feats1):
        # Calculate L2 distances between feat1 and all feats2
        dist = np.linalg.norm(feat1 - feats2, axis=1)

        # Find the indices of the two closest features
        closest_indices = np.argsort(dist)[:2]

        # Calculate the ratio of distances
        ratio = dist[closest_indices[0]] / dist[closest_indices[1]]

        # Check if the ratio is below the threshold
        if ratio < rT:
            idx1.append(i)
            idx2.append(closest_indices[0])

    return idx1, idx2


def final_homography(pts1, pts2, feats1, feats2):
    """ re-estimate the homography based on all inliers

    Args:
       pts1: the coordinates of interest points in img1, array (N, 2)
       pts2: the coordinates of interest points in img2, array (M, 2)
       feats1: SIFT descriptors of interest points in img1, array (N, 128)
       feats2: SIFT descriptors of interest points in img1, array (M, 128)
    
    Returns:
        ransac_return: refitted homography matrix from ransac fucation, array (3, 3)
        idxs1: list of matched points in image 1
        idxs2: list of matched points in image 2
    """

    # Find initial matches using SIFT descriptors
    idxs1, idxs2 = find_matches(feats1, feats2)

    # Extract corresponding points from the initial matches
    initial_pts1 = pts1[idxs1]
    initial_pts2 = pts2[idxs2]

    # Run RANSAC to get the best homography
    ransac_return = ransac(initial_pts1, initial_pts2)

    return ransac_return, idxs1, idxs2
