import math
import random 
from tqdm import tqdm
from scipy.spatial.distance import pdist
from scipy.spatial import KDTree
import numpy as np
from sklearn.decomposition import PCA

def correlation_dim_fixed_r(r, X, sample_size):
    """
    Implementation of equation (1) from Levine and Bickel (2001) 
    for estimating the correlation dimension of a set of points in a metric space.
    """
    # Ensure X is a list of tuples
    if isinstance(X, np.ndarray):
        X = [tuple(p) for p in X]
    n = len(X)
    sample_size = min(n, sample_size)
    
    count = 0
    for _ in range(sample_size):
        i, j = random.sample(range(n), 2)
        distance = euclidean(X[i], X[j])
        if distance < r:
            count += 1   
    C_n_r = count / sample_size

    # Only compute log if C_n_r > 0 and r > 0
    if C_n_r > 0 and r > 0:
        return math.log(C_n_r) / math.log(r)
    else:
        # Return np.nan so you can filter out invalid trials later
        return np.nan

def correlation_dim(X, num_trials=10, sample_size=100):
    interpoint_distances = pdist(X)
    dims = []
    for _ in range(num_trials):
        r = random.choice(interpoint_distances)
        val = correlation_dim_fixed_r(r, X, sample_size=sample_size)
        if not np.isnan(val):
            dims.append(val)
    if dims:
        return np.mean(dims)
    else:
        return np.nan 

def euclidean(p1, p2):
    """Calculate the Euclidean distance between two points.
    Parameters
    ----------
    p1 : tuple
        First point in the metric space, represented as a tuple of coordinates.
    p2 : tuple
        Second point in the metric space, represented as a tuple of coordinates.
    Returns
    -------
    float
    The Euclidean distance between the two points.
    """

    return sum((a - b) ** 2 for a, b in zip(p1, p2)) ** 0.5

def count_covers(points, r):
    """
    Greedy approximation to the minimum r-cover using KD-Tree.
    At each step, pick the point that covers the most uncovered points.
    
    Parameters
    ----------
    points : list of tuples
        Set of points in the metric space.
    r : float
        Radius for the neighborhood around each point.
    
    Returns
    -------
    int
        Number of r-balls needed to cover all points.
    """
    tree = KDTree(points)
    n = len(points)
    all_indices = set(range(n))
    uncovered = set(all_indices)
    centers = []

    while uncovered:
        best_center = None
        best_covered_set = set()

        for idx in uncovered:
            neighbors = tree.query_ball_point(points[idx], r) # get indices of points within radius r using KD-Tree
            covered = uncovered.intersection(neighbors) #filter the neighbors list to only include points that are still uncovered

            if len(covered) > len(best_covered_set):
                best_center = idx
                best_covered_set = covered

        centers.append(points[best_center])
        uncovered -= best_covered_set

    return len(centers)

# n points in a d-dimensional subspace of R^D
def synthetic_subspace(D,d,n,signal_scale=1,noise_scale=0):

    # random unitary matrix
    assert D >= d, "Ambient dimension D must be >= subspace dimension d"
    
    # Random orthonormal basis for a d-dimensional subspace of R^D
    random_matrix = np.random.randn(D,d)
    Q, _ = np.linalg.qr(random_matrix)  # Q has shape (D, d)

    # Sample n points in R^d
    points_subspace = np.random.randn(n,d) * signal_scale

    # Map points into R^D
    X = points_subspace @ Q.T  # shape (n, D)

    X += np.random.randn(n,D) * noise_scale

    return X

def doubling_dim_fixed_r(X, r, sample_size):
    """ Approximator for the doubling dimension of a metric space.

    Parameters
    ----------
    X : list of tuples
        Set of points in the metric space, where each point is represented as a tuple of coordinates.
    r : float
        Radius for the neighborhood around each point.
    sample_size : int
        Number of random centers to sample from the set of points.

    Returns
    -------
    float
    log_2(M): The doubling dimension of the metric space.
    """
    ### create an approximator for doubling dimension
    ### link: https://en.wikipedia.org/wiki/Doubling_space
    X = [tuple(p) for p in X]
    M_vals = []
    for i in tqdm(range(sample_size)):
        x = random.choice(X)
        ball = [y for y in X if euclidean(x, y) < r]
        M = count_covers(ball, r / 2)
        M_vals.append(M)
    
    avg_M = sum(M_vals) / len(M_vals)
    return math.log2(avg_M)

def doubling_dim(X, num_trials=100, sample_size=100):
    # randomly sampled interpoint distances to check doubling dim
    # could also try the 25th, 50th, and 75th percentile interpoint distances , etc
    interpoint_distances = pdist(X)
    dim = 0
    for i in range(num_trials):
        r = random.choice(interpoint_distances)
        dim += doubling_dim_fixed_r(X, r, sample_size)
    return dim / num_trials

def pca_elbow_estimate(X):
    """
    Estimate intrinsic dimension using the elbow method on PCA spectrum.

    Parameters:
        X (ndarray): (n_samples, n_features)

    Returns:
        int: estimated number of principal components
    """
    pca = PCA()
    pca.fit(X)
    eigvals = pca.explained_variance_

    # Coordinates of all points
    n = len(eigvals)
    points = np.column_stack((np.arange(n), eigvals))

    # Line from first to last point
    start, end = points[0], points[-1]
    line_vec = end - start
    line_vec = line_vec / np.linalg.norm(line_vec)

    # Compute distance from each point to the line
    vec_from_start = points - start
    proj_lengths = np.dot(vec_from_start, line_vec)
    proj_points = np.outer(proj_lengths, line_vec) + start
    distances = np.linalg.norm(points - proj_points, axis=1)

    elbow_index = np.argmax(distances)
    return elbow_index + 1  # add 1 to make it 1-based index
