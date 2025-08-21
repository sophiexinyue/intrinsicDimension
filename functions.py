import math
import random 
from tqdm import tqdm
from scipy.spatial.distance import pdist
from scipy.spatial import KDTree
import numpy as np
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from itertools import product



def bottleneck_experiment(full_data, bottleneck_sizes,batch_size = 256, epochs = 3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ---- Dataset ---- #
    train_size = int(0.9 * len(full_data))
    val_size = len(full_data) - train_size
    train_data, val_data = random_split(full_data, [train_size, val_size])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    # ---- Autoencoder Factory ---- #
    def make_autoencoder(bottleneck_dim):
        class Autoencoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(28*28, 128),
                    nn.ReLU(),
                    nn.Linear(128, bottleneck_dim),
                    nn.ReLU()
                )
                self.decoder = nn.Sequential(
                    nn.Linear(bottleneck_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, 28*28),
                    nn.Sigmoid()
                )
            def forward(self, x):
                x = self.encoder(x)
                x = self.decoder(x)
                return x
        return Autoencoder().to(device)

    # ---- Train Function ---- #
    def train(model):
        model.train()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(epochs):
            for imgs, _ in train_loader:
                imgs = imgs.view(imgs.size(0), -1).to(device)
                outputs = model(imgs)
                loss = criterion(outputs, imgs)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    # ---- Evaluate Function ---- #
    def evaluate(model):
        model.eval()
        criterion = nn.MSELoss()
        total_loss = 0
        with torch.no_grad():
            for imgs, _ in val_loader:
                imgs = imgs.view(imgs.size(0), -1).to(device)
                outputs = model(imgs)
                loss = criterion(outputs, imgs)
                total_loss += loss.item() * imgs.size(0)
        return total_loss / len(val_loader.dataset)

    # ---- Run Experiments ---- #
    results = []
    for bottleneck in bottleneck_sizes:
        print(f"Training AE with bottleneck size {bottleneck}...")
        model = make_autoencoder(bottleneck)
        train(model)
        val_loss = evaluate(model)
        print(f" → Validation MSE: {val_loss:.6f}")
        results.append((bottleneck, val_loss))

    # ---- Plot Results ---- #
    b_sizes, errors = zip(*results)
    plt.figure(figsize=(8, 5))
    plt.plot(b_sizes, errors, marker='o')
    plt.xlabel("Bottleneck Size")
    plt.ylabel("Validation Reconstruction Error (MSE)")
    plt.title("Bottleneck Size vs Reconstruction Error")
    plt.grid(True)
    plt.xscale("log", base=2)  # optional: log-scale for clarity
    plt.show()

    return b_sizes, errors

from sklearn.neighbors import radius_neighbors_graph



def random_orthogonal_matrix(D: int) -> np.ndarray:
    """
    Generate a random D x D orthogonal (unitary in R^D) matrix
    using QR decomposition of a Gaussian random matrix.
    """
    A = np.random.randn(D, D)
    Q, R = np.linalg.qr(A)
    # Ensure determinant = +1 (proper rotation, not reflection)
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q

def grid_dataset(m, d, D, rotate=True):
    """
    Generate an m^d grid embedded in R^D, optionally rotated by a random unitary.
    
    Parameters
    ----------
    m : int
        Number of grid points per axis (resolution).
    d : int
        Intrinsic dimension of the grid (number of varying coordinates).
    D : int
        Ambient dimension (>= d).
    rotate : bool, default=True
        If True, apply a random orthogonal transformation in R^D.
    
    Returns
    -------
    X : ndarray of shape (m**d, D)
        The dataset of grid points.
    """
    if D < d:
        raise ValueError("Ambient dimension D must be at least as large as intrinsic dimension d")
    
    # Construct d-dimensional grid in [0,1]^d
    grid_axes = [np.linspace(0, 1, m) for _ in range(d)]
    grid_points = np.array(list(product(*grid_axes)))  # shape (m^d, d)
    
    # Embed in ambient space (pad with zeros)
    X = np.zeros((m**d, D))
    X[:, :d] = grid_points
    
    # Apply random orthogonal transformation if requested
    if rotate:
        Q = random_orthogonal_matrix(D)
        X = X @ Q  # rotate
    
    return X




def correlation_dim_fixed_r2(r,X):
    n = len(X)
    val = np.sum(radius_neighbors_graph(X, r, mode='connectivity', include_self=False).toarray())  / (n*(n-1))
    return math.log(val) / math.log(r) if val > 0 else 0


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


def correlation_dim2(X, num_trials = 20):
    
    distances = np.random.choice(pdist(X), size=num_trials, replace=False)
    x = [math.log(r) for r in distances]
    y = [correlation_dim_fixed_r2(r, X) for r in distances] 
    
    m, b = np.linalg.lstsq(X, y, rcond=None)[0]
    return m


def correlation_dim(X, num_trials=10, sample_size=100):
    interpoint_distances = pdist(X)
    dims = []

    for _ in range(num_trials):
        r = random.choice(interpoint_distances)
        #val = correlation_dim_fixed_r(r, X, sample_size=sample_size)
        val = correlation_dim_fixed_r2(r, X)
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


from sklearn.metrics import pairwise_distances
from scipy.sparse.csgraph import minimum_spanning_tree

def _mst_alpha_sum(X, alpha=0.5, metric="euclidean"):
    """
    Compute E^0_alpha(X) = sum_{e in MST(X)} |e|^alpha.
    """
    # Full pairwise distances (O(n^2) memory) — fine for moderate n
    D = pairwise_distances(X, metric=metric)
    # SciPy expects a CSR/array; returns a sparse upper-triangular MST
    T = minimum_spanning_tree(D)
    # Edge lengths are in the data of the sparse matrix
    edges = np.asarray(T.data).ravel()
    return np.sum(np.power(edges, alpha))

def _ols_slope_with_se(x, y):
    """
    Return slope, intercept, slope standard error for simple OLS.
    """
    x = np.asarray(x); y = np.asarray(y)
    xbar = x.mean(); ybar = y.mean()
    Sxx = np.sum((x - xbar)**2)
    Sxy = np.sum((x - xbar)*(y - ybar))
    slope = Sxy / Sxx
    intercept = ybar - slope * xbar
    # residual variance and slope SE
    yhat = intercept + slope * x
    rss = np.sum((y - yhat)**2)
    dof = max(len(x) - 2, 1)
    s2 = rss / dof
    se_slope = np.sqrt(s2 / Sxx)
    return slope, intercept, se_slope, rss

def estimate_id_mst(
    X,
    alpha=0.5,
    n_subsamples=1,
    min_frac=0.2,
    max_frac=1.0,
    n_repeats=1,
    metric="euclidean",
    random_state=None,
):
    """
    Persistent-homology (PH0) / MST intrinsic dimension estimator.
    
    Parameters
    ----------
    X : array-like, shape (N, D)
        Data points (rows are samples).
    alpha : float in (0, d)
        Exponent in E^0_alpha. Common choices: 0.5, 1.0.
    n_subsamples : int
        Number of distinct sizes between min_frac*N and max_frac*N to evaluate.
    min_frac, max_frac : float in (0,1]
        Smallest/largest fraction of the dataset to use per subsample size.
    n_repeats : int
        Repetitions per size (averages the E_alpha values for stability).
    metric : str
        Distance metric for pairwise_distances (e.g., 'euclidean', 'minkowski', 'cosine').
    random_state : int or None
        RNG seed.

    Returns
    -------
    result : dict with keys
        'd_hat' : estimated intrinsic dimension
        'slope' : fitted slope in log-log
        'slope_se' : standard error of slope
        'alpha' : alpha used
        'sizes' : list of subsample sizes
        'E_alpha' : averaged E_alpha per size
        'log_sizes' : log sizes used in regression
        'log_E' : log E_alpha used in regression
        'rss' : residual sum of squares of the regression
    """
    rng = np.random.default_rng(random_state)
    X = np.asarray(X)
    N = X.shape[0]
    min_n = max(3, int(np.ceil(min_frac * N)))
    max_n = max(3, int(np.floor(max_frac * N)))
    if max_n <= min_n:
        raise ValueError("Choose min_frac and max_frac so that max_n > min_n.")

    sizes = np.unique(np.linspace(min_n, max_n, n_subsamples, dtype=int)).tolist()
    E_alpha_vals = []

    for n in sizes:
        vals = []
        for _ in range(n_repeats):
            idx = rng.choice(N, size=n, replace=False)
            Ea = _mst_alpha_sum(X[idx], alpha=alpha, metric=metric)
            vals.append(Ea)
        E_alpha_vals.append(np.mean(vals))

    log_n = np.log(np.array(sizes, dtype=float))
    log_E = np.log(np.array(E_alpha_vals, dtype=float))

    slope, intercept, slope_se, rss = _ols_slope_with_se(log_n, log_E)
    # From log E ≈ ((d - α)/d) log n + log C => slope = 1 - α/d  => d = α / (1 - slope)
    if np.isclose(1.0 - slope, 0.0):
        d_hat = np.inf
    else:
        d_hat = alpha / (1.0 - slope)

    return {
        "d_hat": float(d_hat),
        "slope": float(slope),
        "slope_se": float(slope_se),
        "alpha": float(alpha),
        "sizes": sizes,
        "E_alpha": E_alpha_vals,
        "log_sizes": log_n.tolist(),
        "log_E": log_E.tolist(),
        "rss": float(rss),
        "intercept": float(intercept),
    }