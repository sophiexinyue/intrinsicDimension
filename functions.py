import math
import random 
from tqdm import tqdm
from scipy.spatial.distance import pdist


def correlation_dim_fixed_r(r, X, sample_size):
    """
    Implementation of equation (1) from Levine and Bickel (2001) 
    for estimating the correlation dimension of a set of points in a metric space.
    
    Parameters
    ----------
    n : int
        Number of points in the set.
    r : float
        Radius for the neighborhood around each point.
    X : list of tuples
        Set of points in the metric space, where each point is represented as a tuple of coordinates. 

    Returns
    -------
    float
    correlation dimension
    """
    n = len(X)
    sample_size = min([n,sample_size])
    
    # Calculate the correlation dimension
    count = 0
    for p in tqdm(range(sample_size)):
        i, j = random.sample(range(n), 2)
        distance = sum((X[i][k] - X[j][k]) ** 2 for k in range(len(X[0]))) ** 0.5
        if distance < r:
            count += 1   
    C_n_r = count / sample_size
    print("C_n_r:", C_n_r)

    # return the correlation dimension
    if C_n_r > 0:
        return math.log(C_n_r) / math.log(r)
    else:
        raise ValueError("C_n(r) is zero or negative, cannot compute logarithm.")

def correlation_dim(X, num_trials=10, sample_size=100):
    # try the 25th, 50th, and 75th percentile interpoint distances 
    interpoint_distances = pdist(X)
    dim = 0
    for i in range(num_trials):
        r = random.choice(interpoint_distances)
        dim += correlation_dim_fixed_r(r,X, sample_size=sample_size)
    return dim / num_trials

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
    # This function runs in O(n^3) time, which is not ideal... maybe we can optimize it later? 
    """
    Greedy approximation to the minimum r-cover.
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
    uncovered = set(points)
    centers = []

    while uncovered:
        best_center = None # best_center is the point that currently covers the most uncovered points.
        max_covered = 0 # max_covered is the number of uncovered points that best_center can cover.
        best_covered_set = set() # best_covered_set is the actual set of points that best_center covers.

        for p in uncovered:
            covered = {q for q in uncovered if euclidean(p, q) <= r}
            if len(covered) > max_covered:
                best_center = p
                best_covered_set = covered
                max_covered = len(covered)

        centers.append(best_center)
        uncovered -= best_covered_set

    return len(centers)


def doubling_dim(X, num_trials=100, sample_size=100):
    # randomly sampled interpoint distances to check doubling dim
    # could also try the 25th, 50th, and 75th percentile interpoint distances , etc
    interpoint_distances = pdist(X)
    dim = 0
    for i in range(num_trials):
        r = random.choice(interpoint_distances)
        dim += doubling_dim_fixed_r(r,X, sample_size=sample_size)
    return dim / num_trials




def doubling_dim_fixed_r(X, r, num_centers):
    """ Approximator for the doubling dimension of a metric space.

    Parameters
    ----------
    X : list of tuples
        Set of points in the metric space, where each point is represented as a tuple of coordinates.
    r : float
        Radius for the neighborhood around each point.
    num_centers : int
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
    for i in tqdm(range(num_centers)):
        x = random.choice(X)
        ball = [y for y in X if euclidean(x, y) < r]
        M = count_covers(ball, r / 2)
        M_vals.append(M)
    
    avg_M = sum(M_vals) / len(M_vals)
    return math.log2(avg_M)
