def correlation_dim(n, r, S_n):
    """
    Implementation of equation (1) from Levine and Bickel (2001) 
    for estimating the correlation dimension of a set of points in a metric space.
    
    Parameters
    ----------
    n : int
        Number of points in the set.
    r : float
        Radius for the neighborhood around each point.
    S_n : list of tuples
        Set of points in the metric space, where each point is represented as a tuple of coordinates. 

    Returns
    -------
    float
    C_n(r): where plotting log C_n(r) against log r and estimating the slope gives the correlation dimension.
    """
    # Validate inputs
    if n <=0 or r <= 0 or not S_n:
        raise ValueError("n must be positive, r must be positive, and S_n must not be empty.")
    if len(S_n) != n:
        raise ValueError("Length of S_n must match n.")
    if not all(isinstance(point, tuple) and len(point) == len(S_n[0]) for point in S_n):
        raise ValueError("All points in S_n must be tuples of the same length.")
    
    # Calculate the correlation dimension
    count = 0
    for i in range(n): #sum from i=1 to n
        for j in range(i + 1, n): #sum from j=i+1 to n
            # Calculate the Euclidean distance between points S_n[i] and S_n[j]
            distance = sum((S_n[i][k] - S_n[j][k]) ** 2 for k in range(len(S_n[0]))) ** 0.5
            if distance < r:
                count += 1   
    return count * (2 / (n * (n-1)))

if __name__ == "__main__":
    print(correlation_dim(2, 1, [(0, 0), (0, 1)]))  # Example usage
    print(correlation_dim(3, (2 ** 0.5), [(0, 0), (0, 1), (1, 0)]))  # Example usage
    print(correlation_dim(4, (2 ** 0.5), [(0, 0), (0, 1), (1, 0), (1, 1)]))  # Example usage