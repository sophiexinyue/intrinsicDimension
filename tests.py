from functions import *
import scanpy as sc
import numpy as np
from IPython import embed

"""# Load the PBMC3K dataset (Peripheral Blood Mononuclear Cells, ~3k cells)
adata = sc.datasets.pbmc3k().X.toarray()

print(correlation_dim(len(adata), 5, adata.tolist(), 2000))
print(doubling_dim(adata.tolist(), 5, 10))
"""

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

# synthetic datasets
synth_data = synthetic_subspace(20,3,1000)

print(correlation_dim(synth_data.tolist(), num_trials = 10))
print(doubling_dim(synth_data.tolist(), num_trials = 10))