from functions import *
import scanpy as sc
from IPython import embed

# Load the PBMC3K dataset (Peripheral Blood Mononuclear Cells, ~3k cells)
adata = sc.datasets.pbmc3k().X.toarray()

print(correlation_dim(len(adata), 5, adata.tolist(), 2000))
print(doubling_dim(adata.tolist(), 5, 10))