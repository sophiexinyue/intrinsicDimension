from functions import *
import scanpy as sc
import numpy as np
from IPython import embed
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the PBMC3K dataset (Peripheral Blood Mononuclear Cells, ~3k cells)
#adata = sc.datasets.pbmc3k().X.toarray()

#print(correlation_dim(len(adata), 5, adata.tolist(), 2000))
#print(doubling_dim(adata.tolist(), 5, 10))




"""
# synthetic datasets
synth_data = synthetic_subspace(D=20,d=3,n=1000)

pca = PCA()
pca.fit(synth_data)

plt.figure(figsize=(8, 4))
plt.plot(np.arange(1, len(pca.explained_variance_)+1), pca.explained_variance_, 'o-')
plt.xlabel('Principal component')
plt.ylabel('Eigenvalue (explained variance)')
plt.title('PCA Spectrum (Scree Plot)')
plt.grid(True)
plt.savefig("pca_scree_plot.png")
plt.show()
"""

print( doubling_dim(synth_data)  )



#print(correlation_dim(synth_data.tolist(), num_trials = 10))
#print(doubling_dim(synth_data.tolist(), num_trials = 10))