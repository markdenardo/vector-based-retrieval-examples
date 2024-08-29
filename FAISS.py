# FAISS (Facebook AI Similarity Search)
# Description: FAISS is a library developed by Facebook AI Research that provides efficient similarity search and clustering of dense vectors.

import faiss
import numpy as np

# Create a set of random vectors
d = 128  # dimension
nb = 1000  # number of vectors
np.random.seed(1234)
vectors = np.random.random((nb, d)).astype('float32')

# Create index
index = faiss.IndexFlatL2(d)
index.add(vectors)

# Perform search
query_vector = np.random.random((1, d)).astype('float32')
distances, indices = index.search(query_vector, k=5)  # top 5 results
