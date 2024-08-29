# HNSWLIB (Hierarchical Navigable Small World)

import hnswlib
import numpy as np

# Create a set of random vectors
dim = 128
num_elements = 10000
data = np.float32(np.random.random((num_elements, dim)))

# Initialize index
p = hnswlib.Index(space='l2', dim=dim)
p.init_index(max_elements=num_elements, ef_construction=200, M=16)

# Add data to the index
p.add_items(data)

# Perform search
query_vector = np.float32(np.random.random((dim)))
labels, distances = p.knn_query(query_vector, k=5)
