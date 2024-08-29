# ScaNN (Scalable Nearest Neighbors)

import scann
import numpy as np

# Create a set of random vectors
vectors = np.random.randn(1000, 128).astype(np.float32)

# Create and configure ScaNN searcher
searcher = scann.scann_ops_pybind.builder(vectors, 10, "dot_product").tree(
    num_leaves=100, num_leaves_to_search=10, training_sample_size=2500).score_ah(
    2, anisotropic_quantization_threshold=0.2).build()

# Perform search
query_vector = np.random.randn(128).astype(np.float32)
neighbors, distances = searcher.search(query_vector)
