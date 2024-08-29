# Annoy (Approximate Nearest Neighbors Oh Yeah)
# Description: Annoy is a library developed by Spotify for efficient approximate nearest neighbor search in high-dimensional spaces. 
# It works well for building recommendation systems and other tasks requiring fast retrieval.

from annoy import AnnoyIndex
import random

# Create an Annoy index
f = 40  # dimension of vectors
t = AnnoyIndex(f, 'angular')  # 'angular', 'euclidean', 'manhattan', etc.

# Add items
for i in range(1000):
    v = [random.gauss(0, 1) for z in range(f)]
    t.add_item(i, v)

t.build(10)  # 10 trees
t.save('test.ann')

# Load index and perform search
u = AnnoyIndex(f, 'angular')
u.load('test.ann')
print(u.get_nns_by_item(0, 5))  # Top 5 nearest neighbors
