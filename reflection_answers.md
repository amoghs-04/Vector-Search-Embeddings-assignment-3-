Q1: What is the difference between IndexFlatL2 and IndexFlatIP in FAISS? When would you use each?

FAISS provides different similarity metrics for comparing vectors.

IndexFlatL2 uses Euclidean Distance (L2 Distance) to measure similarity. Smaller distances indicate vectors are more similar.
IndexFlatIP uses Inner Product (Dot Product) to measure similarity. Larger values indicate greater similarity.

When embeddings are normalized, IndexFlatIP effectively behaves like cosine similarity because the vector magnitudes become 1.

Q2: Why do we normalize embeddings before adding them to FAISS when we want cosine similarity?
Cosine similarity measures the angle between vectors rather than their magnitude.

By applying:

faiss.normalize_L2(embeddings)

all vectors are converted to unit vectors with length equal to 1.

This removes the effect of vector magnitude and allows similarity calculations to depend only on vector direction, which is the core idea behind cosine similarity.

As a result, semantically similar sentences produce vectors that point in similar directions.

Q3: FAISS uses ANN (Approximate Nearest Neighbour) search. What does "approximate" mean here and why is it acceptable?

Approximate Nearest Neighbour (ANN) search does not always guarantee finding the exact closest vector.

Instead, it finds vectors that are extremely close to the best match while significantly reducing computation time.

This trade-off is acceptable because:

Modern vector databases may contain millions or billions of vectors.
Exact nearest-neighbour search becomes computationally expensive at large scale.
ANN provides much faster retrieval with very small accuracy loss.

Because of this speed-accuracy balance, ANN is widely used in production semantic search and RAG systems.
