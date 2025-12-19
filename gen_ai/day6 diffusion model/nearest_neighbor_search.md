# Nearest Neighbor Search (NNS)

## 1. Introduction
Nearest Neighbor Search (NNS) is a fundamental problem in machine learning, data mining, information retrieval, and AI systems. Given a query point, the goal is to find the data point(s) in a dataset that are closest to it according to a defined distance metric.

NNS is widely used in:
- Recommendation systems
- Similarity search (text, images, embeddings)
- Clustering and classification (e.g., k-NN)
- Vector databases and RAG systems

---

## 2. Problem Definition

Given:
- A dataset of points:  
  \( D = \{x_1, x_2, ..., x_n\} \)
- A query point:  
  \( q \)
- A distance function:  
  \( d(x, q) \)

### Objective
Find:
- **1-NN**:  
  \( \arg\min_{x \in D} d(x, q) \)
- **k-NN**: The top *k* closest points to \( q \)

---

## 3. Distance Metrics

Common distance functions:

### 3.1 Euclidean Distance
\[
 d(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}
\]

### 3.2 Manhattan Distance
\[
 d(x, y) = \sum_{i=1}^{n} |x_i - y_i|
\]

### 3.3 Cosine Distance
\[
 d(x, y) = 1 - \frac{x \cdot y}{||x|| ||y||}
\]

### 3.4 Hamming Distance
Used for binary or categorical vectors.

---

## 4. Types of Nearest Neighbor Search

### 4.1 Exact Nearest Neighbor Search
- Guarantees correct neighbors
- Computationally expensive for large datasets
- Typically \( O(n \cdot d) \)

### 4.2 Approximate Nearest Neighbor (ANN)
- Trades accuracy for speed
- Used in large-scale systems
- Backbone of vector databases

---

## 5. Brute Force Nearest Neighbor Search

### Algorithm
1. Compute distance from query to every point
2. Sort distances
3. Select top *k*

### Python Code
```python
import numpy as np

def brute_force_knn(X, query, k=1):
    distances = np.linalg.norm(X - query, axis=1)
    nearest_indices = np.argsort(distances)[:k]
    return nearest_indices, distances[nearest_indices]

# Example
X = np.array([[1, 2], [3, 4], [5, 6]])
query = np.array([2, 3])
indices, dists = brute_force_knn(X, query, k=2)
print(indices, dists)
```

### Pros
- Simple
- Always accurate

### Cons
- Not scalable

---

## 6. k-Nearest Neighbors (k-NN)

k-NN is both a **search method** and a **machine learning algorithm**.

### Steps
1. Choose k
2. Compute distances
3. Select k closest points
4. Aggregate labels (for classification/regression)

### Scikit-learn Example
```python
from sklearn.neighbors import KNeighborsClassifier

X = [[1, 2], [2, 3], [3, 4], [6, 7]]
y = [0, 0, 1, 1]

model = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
model.fit(X, y)

prediction = model.predict([[2, 2]])
print(prediction)
```

---

## 7. Tree-Based Nearest Neighbor Search

### 7.1 KD-Tree
- Binary tree
- Splits data along dimensions
- Efficient for low-dimensional data

### KD-Tree Example
```python
from sklearn.neighbors import KDTree
import numpy as np

X = np.random.random((10, 2))
query = np.array([[0.5, 0.5]])

tree = KDTree(X, leaf_size=2)
dist, ind = tree.query(query, k=3)
print(ind, dist)
```

### 7.2 Ball Tree
- Uses hyperspheres
- Performs better in higher dimensions than KD-tree

```python
from sklearn.neighbors import BallTree

tree = BallTree(X, metric='euclidean')
dist, ind = tree.query(query, k=3)
print(ind)
```

---

## 8. Approximate Nearest Neighbor (ANN)

Used when:
- Dataset size is very large
- Dimensionality is high (embeddings)

### Common ANN Techniques
- Locality Sensitive Hashing (LSH)
- HNSW (Hierarchical Navigable Small World)
- Product Quantization (PQ)

---

## 9. Locality Sensitive Hashing (LSH)

### Idea
Similar points hash to the same bucket with high probability.

### Simple LSH Example
```python
from sklearn.random_projection import GaussianRandomProjection
import numpy as np

X = np.random.random((100, 50))
transformer = GaussianRandomProjection(n_components=10)
X_proj = transformer.fit_transform(X)

query = X[0].reshape(1, -1)
query_proj = transformer.transform(query)

# Compute distance in projected space
from numpy.linalg import norm

distances = norm(X_proj - query_proj, axis=1)
print(np.argsort(distances)[:5])
```

---

## 10. Nearest Neighbor Search in Vector Databases

NNS is the core operation in vector databases used for:
- Semantic search
- RAG pipelines
- Multimodal retrieval

### Typical Workflow
1. Convert data to embeddings
2. Store vectors
3. Query using similarity search

### Example (Conceptual)
```python
query_embedding = embed("What is AI?")
results = vector_db.search(query_embedding, top_k=5)
```

---

## 11. Complexity Analysis

| Method | Time Complexity | Space Complexity |
|------|----------------|------------------|
| Brute Force | O(n·d) | O(1) |
| KD-Tree | O(log n) avg | O(n) |
| Ball Tree | O(log n) avg | O(n) |
| ANN (HNSW) | Sub-linear | O(n) |

---

## 12. Challenges in Nearest Neighbor Search

- Curse of dimensionality
- Trade-off between accuracy and speed
- Memory consumption
- Distance metric selection

---

## 13. Use Cases

- Image similarity search
- Document retrieval
- Recommendation engines
- Fraud detection
- RAG-based LLM systems

---

## 14. Summary

Nearest Neighbor Search is a foundational concept that scales from simple brute-force methods to highly optimized ANN algorithms powering modern AI systems. Understanding both exact and approximate approaches is critical for building scalable, high-performance ML and GenAI applications.

---

## 15. Further Reading

- k-NN Algorithm
- Vector Databases
- HNSW Graphs
- Similarity Search in High Dimensions

