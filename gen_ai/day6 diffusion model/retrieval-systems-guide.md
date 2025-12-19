# Building Retrieval Systems with Embeddings

Comprehensive guide to implementing semantic search, clustering, and recommendation systems using embeddings and FAISS.

---

## Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [Semantic Search with FAISS](#2-semantic-search-with-faiss)
3. [Clustering with Embeddings](#3-clustering-with-embeddings)
4. [Recommendation Systems](#4-recommendation-systems-with-embeddings)
5. [Complete Example Code](#5-complete-example-code)
6. [Production Considerations](#6-production-considerations)

---

## 1. Environment Setup

Install required packages:

```bash
pip install sentence-transformers faiss-cpu scikit-learn numpy pandas
```

Import dependencies:

```python
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
```

Initialize the embedding model:

```python
# Load a pre-trained sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")
```

Prepare sample corpus:

```python
documents = [
    "Learn Python for data science.",
    "Neural networks for image classification.",
    "Building recommendation systems with embeddings.",
    "Introduction to clustering and k-means.",
    "Deploying machine learning models to production.",
    "Semantic search using vector databases.",
    "Collaborative filtering for movie recommendations.",
    "Topic modeling and document clustering.",
]
doc_ids = list(range(len(documents)))

# Generate embeddings
embeddings = model.encode(documents, convert_to_numpy=True, normalize_embeddings=True)
dim = embeddings.shape[1]  # dimension of embeddings (typically 384 for all-MiniLM-L6-v2)

print(f"Generated {len(embeddings)} embeddings of dimension {dim}")
```

---

## 2. Semantic Search with FAISS

Semantic search retrieves documents by meaning rather than exact keyword matching. This is powerful for finding similar content across large corpora.

### 2.1 Build the FAISS Index

FAISS (Facebook AI Similarity Search) is a library for efficient similarity search. We use `IndexFlatIP` for exact inner product search on normalized vectors (equivalent to cosine similarity).

```python
# Create index for exact inner product search
index = faiss.IndexFlatIP(dim)

# Add all document embeddings to the index
index.add(embeddings)

print(f"Index contains {index.ntotal} vectors")
```

### 2.2 Semantic Search Function

```python
def semantic_search(query, top_k=3):
    """
    Search for documents most similar to the query.
    
    Args:
        query (str): The search query
        top_k (int): Number of top results to return
    
    Returns:
        list: List of results with doc_id, similarity score, and text
    """
    # Encode the query
    q_vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    
    # Search the index
    scores, idxs = index.search(q_vec, top_k)
    
    # Format results
    results = []
    for score, idx in zip(scores[0], idxs[0]):
        results.append({
            "doc_id": int(idx),
            "score": float(score),
            "text": documents[idx],
        })
    return results
```

### 2.3 Example Usage

```python
# Search for documents related to recommendations
query = "How to build a recommender?"
results = semantic_search(query, top_k=3)

print(f"\nQuery: {query}\n")
for i, result in enumerate(results, 1):
    print(f"{i}. [Score: {result['score']:.4f}] {result['text']}")
```

---

## 3. Clustering with Embeddings

Clustering groups semantically similar documents without requiring labels. Use embeddings to discover natural topic groupings in your corpus.

### 3.1 K-Means Clustering

```python
# Define number of clusters
num_clusters = 3

# Initialize and fit K-Means
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
labels = kmeans.fit_predict(embeddings)

# Group documents by cluster
clusters = {i: [] for i in range(num_clusters)}
for doc_id, label in zip(doc_ids, labels):
    clusters[int(label)].append({
        "doc_id": doc_id,
        "text": documents[doc_id],
    })

# Display clusters
for cid in range(num_clusters):
    print(f"\n{'='*60}")
    print(f"CLUSTER {cid} ({len(clusters[cid])} documents)")
    print('='*60)
    for doc in clusters[cid]:
        print(f"  [{doc['doc_id']}] {doc['text']}")
```

### 3.2 Hierarchical Clustering Alternative

For more nuanced clustering, use hierarchical methods:

```python
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist

# Compute pairwise distances
distances = pdist(embeddings, metric='cosine')

# Perform hierarchical clustering
Z = linkage(distances, method='ward')

# Cut tree at specified height to get clusters
clusters_hier = fcluster(Z, t=3, criterion='maxclust')

print("Hierarchical Clustering Labels:", clusters_hier)
```

---

## 4. Recommendation Systems with Embeddings

Build content-based recommendation systems by finding similar items in embedding space.

### 4.1 Item-to-Item Recommendations

Given a document, recommend similar documents:

```python
def recommend_similar_items(item_id, top_k=3):
    """
    Recommend documents similar to a given item.
    
    Args:
        item_id (int): ID of the reference item
        top_k (int): Number of recommendations to return
    
    Returns:
        list: List of recommended items with scores
    """
    item_vec = embeddings[item_id : item_id + 1]
    
    # Search for top_k + 1 to account for the item itself
    scores, idxs = index.search(item_vec, top_k + 1)
    
    recs = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx == item_id:
            continue  # Skip the reference item itself
        recs.append({
            "doc_id": int(idx),
            "score": float(score),
            "text": documents[idx],
        })
    return recs[:top_k]

# Example
print("Original item:")
item_id = 2
print(f"  [{item_id}] {documents[item_id]}\n")

print("Similar recommendations:")
recs = recommend_similar_items(item_id, top_k=3)
for i, rec in enumerate(recs, 1):
    print(f"{i}. [Score: {rec['score']:.4f}] {rec['text']}")
```

### 4.2 Query-Based Recommendations

Find documents relevant to a user query:

```python
def recommend_for_query(query, top_k=3):
    """
    Recommend documents based on a user query.
    
    Args:
        query (str): User query
        top_k (int): Number of recommendations
    
    Returns:
        list: Recommended documents
    """
    return semantic_search(query, top_k=top_k)

# Example
query = "I want to learn about clustering"
print(f"\nQuery: {query}\n")
recs = recommend_for_query(query, top_k=3)
for i, rec in enumerate(recs, 1):
    print(f"{i}. [Score: {rec['score']:.4f}] {rec['text']}")
```

### 4.3 User Preference-Based Recommendations

Recommend items based on user's liked items:

```python
def recommend_based_on_preferences(liked_doc_ids, top_k=3, diversity=False):
    """
    Recommend documents based on user's liked items.
    
    Args:
        liked_doc_ids (list): IDs of documents the user liked
        top_k (int): Number of recommendations
        diversity (bool): If True, diversify results
    
    Returns:
        list: Recommended documents
    """
    # Average embedding of liked documents
    liked_embeddings = embeddings[liked_doc_ids]
    avg_embedding = np.mean(liked_embeddings, axis=0)
    avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
    
    # Reshape for FAISS
    avg_embedding = avg_embedding.reshape(1, -1)
    
    # Search
    scores, idxs = index.search(avg_embedding, top_k + len(liked_doc_ids))
    
    recs = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx not in liked_doc_ids:  # Don't recommend already-liked items
            recs.append({
                "doc_id": int(idx),
                "score": float(score),
                "text": documents[idx],
            })
    return recs[:top_k]

# Example
liked_docs = [0, 3]  # User liked docs 0 and 3
print("User liked:")
for doc_id in liked_docs:
    print(f"  - {documents[doc_id]}")

print("\nRecommendations:")
recs = recommend_based_on_preferences(liked_docs, top_k=3)
for i, rec in enumerate(recs, 1):
    print(f"{i}. [Score: {rec['score']:.4f}] {rec['text']}")
```

---

## 5. Complete Example Code

Here's a complete, runnable script combining all components:

```python
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from sklearn.cluster import KMeans

# 1. SETUP
model = SentenceTransformer("all-MiniLM-L6-v2")

documents = [
    "Learn Python for data science.",
    "Neural networks for image classification.",
    "Building recommendation systems with embeddings.",
    "Introduction to clustering and k-means.",
    "Deploying machine learning models to production.",
    "Semantic search using vector databases.",
    "Collaborative filtering for movie recommendations.",
    "Topic modeling and document clustering.",
]

embeddings = model.encode(documents, convert_to_numpy=True, normalize_embeddings=True)
dim = embeddings.shape[1]

# 2. BUILD INDEX
index = faiss.IndexFlatIP(dim)
index.add(embeddings)

# 3. SEMANTIC SEARCH
def semantic_search(query, top_k=3):
    q_vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores, idxs = index.search(q_vec, top_k)
    return [(int(idx), float(score), documents[idx]) for score, idx in zip(scores[0], idxs[0])]

# 4. CLUSTERING
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(embeddings)

# 5. RECOMMENDATIONS
def recommend_similar(item_id, top_k=3):
    item_vec = embeddings[item_id:item_id+1]
    scores, idxs = index.search(item_vec, top_k + 1)
    recs = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx != item_id:
            recs.append((int(idx), float(score), documents[idx]))
    return recs[:top_k]

# TEST
print("="*60)
print("SEMANTIC SEARCH")
print("="*60)
results = semantic_search("embedding-based systems", top_k=3)
for doc_id, score, text in results:
    print(f"[{score:.4f}] {text}\n")

print("="*60)
print("RECOMMENDATIONS")
print("="*60)
recs = recommend_similar(2, top_k=3)
for doc_id, score, text in recs:
    print(f"[{score:.4f}] {text}\n")

print("="*60)
print("CLUSTERING")
print("="*60)
for cluster_id in range(3):
    print(f"\nCluster {cluster_id}:")
    for idx, label in enumerate(labels):
        if label == cluster_id:
            print(f"  - {documents[idx]}")
```

---

## 6. Production Considerations

### 6.1 Scalability

For large-scale applications (millions of documents):

- **Approximate Nearest Neighbors (ANN):** Use `IndexIVFFlat`, `IndexHNSW`, or `IndexLSH` in FAISS for faster search with trade-offs in accuracy.
- **Vector Databases:** Consider specialized solutions like Pinecone, Weaviate, Qdrant, or Milvus for production deployments.
- **Distributed Indexing:** Shard embeddings across multiple machines.

```python
# Example: HNSW index for faster approximate search
index_hnsw = faiss.IndexHNSWFlat(dim, 32)  # 32 connections per node
index_hnsw.add(embeddings)
scores, idxs = index_hnsw.search(q_vec, top_k)
```

### 6.2 Metadata and Filtering

Store metadata alongside embeddings:

```python
# Store metadata in a dictionary
metadata = {
    0: {"category": "beginner", "date": "2024-01-01"},
    1: {"category": "advanced", "date": "2024-01-02"},
    # ...
}

# Filter results by metadata
def search_with_filter(query, category_filter=None, top_k=3):
    results = semantic_search(query, top_k * 3)  # Get more to filter
    filtered = [r for r in results if category_filter is None or 
                metadata.get(r[0], {}).get("category") == category_filter]
    return filtered[:top_k]
```

### 6.3 Hybrid Search

Combine lexical (BM25) and semantic search:

```python
from rank_bm25 import BM25Okapi

corpus_tokens = [doc.lower().split() for doc in documents]
bm25 = BM25Okapi(corpus_tokens)

def hybrid_search(query, top_k=3, semantic_weight=0.7):
    # Semantic scores
    semantic_results = semantic_search(query, top_k * 2)
    semantic_scores = {r[0]: r[1] for r in semantic_results}
    
    # BM25 scores
    query_tokens = query.lower().split()
    bm25_scores = bm25.get_scores(query_tokens)
    
    # Combine scores
    combined = {}
    for doc_id in range(len(documents)):
        semantic = semantic_scores.get(doc_id, 0)
        bm25_norm = bm25_scores[doc_id] / (max(bm25_scores) + 1e-10)
        combined[doc_id] = semantic_weight * semantic + (1 - semantic_weight) * bm25_norm
    
    # Return top-k
    sorted_docs = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [(doc_id, score, documents[doc_id]) for doc_id, score in sorted_docs]
```

### 6.4 Caching and API

Expose your system via an API with caching:

```python
from fastapi import FastAPI
from functools import lru_cache

app = FastAPI()

@lru_cache(maxsize=1000)
def cached_search(query: str, top_k: int):
    return semantic_search(query, top_k)

@app.get("/search/")
def search_endpoint(q: str, top_k: int = 3):
    results = cached_search(q, top_k)
    return [{"doc_id": r[0], "score": r[1], "text": r[2]} for r in results]

@app.get("/recommend/")
def recommend_endpoint(item_id: int, top_k: int = 3):
    results = recommend_similar(item_id, top_k)
    return [{"doc_id": r[0], "score": r[1], "text": r[2]} for r in results]

# Run with: uvicorn script:app --reload
```

### 6.5 Monitoring and Evaluation

Track system performance:

```python
from datetime import datetime

metrics = {
    "searches": 0,
    "avg_latency": 0,
    "total_time": 0,
    "timestamps": []
}

def monitored_search(query, top_k=3):
    import time
    start = time.time()
    results = semantic_search(query, top_k)
    latency = time.time() - start
    
    metrics["searches"] += 1
    metrics["total_time"] += latency
    metrics["avg_latency"] = metrics["total_time"] / metrics["searches"]
    metrics["timestamps"].append(datetime.now().isoformat())
    
    return results, latency

# Get metrics
results, latency = monitored_search("test query")
print(f"Queries processed: {metrics['searches']}")
print(f"Average latency: {metrics['avg_latency']:.4f}s")
```

---

## Summary

**Semantic Search:** Retrieve documents by meaning using embeddings and similarity search.

**Clustering:** Group documents into topics using K-Means or hierarchical methods on embeddings.

**Recommendations:** Find similar items (item-to-item) or relevant content (query-based) using embedding similarity.

All three applications share the same foundation: embeddings + similarity search. Scale them with FAISS indices, vector databases, and production patterns like caching, filtering, and hybrid approaches.

For questions or updates, refer to official documentation:
- [Sentence Transformers](https://www.sbert.net/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Scikit-learn](https://scikit-learn.org/)
