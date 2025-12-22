# Graph RAG – Knowledge Graph + FAISS Retrieval

This markdown file explains **Graph RAG** workflow using **FAISS for vector retrieval** and **NetworkX for graph visualization**.

---

## 1. Dummy Data for RAG

```python
documents = [
    {'id': 'doc1', 'title': 'Machine Learning', 'text': 'Machine learning allows systems to learn from data. Supervised and unsupervised learning are common types.'},
    {'id': 'doc2', 'title': 'Deep Learning', 'text': 'Deep learning is a subset of machine learning. It uses neural networks with many layers.'},
    {'id': 'doc3', 'title': 'RAG', 'text': 'Retrieval Augmented Generation combines retrieval systems with large language models.'},
    {'id': 'doc4', 'title': 'Graph Neural Networks', 'text': 'Graph neural networks work on graph structured data like nodes and edges.'}
]
```

---

## 2. Chunking

```python
def chunk_text(text, chunk_size=20):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

chunks = []
metadata = []
for doc in documents:
    for chunk in chunk_text(doc['text']):
        chunks.append(chunk)
        metadata.append({"doc_id": doc['id'], "title": doc['title']})
```

---

## 3. FAISS Vector Store

```python
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(chunks)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))
```

---

## 4. Knowledge Graph

```python
import networkx as nx

graph = nx.Graph()

# Add nodes
graph.add_node('doc1', title='Machine Learning')
graph.add_node('doc2', title='Deep Learning')
graph.add_node('doc3', title='RAG')
graph.add_node('doc4', title='Graph Neural Networks')

# Add edges
graph.add_edge('doc1', 'doc2', relation='subset')
graph.add_edge('doc1', 'doc3', relation='used_in')
graph.add_edge('doc2', 'doc4', relation='inspired')
```

---

## 5. Graph RAG Query Function

```python
def graph_rag_query(query, top_k=2):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)

    retrieved_chunks = []
    related_docs = set()

    for idx in indices[0]:
        retrieved_chunks.append(chunks[idx])
        related_docs.add(metadata[idx]['doc_id'])

    graph_context = []
    for doc_id in related_docs:
        for neighbor in graph.neighbors(doc_id):
            graph_context.append(f"{doc_id} -> {neighbor} ({graph.edges[doc_id, neighbor]['relation']})")

    return retrieved_chunks, graph_context

query = "What is the relationship between machine learning and deep learning?"
retrieved_text, graph_info = graph_rag_query(query)

print("Retrieved Chunks:")
for t in retrieved_text:
    print("-", t)

print("\nGraph Context:")
for g in graph_info:
    print("-", g)
```

---

## 6. Visualize Graph RAG

```python
def build_query_graph(full_graph, related_docs):
    query_graph = nx.Graph()
    for doc in related_docs:
        query_graph.add_node(doc, title=full_graph.nodes[doc]['title'])
        for neighbor in full_graph.neighbors(doc):
            query_graph.add_node(neighbor, title=full_graph.nodes[neighbor]['title'])
            query_graph.add_edge(doc, neighbor, relation=full_graph.edges[doc, neighbor]['relation'])
    return query_graph

# Extract doc_ids from graph_info
related_docs = set()
for info in graph_info:
    doc_id = info.split('->')[0].strip()
    related_docs.add(doc_id)

query_graph = build_query_graph(graph, related_docs)

import matplotlib.pyplot as plt

plt.figure(figsize=(9,7))
pos = nx.spring_layout(query_graph, seed=42)
nx.draw(query_graph, pos, with_labels=True, node_size=3500)
edge_labels = nx.get_edge_attributes(query_graph, 'relation')
nx.draw_networkx_edge_labels(query_graph, pos, edge_labels=edge_labels)
plt.title("Graph RAG – Query-Aware Knowledge Graph")
plt.show()
```

---

### Notes:
- FAISS retrieves relevant text chunks based on semantic similarity.
- The graph shows relationships between documents to provide context.
- Query-aware subgraph highlights only the relevant nodes and edges.

This is the **Graph RAG workflow** combining vector search and knowledge graph reasoning.

