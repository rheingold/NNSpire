# Vector Databases

**Document Version:** 1.0  
**Generated:** December 4, 2025  
**Standard:** Vector Database API Specifications  
**Status:** Industry Standards

---

## Overview

**Vector databases** are specialized systems for storing and searching high-dimensional embeddings. They enable semantic search, similarity matching, and AI-powered retrieval at scale.

### Key Capabilities

- **ANN Search**: Approximate nearest neighbor search
- **Semantic Search**: Find similar items by meaning
- **Scalability**: Handle millions/billions of vectors
- **Filtering**: Metadata-based filtering with vectors
- **Real-time**: Fast index updates
- **Distributed**: Multi-node clusters

### Index Structures

| Type | Complexity | Accuracy | Speed | Best For |
|------|-----------|----------|-------|----------|
| **HNSW** | O(log N) | 95-99% | Medium | General purpose |
| **IVF** | O(k + log N) | 90-95% | Fast | Large-scale |
| **LSH** | O(1) | 70-90% | Very Fast | Approximate |
| **Annoy** | O(log N) | 90-95% | Medium | Sparse data |
| **ScaNN** | O(log N) | 95-99% | Fast | Google |

### HNSW (Hierarchical Navigable Small World) — In Depth

HNSW is the **most widely used ANN index** across vector databases (Pinecone, Weaviate, Qdrant, Chroma, Milvus). It builds a multi-layer graph of proximity links for logarithmic-time nearest-neighbor search.

**Core Idea**:
```
Layer 3 (fewest nodes):   A ─────────────────── D        ← long-range links
Layer 2:                  A ──── B ──────────── D        ← medium-range links
Layer 1:                  A ── B ── C ── D ── E ── F    ← short-range links
Layer 0 (all nodes):      A─B─C─D─E─F─G─H─I─J─K─L─M   ← dense local links

Search starts at top layer → greedy descend → refine at bottom.
```

**Key Parameters**:
| Parameter | Meaning | Typical Range |
|-----------|---------|---------------|
| `M` | Max connections per node per layer | 16–64 |
| `efConstruction` | Beam width during index build | 100–200 |
| `efSearch` | Beam width during query | 50–200 |

Higher values → better recall, slower speed.

**Build Pseudocode**:
```
PROCEDURE HNSW_Insert(graph, new_node, M, efConstruction)
  // Step 1: Assign a random layer level (exponential decay)
  max_layer ← Floor(-ln(Random()) × (1 / ln(M)))

  // Step 2: Find entry point at the top layer
  entry_point ← graph.top_entry_point

  // Step 3: Greedy traverse from top to (max_layer + 1) — single nearest
  FOR layer FROM graph.max_layer DOWN TO (max_layer + 1):
    entry_point ← GreedyClosest(entry_point, new_node, layer)

  // Step 4: For each layer the node belongs to, find & connect neighbors
  FOR layer FROM min(max_layer, graph.max_layer) DOWN TO 0:
    candidates ← SearchLayer(entry_point, new_node, efConstruction, layer)
    neighbors  ← SelectBest(candidates, M)       // heuristic or simple

    // Bi-directional links
    FOR EACH neighbor IN neighbors:
      AddEdge(new_node, neighbor, layer)
      AddEdge(neighbor, new_node, layer)

      // Prune if neighbor has too many connections
      IF Degree(neighbor, layer) > M_max:
        PruneConnections(neighbor, M_max, layer)

  // Step 5: Update top entry point if new node is higher
  IF max_layer > graph.max_layer:
    graph.top_entry_point ← new_node
    graph.max_layer       ← max_layer
END PROCEDURE
```

**Search Pseudocode**:
```
FUNCTION HNSW_Search(graph, query, K, efSearch) → top_K_results
  entry_point ← graph.top_entry_point

  // Phase 1: Greedy descent through upper layers (single nearest)
  FOR layer FROM graph.max_layer DOWN TO 1:
    entry_point ← GreedyClosest(entry_point, query, layer)

  // Phase 2: Beam search on layer 0
  candidates ← PriorityQueue()   // min-heap by distance
  results    ← PriorityQueue()   // max-heap, bounded by efSearch
  visited    ← Set()

  candidates.Push(entry_point)
  results.Push(entry_point)
  visited.Add(entry_point)

  WHILE candidates IS NOT EMPTY:
    closest ← candidates.PopMin()

    IF Distance(closest, query) > Distance(results.PeekMax(), query):
      BREAK   // no closer candidates possible

    FOR EACH neighbor OF closest AT layer 0:
      IF neighbor NOT IN visited:
        visited.Add(neighbor)
        IF Distance(neighbor, query) < Distance(results.PeekMax(), query)
           OR |results| < efSearch:
          candidates.Push(neighbor)
          results.Push(neighbor)
          IF |results| > efSearch:
            results.PopMax()   // keep bounded

  RETURN results.TopK(K)
END FUNCTION
```

**Complexity**:
- Build: $O(N \log N)$ — each of $N$ inserts costs $O(\log N)$
- Search: $O(\log N)$ — multi-layer greedy + beam search
- Memory: $O(N \times M)$ — adjacency lists per node

---

## Authoritative References

- **Pinecone**: https://docs.pinecone.io
- **Weaviate**: https://weaviate.io/developers/weaviate/
- **Milvus**: https://milvus.io/docs
- **Qdrant**: https://qdrant.tech/documentation
- **FAISS**: https://github.com/facebookresearch/faiss

---

## Vector Database Specifications

### Pinecone

**API Endpoint Structure**:
```
POST https://controller.{region}.pinecone.io/indexes
GET  https://controller.{region}.pinecone.io/indexes
POST https://{index_name}-{project_id}.svc.{region}.pinecone.io/vectors/upsert
POST https://{index_name}-{project_id}.svc.{region}.pinecone.io/query
```

**Create Index Request**:
```json
{
  "name": "document-index",
  "dimension": 1536,
  "metric": "cosine",
  "spec": {
    "serverless": {
      "cloud": "aws",
      "region": "us-west-2"
    }
  }
}
```

**Upsert Vectors**:
```json
{
  "vectors": [
    {
      "id": "doc-001",
      "values": [0.1, 0.2, 0.3, ...],
      "metadata": {
        "source": "document.pdf",
        "page": 1,
        "timestamp": "2024-01-01T00:00:00Z"
      }
    }
  ],
  "namespace": "production"
}
```

**Query**:
```json
{
  "vector": [0.1, 0.2, 0.3, ...],
  "topK": 10,
  "includeMetadata": true,
  "namespace": "production",
  "filter": {
    "source": {"$eq": "document.pdf"}
  }
}
```

**Response**:
```json
{
  "matches": [
    {
      "id": "doc-001",
      "score": 0.89,
      "metadata": {
        "source": "document.pdf",
        "page": 1
      }
    }
  ],
  "namespace": "production"
}
```

**Python Client**:
```python
from pinecone import Pinecone

pc = Pinecone(api_key="xxx")
index = pc.Index("document-index")

# Upsert
index.upsert(
    vectors=[
        ("doc-001", [0.1, 0.2, 0.3], {"source": "doc.pdf"}),
        ("doc-002", [0.2, 0.3, 0.4], {"source": "doc.pdf"}),
    ]
)

# Query
results = index.query(
    vector=[0.1, 0.2, 0.3],
    top_k=10,
    include_metadata=True,
    filter={"source": {"$eq": "doc.pdf"}}
)

for match in results.matches:
    print(f"{match.id}: {match.score}")
```

### Weaviate

**REST API Structure**:
```
POST  /v1/objects                  # Create object
GET   /v1/objects/{id}             # Get object
POST  /v1/objects/search           # Search
POST  /v1/graphql                  # GraphQL query
POST  /v1/classifiers              # Classification
```

**Class Schema**:
```json
{
  "class": "Document",
  "description": "A text document",
  "vectorIndexConfig": {
    "distance": "cosine",
    "ef": 100,
    "efConstruction": 128,
    "maxConnections": 30,
    "vectorCacheMaxObjects": 100000
  },
  "properties": [
    {
      "name": "content",
      "dataType": ["text"],
      "description": "Document content"
    },
    {
      "name": "source",
      "dataType": ["string"],
      "description": "Document source"
    },
    {
      "name": "timestamp",
      "dataType": ["date"],
      "description": "Creation time"
    }
  ]
}
```

**Add Object**:
```json
{
  "class": "Document",
  "properties": {
    "content": "This is document content",
    "source": "document.pdf",
    "timestamp": "2024-01-01T00:00:00Z"
  }
}
```

**Vector Search Query**:
```graphql
{
  Get {
    Document(
      nearVector: {
        vector: [0.1, 0.2, 0.3]
      }
      limit: 10
      where: {
        path: ["source"]
        operator: Equal
        valueString: "document.pdf"
      }
    ) {
      _additional {
        distance
        vector
      }
      content
      source
      timestamp
    }
  }
}
```

**Python Client**:
```python
import weaviate

client = weaviate.connect_to_local()

# Create class
class_obj = {
    "class": "Document",
    "properties": [
        {"name": "content", "dataType": ["text"]},
        {"name": "source", "dataType": ["string"]}
    ]
}
client.collections.create_from_dict(class_obj)

# Add object
doc_collection = client.collections.get("Document")
doc_collection.data.insert({
    "content": "Document text",
    "source": "doc.pdf"
})

# Vector search
response = doc_collection.query.near_vector(
    near_vector=[0.1, 0.2, 0.3],
    limit=10,
    where=weaviate.classes.query.Filter.by_property("source").equal("doc.pdf")
)

for obj in response.objects:
    print(f"Score: {obj.metadata.distance}")
    print(f"Content: {obj.properties['content']}")
```

### Milvus

**Abstract API Endpoint Listing**:
```
Milvus gRPC / RESTful API (v2.x)
─────────────────────────────────────────────────────────
Collection Management
  POST   /v2/vectordb/collections/create     Create collection with schema
  GET    /v2/vectordb/collections/list        List all collections
  POST   /v2/vectordb/collections/describe    Get collection details
  POST   /v2/vectordb/collections/drop        Delete collection
  POST   /v2/vectordb/collections/load        Load collection into memory
  POST   /v2/vectordb/collections/release     Release from memory

Index Management
  POST   /v2/vectordb/indexes/create          Build ANN index (IVF, HNSW, …)
  POST   /v2/vectordb/indexes/describe        Get index details
  POST   /v2/vectordb/indexes/drop            Drop an index

Data Operations
  POST   /v2/vectordb/entities/insert         Insert vectors + scalars
  POST   /v2/vectordb/entities/upsert         Upsert vectors
  POST   /v2/vectordb/entities/delete         Delete by ID or expression
  POST   /v2/vectordb/entities/get            Get by primary key

Search & Query
  POST   /v2/vectordb/entities/search         ANN vector search
  POST   /v2/vectordb/entities/query          Scalar-filter query
  POST   /v2/vectordb/entities/hybrid_search  Hybrid vector + keyword

Typical Flow:
  CreateCollection(schema) → CreateIndex(field, type) → Load()
  → Insert(vectors) → Flush() → Search(query_vector, top_k, filter)
```

**Collection Definition** (Python):
```python
from pymilvus import Collection, FieldSchema, CollectionSchema, DataType

fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536),
    FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=256)
]

schema = CollectionSchema(fields, "Document collection")
collection = Collection("documents", schema)

# Create index
index_params = {
    "index_type": "IVF_FLAT",
    "metric_type": "L2",
    "params": {"nlist": 1024}
}
collection.create_index("embedding", index_params)
```

**Insert Data**:
```python
data = [
    [1, 2, 3],  # IDs
    ["doc1", "doc2", "doc3"],  # content
    [[0.1, 0.2, ..., 0.1536], ...],  # embeddings
    ["source1", "source2", "source3"]  # source
]

collection.insert(data)
collection.flush()
```

**Search Query**:
```python
search_params = {
    "metric_type": "L2",
    "params": {"nprobe": 10}
}

query_vector = [0.1, 0.2, ..., 0.1536]

results = collection.search(
    data=[query_vector],
    anns_field="embedding",
    param=search_params,
    limit=10,
    expr="source == 'source1'",
    output_fields=["id", "content", "source"]
)

for hit in results[0]:
    print(f"ID: {hit.id}, Distance: {hit.distance}")
```

### Qdrant

**API Structure**:
```
POST   /collections                 # Create collection
POST   /collections/{name}/points   # Upsert points
POST   /collections/{name}/points/search  # Search
DELETE /collections/{name}          # Delete collection
```

**Create Collection**:
```json
{
  "vectors": {
    "size": 1536,
    "distance": "Cosine",
    "on_disk": true
  },
  "quantization_config": {
    "scalar": {
      "type": "int8",
      "quantile": 0.99,
      "always_ram": false
    }
  }
}
```

**Upsert Points**:
```json
{
  "points": [
    {
      "id": 1,
      "vector": [0.1, 0.2, 0.3, ...],
      "payload": {
        "source": "document.pdf",
        "page": 1,
        "timestamp": "2024-01-01T00:00:00Z"
      }
    }
  ]
}
```

**Search Request**:
```json
{
  "vector": [0.1, 0.2, 0.3, ...],
  "limit": 10,
  "filter": {
    "must": [
      {
        "key": "source",
        "match": {
          "text": "document.pdf"
        }
      }
    ]
  }
}
```

**Python Client**:
```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

client = QdrantClient(":memory:")

# Create collection
client.create_collection(
    collection_name="documents",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
)

# Upsert points
client.upsert(
    collection_name="documents",
    points=[
        PointStruct(
            id=1,
            vector=[0.1, 0.2, 0.3, ...],
            payload={"source": "doc.pdf", "page": 1}
        )
    ]
)

# Search
results = client.search(
    collection_name="documents",
    query_vector=[0.1, 0.2, 0.3, ...],
    limit=10,
    query_filter={
        "must": [
            {"key": "source", "match": {"text": "doc.pdf"}}
        ]
    }
)

for result in results:
    print(f"ID: {result.id}, Score: {result.score}")
```

### Chroma

**Simple Local Usage**:
```python
import chromadb

# Create client
client = chromadb.Client()

# Create collection
collection = client.create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"}
)

# Add documents
collection.add(
    ids=["doc1", "doc2", "doc3"],
    documents=[
        "First document",
        "Second document",
        "Third document"
    ],
    metadatas=[
        {"source": "file1.pdf"},
        {"source": "file2.pdf"},
        {"source": "file1.pdf"}
    ]
)

# Query (embeddings computed automatically)
results = collection.query(
    query_texts=["similar content"],
    n_results=10,
    where={"source": {"$eq": "file1.pdf"}}
)

print(results)
```

### FAISS (Facebook AI Similarity Search)

**Abstract Algorithm Overview**:
```
FAISS provides several index types. Selection depends on dataset size and accuracy needs:

┌──────────────────────────────────────────────────────────────────┐
│                     FAISS Index Selection                        │
├──────────────┬───────────────────────────────────────────────────┤
│ IndexFlatL2  │ Brute-force exact search.  O(N×d) per query.     │
│              │ Best for < 100K vectors or ground-truth baseline. │
├──────────────┼───────────────────────────────────────────────────┤
│ IndexIVFFlat │ Inverted file with Voronoi partitioning.          │
│              │ Train on sample → assign vectors to clusters.     │
│              │ Query scans only `nprobe` clusters.  ~10–50× fast.│
├──────────────┼───────────────────────────────────────────────────┤
│ IndexIVFPQ   │ IVF + Product Quantization (PQ) compression.     │
│              │ Sub-vectors quantized to centroids → 10–50×       │
│              │ memory savings. Slight accuracy loss.             │
├──────────────┼───────────────────────────────────────────────────┤
│ IndexHNSWFlat│ HNSW graph index. Best recall/speed trade-off.   │
│              │ O(log N) search, higher memory (adjacency lists). │
└──────────────┴───────────────────────────────────────────────────┘

Pseudocode — typical FAISS workflow:

  PROCEDURE BuildAndSearch(vectors[N×d], query[1×d], k)
    // 1. Choose index type
    index ← CreateIndex(type, dimension=d)

    // 2. (Optional) Train on representative sample
    IF index.requires_training:
      index.Train(vectors[:sample_size])

    // 3. Add all vectors
    index.Add(vectors)          // stores vectors + builds internal structure

    // 4. Search
    distances[], indices[] ← index.Search(query, k)
    // distances: k nearest distances
    // indices:   positions in original vectors array

    // 5. (Optional) Move to GPU for 10-100× speedup
    gpu_index ← CopyToGPU(index, gpu_id=0)
    distances[], indices[] ← gpu_index.Search(query, k)
  END PROCEDURE
```

**Index Construction** (Python):
```python
import faiss
import numpy as np

# Data
vectors = np.random.random((100000, 1536)).astype('float32')
index = faiss.IndexFlatL2(1536)  # L2 distance

# Add vectors
index.add(vectors)

# Search
query = np.random.random((1, 1536)).astype('float32')
distances, indices = index.search(query, k=10)

print(f"Nearest neighbors: {indices[0]}")
print(f"Distances: {distances[0]}")
```

**With GPU**:
```python
import faiss

# Create GPU index
res = faiss.StandardGpuResources()
index_cpu = faiss.IndexFlatL2(1536)
index = faiss.index_cpu_to_gpu(res, 0, index_cpu)

# Fast GPU search
distances, indices = index.search(query, k=10)
```

---

## Comparison Matrix

| Feature | Pinecone | Weaviate | Milvus | Qdrant | Chroma |
|---------|----------|----------|--------|--------|--------|
| **Deployment** | Serverless | Self-hosted | Self-hosted | Both | Local/Cloud |
| **Index Type** | HNSW | HNSW | IVF/HNSW | HNSW | HNSW |
| **Filtering** | Native | GraphQL | SQL | Payload | Field filter |
| **Scalability** | Managed | Horizontal | Distributed | Distributed | Single node |
| **Pricing** | Usage-based | Free | Free | Free | Free |
| **API** | REST/Python | REST/GraphQL | Python/gRPC | REST/Python | Python |

---

## Procedural Use-Cases

### Use-Case 1: Build RAG System with Pinecone

```python
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import os

class RAGSystem:
    def __init__(self, index_name: str):
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index = self.pc.Index(index_name)
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
    
    def ingest_documents(self, documents: list, batch_size: int = 100):
        """Embed and store documents"""
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            # Embed batch
            embeddings = self.model.encode(batch)
            
            # Prepare vectors
            vectors = [
                (f"doc-{i+j}", embeddings[j], {"text": batch[j]})
                for j in range(len(batch))
            ]
            
            # Upsert to Pinecone
            self.index.upsert(vectors=vectors)
    
    def retrieve_documents(self, query: str, top_k: int = 5) -> list:
        """Find relevant documents"""
        # Embed query
        query_embedding = self.model.encode(query)
        
        # Search
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        # Extract documents
        documents = [match.metadata["text"] for match in results.matches]
        return documents

# Usage
rag = RAGSystem("document-index")
docs = ["Document 1", "Document 2", ...]
rag.ingest_documents(docs)

results = rag.retrieve_documents("search query")
print(results)
```

### Use-Case 2: Semantic Search with Weaviate

```python
import weaviate
from weaviate.classes.query import Filter

class SemanticSearch:
    def __init__(self):
        self.client = weaviate.connect_to_local()
    
    def setup(self):
        """Initialize Weaviate schema"""
        schema = {
            "class": "Article",
            "properties": [
                {"name": "title", "dataType": ["text"]},
                {"name": "content", "dataType": ["text"]},
                {"name": "category", "dataType": ["string"]}
            ]
        }
        self.client.collections.create_from_dict(schema)
    
    def search(self, query: str, category: str = None):
        """Perform semantic search"""
        articles = self.client.collections.get("Article")
        
        filters = None
        if category:
            filters = Filter.by_property("category").equal(category)
        
        response = articles.query.near_text(
            query=query,
            limit=10,
            where=filters
        )
        
        return response.objects

# Usage
searcher = SemanticSearch()
results = searcher.search("machine learning", category="AI")
```

---

## Tools & Ecosystem

| Tool | Purpose | URL |
|------|---------|-----|
| **Embedding Models** | Generate vectors | https://huggingface.co/models?task=sentence-similarity |
| **Vector CLI** | DB management | https://github.com/qdrant/vector-db-benchmark |
| **Vanna** | RAG on databases | https://github.com/vanna-ai/vanna |
| **LangChain** | Orchestration | https://docs.langchain.com/docs/integrations/vectorstores |

---

**Navigation**: [Back to Index](../INDEX.md) | [Previous: Vector Embeddings](./A01-Vector-Embeddings-Standards.md) | [Next: RAG Patterns](./A03-RAG-Patterns.md)
