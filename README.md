# Vektr — Production RAG Engine

![Vektr Dashboard](docs/images/dashboard.png)

Production RAG engine in Java/Python. The HNSW vector index is written from scratch — no FAISS, no Chroma, no Pinecone.

**84 vectors · 31 Wikipedia articles · 35ms query latency · recall@10 = 0.984**

---

## What it does

You POST documents → Vektr chunks, embeds, and indexes them.
You ask a question → Vektr finds the most relevant chunks in milliseconds.

This is the infrastructure layer that powers document Q&A — what Pinecone, Google Vertex AI Search, and Amazon Kendra sell as a product.

---

## How it works

1. Document comes in → split into overlapping sentence chunks
2. Each chunk → embedded into a 384-dim vector via sentence-transformers
3. Vectors stored in a hand-rolled HNSW graph index
4. Query comes in → check LRU cache → embed query → search HNSW + BM25 in parallel
5. Results fused via Reciprocal Rank Fusion → ranked chunks returned

---

## Key components

**HNSW Vector Index**
Implements Malkov and Yashunin 2018. Multi-layer graph where upper layers are long-range highway links and layer 0 is dense local connections. Probabilistic layer assignment, greedy descent, beam search. recall@10 = 0.984 on 1000 vectors.

**BM25 + Dense Hybrid Retrieval**
Dense retrieval alone misses exact keyword matches. BM25 alone misses semantic meaning. Reciprocal Rank Fusion combines both ranked lists without needing to normalize scores across different scales.

**HyDE Query Rewriting**
Gao et al. 2022 (Google Brain). Instead of embedding the raw query, generate a hypothetical answer and embed that. The embedding lives in answer-space, much closer to real documents. Improves recall on short queries.

**LRU Embedding Cache**
Caches query embeddings with a ReadWriteLock — concurrent readers never block each other. Cache hits skip the Python round-trip entirely. Under 1ms for repeated queries.

**Persistent Index**
Serializes HNSW graph to binary (float32 vectors + neighbor lists per layer). Atomic write pattern: write to .tmp file, then rename. Index survives restarts and loads in under 15ms.

---

## Benchmarks

| Metric | Value |
|--------|-------|
| HNSW Recall@10 | 0.984 |
| Embedding latency (warm batch) | 28ms |
| Search latency cold | 35ms |
| Search latency cached | under 1ms |
| Index load from disk | under 15ms |
| Dataset | 31 Wikipedia articles, 84 vectors |

---

## Quick Start

Start the embedding service:

    cd ml
    pip install -r requirements.txt
    TRANSFORMERS_OFFLINE=1 python3 embed_server.py

Start the Java server:

    mvn package -q -DskipTests
    java -Xmx512m -jar target/vektr-1.0.0.jar

Open the dashboard at http://localhost:8080

Run the recall benchmark:

    mvn test

---

## Project Structure

    vektr/
    src/main/java/com/vektr/
        index/      HnswIndex, HnswNode, HnswConfig, DistanceFunction
        retrieval/  BM25Index, ReciprocalRankFusion
        pipeline/   DocumentChunker, EmbeddingClient
        query/      QueryCache, QueryRewriter (HyDE)
        storage/    IndexPersistence
        server/     VektrServer
    src/test/       HnswIndexTest (recall benchmark)
    ml/
        embed_server.py
        ingest_wikipedia.py
    benchmarks/RESULTS.md

---

## References

- Malkov and Yashunin 2018 HNSW https://arxiv.org/abs/1603.09320
- Gao et al 2022 HyDE https://arxiv.org/abs/2212.10496
- Cormack et al 2009 Reciprocal Rank Fusion SIGIR
- sentence-transformers all-MiniLM-L6-v2
