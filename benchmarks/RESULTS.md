# Vektr Benchmark Results

All measurements on **MacBook Air (Apple Silicon M-series)**, single process,
CPU-mode embeddings (all-MiniLM-L6-v2), localhost.

---

## 1. HNSW Recall@K (Java unit tests)

| Corpus size | K  | Queries | Recall@K  | Config        |
|-------------|----|---------|-----------|---------------|
| 1,000       | 10 | 50      | **0.982** | M=16, ef=200  |

Recall@10 = 0.982 with default config. Exceeds 0.90 target.

To reproduce:
```bash
mvn test
# Output: Recall@10 over 50 queries: 0.9820
```

## 2. Embedding Latency (Python, CPU mode)

| Call type     | Texts | Latency   |
|---------------|-------|-----------|
| Cold start    | 1     | ~1,500ms  |
| Warm (single) | 1     | **4-9ms** |
| Warm (batch)  | 3     | **20ms**  |

## 3. End-to-End Request Latency

| Operation       | Latency  | Notes                           |
|-----------------|----------|---------------------------------|
| Ingest (1 doc)  | ~800ms   | Chunking + batch embed call     |
| Search (cold)   | ~9ms     | HNSW + BM25 + RRF + embed call  |
| Search (cached) | **<1ms** | LRU cache hit, no embed call    |

## 4. Retrieval Quality

Query: "how does vector search work" against 2 ingested documents.
Top result: chunk containing "dense vector search or sparse BM25 or hybrid RRF"
Correct semantic match confirmed. RRF correctly ranked semantically relevant
document above keyword-unrelated document.

## 5. HNSW Layer Distribution (6 vectors)

Layer 0: 5 nodes, Layer 1: 1 node — matches expected exponential distribution.
