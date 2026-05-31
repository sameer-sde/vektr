# Vektr Benchmark Results

All measurements on MacBook Air (Apple Silicon M-series), single process,
CPU-mode embeddings (all-MiniLM-L6-v2), localhost.

## 1. HNSW Recall@K (Java unit tests)

| Corpus size | K  | Queries | Recall@K  | Config       |
|-------------|----|---------|-----------|--------------|
| 1,000       | 10 | 50      | **0.982** | M=16, ef=200 |

Recall@10 = 0.982 with default config. Exceeds 0.90 target by large margin.

To reproduce:
  mvn test
  # Output: Recall@10 over 50 queries: 0.9820

## 2. Embedding Latency (Python, CPU mode)

| Call type     | Texts | Latency    |
|---------------|-------|------------|
| Cold start    | 1     | ~1,500ms   |
| Warm (single) | 1     | **4-9ms**  |
| Warm (batch)  | 3     | **20ms**   |

After first call warms up the model, subsequent calls run at 4-9ms single,
20ms for batch of 3. This is the bottleneck in the ingest pipeline.

## 3. End-to-End Request Latency

| Operation       | Latency  | Notes                          |
|-----------------|----------|--------------------------------|
| Ingest (1 doc)  | ~800ms   | Chunking + batch embed call    |
| Search (cold)   | ~9ms     | HNSW + BM25 + RRF + embed call |
| Search (cached) | **<1ms** | LRU cache hit, no embed call   |

LRU cache capacity: 10,000 entries. Cache hit rate reaches 100% on repeated
identical queries. ReadWriteLock allows concurrent readers without blocking.

## 4. Retrieval Quality (manual evaluation)

Query: "how does vector search work" against 2 ingested documents.

Rank 1: "...dense vector search or sparse BM25 or hybrid RRF"  (rag-explained)
Rank 2: "...reduces hallucinations...dense vector search..."   (rag-explained)
Rank 3: "Retrieval Augmented Generation combines..."           (rag-explained)

Correct semantic match confirmed. RRF correctly ranked the semantically
relevant document above the HNSW-paper document for this query.

## 5. HNSW Layer Distribution (6 vectors indexed)

  Layer 0: 5 nodes  (base layer)
  Layer 1: 1 node   (highway layer)
  Max layer: 1

Matches expected exponential distribution from the HNSW paper:
P(layer=k) = floor(-ln(uniform) * mL), mL = 1/ln(M) = 1/ln(16) = 0.36

## 6. Planned Benchmarks

- [ ] Recall@K at 10k, 100k, 1M vectors
- [ ] QPS load test with k6 (target: 1k QPS at p99 <50ms)
- [ ] BM25+dense vs dense-only recall comparison (RRF uplift)
- [ ] Cache hit rate at various query distributions

## Reproducing

  source venv/bin/activate
  TRANSFORMERS_OFFLINE=1 python3 ml/embed_server.py &
  java -Xmx512m -jar target/vektr-1.0.0.jar &
  mvn test

---

## 7. Real Dataset Results (Week 6)

Ingested 31 Wikipedia articles across AI/ML/CS topics.

| Metric | Value |
|--------|-------|
| Total articles | 31 |
| Total vectors | 84 |
| Avg chunks per article | 2.7 |
| Avg embed latency per batch | ~28ms |
| Index persisted to disk | yes (atomic rename) |
| Index load time on restart | <15ms |

Sample query: "how does attention mechanism work"
Results correctly ranked chunks from:
  - Attention_(machine_learning)  [rank 1-2]
  - Transformer architecture      [rank 3]
  - BERT                          [rank 4]

Cross-document semantic retrieval working correctly.
