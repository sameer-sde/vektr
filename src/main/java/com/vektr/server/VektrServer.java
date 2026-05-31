package com.vektr.server;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.vektr.index.HnswConfig;
import com.vektr.index.HnswIndex;
import com.vektr.index.SearchResult;
import com.vektr.pipeline.DocumentChunker;
import com.vektr.pipeline.EmbeddingClient;
import com.vektr.retrieval.BM25Index;
import com.vektr.retrieval.ReciprocalRankFusion;
import io.undertow.Undertow;
import io.undertow.server.RoutingHandler;
import io.undertow.util.Headers;
import io.undertow.util.Methods;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

public class VektrServer {
    private static final Logger log = LoggerFactory.getLogger(VektrServer.class);
    private static final ObjectMapper MAPPER = new ObjectMapper();
    private final HnswIndex hnswIndex = new HnswIndex(HnswConfig.defaults());
    private final BM25Index bm25Index = new BM25Index();
    private final ReciprocalRankFusion rrf = new ReciprocalRankFusion();
    private final DocumentChunker chunker = new DocumentChunker();
    private final EmbeddingClient embeddingClient = EmbeddingClient.defaultClient();
    private final LlmClient llmClient = LlmClient.defaultClient();
    private final Map<String, String> chunkStore = new ConcurrentHashMap<>();
    private volatile long totalIngest = 0, totalSearch = 0, totalRag = 0;

    private void handleIngest(io.undertow.server.HttpServerExchange ex) {
        ex.getRequestReceiver().receiveFullBytes((e, body) -> {
            try {
                @SuppressWarnings("unchecked")
                Map<String, String> req = MAPPER.readValue(body, Map.class);
                String docId = req.get("doc_id"), text = req.get("text");
                if (docId == null || text == null) { sendError(e, 400, "doc_id and text required"); return; }
                List<DocumentChunker.Chunk> chunks = chunker.chunk(docId, text);
                List<String> texts = chunks.stream().map(DocumentChunker.Chunk::text).toList();
                List<float[]> embeddings = embeddingClient.embedBatch(texts);
                int indexed = 0;
                for (int i = 0; i < chunks.size(); i++) {
                    DocumentChunker.Chunk c = chunks.get(i);
                    if (!hnswIndex.contains(c.id())) hnswIndex.insert(c.id(), embeddings.get(i));
                    bm25Index.addDocument(c.id(), c.text());
                    chunkStore.put(c.id(), c.text());
                    indexed++;
                }
                totalIngest++;
                sendJson(e, 200, Map.of("doc_id", docId, "chunks_indexed", indexed, "total_vectors", hnswIndex.size()));
            } catch (Exception err) { log.error("Ingest error", err); sendError(e, 500, err.getMessage()); }
        });
    }

    private void handleSearch(io.undertow.server.HttpServerExchange ex) {
        ex.getRequestReceiver().receiveFullBytes((e, body) -> {
            try {
                @SuppressWarnings("unchecked")
                Map<String, Object> req = MAPPER.readValue(body, Map.class);
                String query = (String) req.get("query");
                int k = req.containsKey("k") ? (int) req.get("k") : 5;
                if (query == null || query.isBlank()) { sendError(e, 400, "query required"); return; }
                float[] qEmbed = embeddingClient.embed(query);
                List<SearchResult> dense = hnswIndex.search(qEmbed, k * 2);
                List<BM25Index.BM25Result> sparse = bm25Index.search(query, k * 2);
                List<ReciprocalRankFusion.FusedResult> fused = rrf.fuse(sparse, dense, k);
                List<Map<String, Object>> results = fused.stream().map(r -> {
                    Map<String, Object> m = new LinkedHashMap<>();
                    m.put("chunk_id", r.docId());
                    m.put("rrf_score", r.rrfScore());
                    m.put("text", chunkStore.getOrDefault(r.docId(), ""));
                    return m;
                }).toList();
                totalSearch++;
                sendJson(e, 200, Map.of("query", query, "results", results));
            } catch (Exception err) { log.error("Search error", err); sendError(e, 500, err.getMessage()); }
        });
    }

    private void handleHealth(io.undertow.server.HttpServerExchange ex) {
        sendJson(ex, 200, Map.of("status", "ok", "vectors_indexed", hnswIndex.size(),
            "bm25_docs", bm25Index.size(), "embedding_service", embeddingClient.isHealthy() ? "up" : "down"));
    }

    private void handleMetrics(io.undertow.server.HttpServerExchange ex) {
        HnswIndex.IndexStats s = hnswIndex.stats();
        sendJson(ex, 200, Map.of("total_ingest", totalIngest, "total_search", totalSearch,
            "total_rag", totalRag, "index_size", s.totalNodes(),
            "max_layer", s.maxLayer(), "layer_distribution", s.layerDistribution()));
    }

    public void start(int port) {
        RoutingHandler router = new RoutingHandler()
            .add(Methods.POST, "/ingest",   this::handleIngest)
            .add(Methods.POST, "/search",   this::handleSearch)
            .add(Methods.GET,  "/health",   this::handleHealth)
            .add(Methods.GET,  "/metrics",  this::handleMetrics);
        Undertow.builder().addHttpListener(port, "0.0.0.0").setHandler(router).build().start();
        log.info("Vektr started on :{}", port);
        log.info("POST /ingest  POST /search  GET /health  GET /metrics");
    }

    public static void main(String[] args) {
        int port = args.length > 0 ? Integer.parseInt(args[0]) : 8080;
        new VektrServer().start(port);
    }

    private void sendJson(io.undertow.server.HttpServerExchange e, int status, Object body) {
        try {
            e.setStatusCode(status);
            e.getResponseHeaders().put(Headers.CONTENT_TYPE, "application/json");
            e.getResponseSender().send(MAPPER.writeValueAsString(body));
        } catch (Exception ex) { log.error("Send error", ex); }
    }

    private void sendError(io.undertow.server.HttpServerExchange e, int status, String msg) {
        sendJson(e, status, Map.of("error", msg));
    }
}
