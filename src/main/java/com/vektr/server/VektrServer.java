package com.vektr.server;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.vektr.index.HnswConfig;
import com.vektr.index.HnswIndex;
import com.vektr.index.SearchResult;
import com.vektr.pipeline.DocumentChunker;
import com.vektr.pipeline.EmbeddingClient;
import com.vektr.query.QueryCache;
import com.vektr.retrieval.BM25Index;
import com.vektr.retrieval.ReciprocalRankFusion;
import com.vektr.storage.IndexPersistence;
import io.undertow.Undertow;
import io.undertow.server.HttpHandler;
import io.undertow.server.HttpServerExchange;
import io.undertow.server.RoutingHandler;
import io.undertow.server.handlers.BlockingHandler;
import io.undertow.util.Headers;
import io.undertow.util.Methods;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

public class VektrServer {
    private static final Logger log = LoggerFactory.getLogger(VektrServer.class);
    private static final ObjectMapper MAPPER = new ObjectMapper();

    private HnswIndex hnswIndex;
    private BM25Index bm25Index;
    private final ReciprocalRankFusion rrf = new ReciprocalRankFusion();
    private final DocumentChunker chunker = new DocumentChunker();
    private final EmbeddingClient embeddingClient = EmbeddingClient.defaultClient();
    private final QueryCache queryCache = QueryCache.defaultCache();
    private final IndexPersistence persistence = IndexPersistence.defaultStorage();
    private Map<String, String> chunkStore;
    private volatile long totalIngest = 0, totalSearch = 0, totalCacheHits = 0;

    public VektrServer() {
        // Load from disk if index exists, otherwise start fresh
        if (persistence.exists()) {
            try {
                IndexPersistence.LoadedState state = persistence.load();
                this.hnswIndex = state.index();
                this.bm25Index = state.bm25();
                this.chunkStore = new ConcurrentHashMap<>(state.chunkStore());
                log.info("Resumed from disk: {} vectors, {} chunks",
                    hnswIndex.size(), chunkStore.size());
            } catch (Exception e) {
                log.warn("Failed to load index from disk, starting fresh: {}", e.getMessage());
                initFresh();
            }
        } else {
            log.info("No existing index found, starting fresh");
            initFresh();
        }
    }

    private void initFresh() {
        this.hnswIndex = new HnswIndex(HnswConfig.defaults());
        this.bm25Index = new BM25Index();
        this.chunkStore = new ConcurrentHashMap<>();
    }

    private void handleIngest(HttpServerExchange ex) throws Exception {
        byte[] bytes = ex.getInputStream().readAllBytes();
        @SuppressWarnings("unchecked")
        Map<String, String> req = MAPPER.readValue(bytes, Map.class);
        String docId = req.get("doc_id"), text = req.get("text");
        if (docId == null || text == null) { sendError(ex, 400, "doc_id and text required"); return; }
        List<DocumentChunker.Chunk> chunks = chunker.chunk(docId, text);
        log.info("Ingesting doc={} chunks={}", docId, chunks.size());
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

        // Persist after every ingest
        persistence.save(hnswIndex, chunkStore);
        log.info("Persisted index to disk");

        sendJson(ex, 200, Map.of("doc_id", docId, "chunks_indexed", indexed,
            "total_vectors", hnswIndex.size(), "persisted", true));
    }

    private void handleSearch(HttpServerExchange ex) throws Exception {
        byte[] bytes = ex.getInputStream().readAllBytes();
        @SuppressWarnings("unchecked")
        Map<String, Object> req = MAPPER.readValue(bytes, Map.class);
        String query = (String) req.get("query");
        int k = req.containsKey("k") ? (int) req.get("k") : 5;
        if (query == null || query.isBlank()) { sendError(ex, 400, "query required"); return; }
        float[] qEmbed = queryCache.get(query);
        boolean cacheHit = (qEmbed != null);
        if (!cacheHit) {
            qEmbed = embeddingClient.embed(query);
            queryCache.put(query, qEmbed);
        } else { totalCacheHits++; }
        List<SearchResult> dense = hnswIndex.search(qEmbed, k * 2);
        List<BM25Index.BM25Result> sparse = bm25Index.search(query, k * 2);
        List<ReciprocalRankFusion.FusedResult> fused = rrf.fuse(sparse, dense, k);
        List<Map<String, Object>> results = fused.stream().map(r -> {
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("chunk_id", r.docId()); m.put("rrf_score", r.rrfScore());
            m.put("text", chunkStore.getOrDefault(r.docId(), ""));
            return m;
        }).toList();
        totalSearch++;
        sendJson(ex, 200, Map.of("query", query, "cache_hit", cacheHit, "results", results));
    }

    private void handleHealth(HttpServerExchange ex) throws Exception {
        sendJson(ex, 200, Map.of(
            "status", "ok",
            "vectors_indexed", hnswIndex.size(),
            "bm25_docs", bm25Index.size(),
            "embedding_service", embeddingClient.isHealthy() ? "up" : "down",
            "index_persisted", persistence.exists()
        ));
    }

    private void handleMetrics(HttpServerExchange ex) throws Exception {
        HnswIndex.IndexStats s = hnswIndex.stats();
        QueryCache.CacheStats cs = queryCache.stats();
        sendJson(ex, 200, Map.of(
            "total_ingest", totalIngest, "total_search", totalSearch,
            "cache_hit_rate", cs.hitRate(), "cache_size", cs.size(),
            "index_size", s.totalNodes(), "max_layer", s.maxLayer(),
            "layer_distribution", s.layerDistribution()));
    }

    private HttpHandler blocking(CheckedHandler h) {
        return new BlockingHandler(ex -> {
            try { h.handle(ex); }
            catch (Exception e) { log.error("Handler error", e); sendError(ex, 500, e.getMessage()); }
        });
    }

    @FunctionalInterface
    interface CheckedHandler { void handle(HttpServerExchange ex) throws Exception; }

    public void start(int port) {
        RoutingHandler router = new RoutingHandler()
            .add(Methods.POST, "/ingest",  blocking(this::handleIngest))
            .add(Methods.POST, "/search",  blocking(this::handleSearch))
            .add(Methods.GET,  "/health",  blocking(this::handleHealth))
            .add(Methods.GET,  "/metrics", blocking(this::handleMetrics));
        Undertow.builder().addHttpListener(port, "0.0.0.0").setHandler(router).build().start();
        log.info("Vektr started on :{} | vectors={} chunks={}",
            port, hnswIndex.size(), chunkStore.size());
    }

    public static void main(String[] args) {
        int port = args.length > 0 ? Integer.parseInt(args[0]) : 8080;
        new VektrServer().start(port);
    }

    private void sendJson(HttpServerExchange e, int status, Object body) {
        try {
            e.setStatusCode(status);
            e.getResponseHeaders().put(Headers.CONTENT_TYPE, "application/json");
            e.getResponseSender().send(MAPPER.writeValueAsString(body));
        } catch (Exception ex) { log.error("Send error", ex); }
    }

    private void sendError(HttpServerExchange e, int status, String msg) {
        sendJson(e, status, Map.of("error", msg));
    }
}
