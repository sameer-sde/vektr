package com.vektr.query;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import java.util.concurrent.atomic.AtomicLong;

/**
 * LRU cache for query embeddings.
 *
 * Problem it solves: every search request calls the Python embedding
 * service (~5-15ms round trip). The same queries get asked repeatedly
 * (e.g. "what is machine learning" from many users). Caching the
 * embedding vector avoids the Python call entirely.
 *
 * Design:
 * - LinkedHashMap in access-order mode = LRU eviction built in
 * - ReentrantReadWriteLock: many concurrent readers, exclusive writers
 * - Key: normalized query string (lowercased, trimmed)
 * - Value: float[] embedding vector
 *
 * Same pattern used in Sentinel's model cache — the interview story:
 * "I applied the same LRU cache pattern across both projects, but the
 * tradeoff is different: in Sentinel we cache model decisions, here we
 * cache embedding round-trips to a Python microservice."
 */
public class QueryCache {
    private static final Logger log = LoggerFactory.getLogger(QueryCache.class);

    private final int capacity;
    private final LinkedHashMap<String, float[]> cache;
    private final ReentrantReadWriteLock lock = new ReentrantReadWriteLock();

    private final AtomicLong hits = new AtomicLong(0);
    private final AtomicLong misses = new AtomicLong(0);

    public QueryCache(int capacity) {
        this.capacity = capacity;
        this.cache = new LinkedHashMap<>(capacity, 0.75f, true) {
            @Override
            protected boolean removeEldestEntry(Map.Entry<String, float[]> eldest) {
                return size() > capacity;
            }
        };
        log.info("QueryCache initialized: capacity={}", capacity);
    }

    public static QueryCache defaultCache() {
        return new QueryCache(10_000);
    }

    /**
     * Look up a cached embedding for this query.
     * @return float[] if cached, null if miss
     */
    public float[] get(String query) {
        String key = normalize(query);
        lock.readLock().lock();
        try {
            float[] cached = cache.get(key);
            if (cached != null) { hits.incrementAndGet(); return cached; }
        } finally { lock.readLock().unlock(); }

        // Upgrade to write lock for LRU access-order update
        lock.writeLock().lock();
        try {
            float[] cached = cache.get(key);
            if (cached != null) { hits.incrementAndGet(); return cached; }
            misses.incrementAndGet();
            return null;
        } finally { lock.writeLock().unlock(); }
    }

    /**
     * Store an embedding in the cache.
     */
    public void put(String query, float[] embedding) {
        String key = normalize(query);
        lock.writeLock().lock();
        try {
            cache.put(key, embedding);
        } finally { lock.writeLock().unlock(); }
    }

    public CacheStats stats() {
        lock.readLock().lock();
        try {
            long h = hits.get(), m = misses.get(), total = h + m;
            double hitRate = total == 0 ? 0.0 : (double) h / total;
            return new CacheStats(h, m, cache.size(), capacity, hitRate);
        } finally { lock.readLock().unlock(); }
    }

    public void clear() {
        lock.writeLock().lock();
        try { cache.clear(); hits.set(0); misses.set(0); }
        finally { lock.writeLock().unlock(); }
    }

    private String normalize(String query) {
        return query == null ? "" : query.toLowerCase().trim();
    }

    public record CacheStats(long hits, long misses, int size, int capacity, double hitRate) {
        @Override public String toString() {
            return "CacheStats{hits=%d, misses=%d, size=%d, hitRate=%.3f}"
                .formatted(hits, misses, size, hitRate);
        }
    }
}
