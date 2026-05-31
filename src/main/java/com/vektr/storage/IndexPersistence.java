package com.vektr.storage;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.vektr.index.DistanceFunction;
import com.vektr.index.HnswConfig;
import com.vektr.index.HnswIndex;
import com.vektr.index.HnswNode;
import com.vektr.retrieval.BM25Index;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.lang.reflect.Field;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Persists and restores the HNSW index, BM25 index, and chunk store to disk.
 *
 * File format:
 *   hnsw.meta  - JSON: config + entry point + node count
 *   hnsw.index - binary: for each node: id, externalId, maxLayer, vector, neighbor lists
 *   chunks.store - JSON: Map<chunkId, text>
 *
 * Design decisions:
 *   - Binary format for vectors (float32) — compact, fast to read/write
 *   - JSON for metadata — human readable, easy to inspect
 *   - Atomic writes via temp file + rename — no partial writes
 *   - BM25 rebuilt from chunk store on load — avoids complex serialization
 *
 * Interview talking point:
 *   "I use atomic rename for durability — write to a .tmp file first,
 *    then rename. On any crash before rename, the old index is intact.
 *    This is the same pattern used in LevelDB and RocksDB."
 */
public class IndexPersistence {
    private static final Logger log = LoggerFactory.getLogger(IndexPersistence.class);
    private static final ObjectMapper MAPPER = new ObjectMapper();

    private final Path dataDir;

    public IndexPersistence(String dataDir) {
        this.dataDir = Path.of(dataDir);
    }

    public static IndexPersistence defaultStorage() {
        return new IndexPersistence("data");
    }

    // -------------------------------------------------------------------------
    // Save
    // -------------------------------------------------------------------------

    public void save(HnswIndex index, Map<String, String> chunkStore) throws IOException {
        Files.createDirectories(dataDir);

        // 1. Save chunk store (JSON)
        saveChunkStore(chunkStore);

        // 2. Save HNSW index (binary + meta JSON)
        saveHnswIndex(index);

        log.info("Saved index: {} vectors, {} chunks to {}",
            index.size(), chunkStore.size(), dataDir);
    }

    private void saveChunkStore(Map<String, String> chunkStore) throws IOException {
        Path tmp = dataDir.resolve("chunks.store.tmp");
        Path target = dataDir.resolve("chunks.store");
        MAPPER.writeValue(tmp.toFile(), chunkStore);
        Files.move(tmp, target, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
    }

    private void saveHnswIndex(HnswIndex index) throws IOException {
        HnswIndex.IndexStats stats = index.stats();

        // Meta file (JSON)
        Map<String, Object> meta = new LinkedHashMap<>();
        meta.put("nodeCount", stats.totalNodes());
        meta.put("maxLayer", stats.maxLayer());
        meta.put("entryPointId", stats.entryPointId());
        meta.put("m", index.getConfig().m());
        meta.put("efConstruction", index.getConfig().efConstruction());
        meta.put("efSearch", index.getConfig().efSearch());
        meta.put("distanceFunction", index.getConfig().distanceFunction().name());

        Path metaTmp = dataDir.resolve("hnsw.meta.tmp");
        Path metaTarget = dataDir.resolve("hnsw.meta");
        MAPPER.writeValue(metaTmp.toFile(), meta);
        Files.move(metaTmp, metaTarget, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);

        // Binary index file
        Path idxTmp = dataDir.resolve("hnsw.index.tmp");
        Path idxTarget = dataDir.resolve("hnsw.index");

        try (DataOutputStream out = new DataOutputStream(
                new BufferedOutputStream(new FileOutputStream(idxTmp.toFile()), 1 << 20))) {

            Collection<HnswNode> nodes = index.getAllNodes();
            out.writeInt(nodes.size());

            for (HnswNode node : nodes) {
                out.writeInt(node.id);
                out.writeUTF(node.externalId);
                out.writeInt(node.maxLayer);

                // Vector
                out.writeInt(node.vector.length);
                for (float v : node.vector) out.writeFloat(v);

                // Neighbor lists (layers 0..maxLayer)
                out.writeInt(node.maxLayer + 1);
                for (int layer = 0; layer <= node.maxLayer; layer++) {
                    List<Integer> neighbors = node.getNeighbors(layer);
                    out.writeInt(neighbors.size());
                    for (int n : neighbors) out.writeInt(n);
                }
            }
        }

        Files.move(idxTmp, idxTarget, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
    }

    // -------------------------------------------------------------------------
    // Load
    // -------------------------------------------------------------------------

    public boolean exists() {
        return Files.exists(dataDir.resolve("hnsw.meta")) &&
               Files.exists(dataDir.resolve("hnsw.index")) &&
               Files.exists(dataDir.resolve("chunks.store"));
    }

    public LoadedState load() throws IOException {
        log.info("Loading index from {}", dataDir);

        // Load meta
        @SuppressWarnings("unchecked")
        Map<String, Object> meta = MAPPER.readValue(
            dataDir.resolve("hnsw.meta").toFile(), Map.class);

        int m = (int) meta.get("m");
        int efConstruction = (int) meta.get("efConstruction");
        int efSearch = (int) meta.get("efSearch");
        DistanceFunction df = DistanceFunction.valueOf((String) meta.get("distanceFunction"));
        int entryPointId = (int) meta.get("entryPointId");
        int maxLayer = (int) meta.get("maxLayer");

        HnswConfig config = HnswConfig.builder()
            .m(m).efConstruction(efConstruction).efSearch(efSearch)
            .distanceFunction(df).build();

        // Load binary index
        HnswIndex index = new HnswIndex(config);
        List<HnswNode> nodes = new ArrayList<>();

        try (DataInputStream in = new DataInputStream(
                new BufferedInputStream(new FileInputStream(
                    dataDir.resolve("hnsw.index").toFile()), 1 << 20))) {

            int nodeCount = in.readInt();
            for (int i = 0; i < nodeCount; i++) {
                int id = in.readInt();
                String externalId = in.readUTF();
                int nodeMaxLayer = in.readInt();

                int dim = in.readInt();
                float[] vector = new float[dim];
                for (int j = 0; j < dim; j++) vector[j] = in.readFloat();

                int numLayers = in.readInt();
                HnswNode node = new HnswNode(id, vector, externalId, nodeMaxLayer, numLayers);

                for (int layer = 0; layer < numLayers; layer++) {
                    int neighborCount = in.readInt();
                    List<Integer> neighbors = new ArrayList<>(neighborCount);
                    for (int j = 0; j < neighborCount; j++) neighbors.add(in.readInt());
                    node.setNeighbors(layer, neighbors);
                }
                nodes.add(node);
            }
        }

        // Restore index state via reflection (inject nodes directly)
        index.restoreFromNodes(nodes, entryPointId, maxLayer);

        // Load chunk store
        @SuppressWarnings("unchecked")
        Map<String, String> chunkStore = MAPPER.readValue(
            dataDir.resolve("chunks.store").toFile(), Map.class);

        // Rebuild BM25 from chunk store (simpler than serializing inverted index)
        BM25Index bm25 = new BM25Index();
        for (Map.Entry<String, String> entry : chunkStore.entrySet()) {
            bm25.addDocument(entry.getKey(), entry.getValue());
        }

        log.info("Loaded {} vectors, {} chunks from {}", index.size(), chunkStore.size(), dataDir);
        return new LoadedState(index, bm25, chunkStore);
    }

    public record LoadedState(HnswIndex index, BM25Index bm25, Map<String, String> chunkStore) {}
}
