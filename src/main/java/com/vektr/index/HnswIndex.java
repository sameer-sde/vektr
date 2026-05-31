package com.vektr.index;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.ReentrantLock;

public class HnswIndex {
    private static final Logger log = LoggerFactory.getLogger(HnswIndex.class);
    private final HnswConfig config;
    private final ConcurrentHashMap<Integer, HnswNode> nodes = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<String, Integer> externalToInternal = new ConcurrentHashMap<>();
    private final AtomicInteger nodeCounter = new AtomicInteger(0);
    private volatile int entryPointId = -1;
    private volatile int maxLayerSeen = -1;
    private final ReentrantLock entryPointLock = new ReentrantLock();
    private static final int LOCK_COUNT = 128;
    private final ReentrantLock[] nodeLocks = new ReentrantLock[LOCK_COUNT];

    public HnswIndex(HnswConfig config) {
        this.config = config;
        for (int i = 0; i < LOCK_COUNT; i++) nodeLocks[i] = new ReentrantLock();
        log.info("HnswIndex M={} ef={}", config.m(), config.efConstruction());
    }

    public int insert(String externalId, float[] vector) {
        if (externalToInternal.containsKey(externalId))
            throw new IllegalArgumentException("Already exists: " + externalId);
        int newLayer = assignLayer();
        int id = nodeCounter.getAndIncrement();
        HnswNode node = new HnswNode(id, vector, externalId, newLayer, newLayer + 1);
        nodes.put(id, node);
        externalToInternal.put(externalId, id);
        entryPointLock.lock();
        try {
            if (entryPointId == -1) {
                entryPointId = id; maxLayerSeen = newLayer; return id;
            }
        } finally { entryPointLock.unlock(); }

        List<Integer> eps = new ArrayList<>();
        eps.add(entryPointId);
        for (int layer = maxLayerSeen; layer > newLayer; layer--)
            eps = greedySearch(vector, eps, 1, layer);
        for (int layer = Math.min(newLayer, maxLayerSeen); layer >= 0; layer--) {
            List<Integer> cands = beamSearch(vector, eps, config.efConstruction(), layer);
            int maxConn = (layer == 0) ? config.mMax0() : config.m();
            List<Integer> selected = nearest(vector, cands, maxConn);
            node.setNeighbors(layer, selected);
            for (int nbId : selected) {
                HnswNode nb = nodes.get(nbId);
                ReentrantLock lk = nodeLocks[nbId % LOCK_COUNT];
                lk.lock();
                try {
                    List<Integer> nc = new ArrayList<>(nb.getNeighbors(layer));
                    nc.add(id);
                    if (nc.size() > maxConn) nc = nearest(nb.vector, nc, maxConn);
                    nb.setNeighbors(layer, nc);
                } finally { lk.unlock(); }
            }
            eps = cands;
        }
        entryPointLock.lock();
        try {
            if (newLayer > maxLayerSeen) { entryPointId = id; maxLayerSeen = newLayer; }
        } finally { entryPointLock.unlock(); }
        return id;
    }

    public List<SearchResult> search(float[] query, int k) {
        return search(query, k, config.efSearch());
    }

    public List<SearchResult> search(float[] query, int k, int ef) {
        if (entryPointId == -1) return Collections.emptyList();
        ef = Math.max(ef, k);
        List<Integer> eps = new ArrayList<>();
        eps.add(entryPointId);
        for (int layer = maxLayerSeen; layer > 0; layer--)
            eps = greedySearch(query, eps, 1, layer);
        List<Integer> cands = beamSearch(query, eps, ef, 0);
        return cands.stream()
            .map(id -> SearchResult.of(nodes.get(id), dist(query, nodes.get(id).vector)))
            .sorted().limit(k).toList();
    }

    public int size() { return nodes.size(); }
    public boolean contains(String externalId) { return externalToInternal.containsKey(externalId); }
    public HnswConfig getConfig() { return config; }
    public Collection<HnswNode> getAllNodes() { return nodes.values(); }

    public void restoreFromNodes(List<HnswNode> nodeList, int restoredEntryPoint, int restoredMaxLayer) {
        for (HnswNode node : nodeList) {
            nodes.put(node.id, node);
            externalToInternal.put(node.externalId, node.id);
            nodeCounter.set(Math.max(nodeCounter.get(), node.id + 1));
        }
        this.entryPointId = restoredEntryPoint;
        this.maxLayerSeen = restoredMaxLayer;
        log.info("Restored {} nodes, entryPoint={}, maxLayer={}", nodeList.size(), restoredEntryPoint, restoredMaxLayer);
    }

    private int assignLayer() {
        return (int) Math.floor(-Math.log(ThreadLocalRandom.current().nextDouble()) * config.mL());
    }

    private float dist(float[] a, float[] b) {
        return config.distanceFunction().distance(a, b);
    }

    private List<Integer> greedySearch(float[] q, List<Integer> eps, int num, int layer) {
        Set<Integer> visited = new HashSet<>(eps);
        PriorityQueue<float[]> cands = new PriorityQueue<>(Comparator.comparingDouble(a -> a[1]));
        PriorityQueue<float[]> res = new PriorityQueue<>((a, b) -> Float.compare(b[1], a[1]));
        for (int ep : eps) {
            float d = dist(q, nodes.get(ep).vector);
            cands.offer(new float[]{ep, d});
            res.offer(new float[]{ep, d});
        }
        while (!cands.isEmpty()) {
            float[] cur = cands.poll();
            int cId = (int) cur[0];
            if (!res.isEmpty() && res.size() >= num && cur[1] > res.peek()[1]) break;
            for (int nId : neighborsAt(nodes.get(cId), layer)) {
                if (visited.add(nId)) {
                    float nd = dist(q, nodes.get(nId).vector);
                    cands.offer(new float[]{nId, nd});
                    res.offer(new float[]{nId, nd});
                    while (res.size() > num) res.poll();
                }
            }
        }
        return res.stream().map(r -> (int) r[0]).toList();
    }

    private List<Integer> beamSearch(float[] q, List<Integer> eps, int ef, int layer) {
        Set<Integer> visited = new HashSet<>(eps);
        PriorityQueue<float[]> cands = new PriorityQueue<>(Comparator.comparingDouble(a -> a[1]));
        PriorityQueue<float[]> res = new PriorityQueue<>((a, b) -> Float.compare(b[1], a[1]));
        for (int ep : eps) {
            float d = dist(q, nodes.get(ep).vector);
            cands.offer(new float[]{ep, d});
            res.offer(new float[]{ep, d});
        }
        while (!cands.isEmpty()) {
            float[] cur = cands.poll();
            int cId = (int) cur[0];
            if (!res.isEmpty() && res.size() >= ef && cur[1] > res.peek()[1]) break;
            for (int nId : neighborsAt(nodes.get(cId), layer)) {
                if (visited.add(nId)) {
                    float nd = dist(q, nodes.get(nId).vector);
                    if (res.size() < ef || nd < res.peek()[1]) {
                        cands.offer(new float[]{nId, nd});
                        res.offer(new float[]{nId, nd});
                        if (res.size() > ef) res.poll();
                    }
                }
            }
        }
        return res.stream().map(r -> (int) r[0]).toList();
    }

    private List<Integer> nearest(float[] q, List<Integer> cands, int m) {
        return cands.stream()
            .sorted(Comparator.comparingDouble(id -> dist(q, nodes.get(id).vector)))
            .limit(m).toList();
    }

    private List<Integer> neighborsAt(HnswNode node, int layer) {
        if (layer > node.maxLayer) return Collections.emptyList();
        return node.getNeighbors(layer);
    }

    public IndexStats stats() {
        Map<Integer, Integer> dist = new TreeMap<>();
        for (HnswNode n : nodes.values()) dist.merge(n.maxLayer, 1, Integer::sum);
        return new IndexStats(nodes.size(), maxLayerSeen, entryPointId, dist);
    }

    public record IndexStats(int totalNodes, int maxLayer, int entryPointId,
                             Map<Integer, Integer> layerDistribution) {}
}
