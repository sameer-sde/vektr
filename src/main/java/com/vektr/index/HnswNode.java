package com.vektr.index;

import java.util.*;
import java.util.concurrent.CopyOnWriteArrayList;

public class HnswNode {
    public final int id;
    public final float[] vector;
    public final String externalId;
    public final int maxLayer;
    private final List<List<Integer>> neighbors;

    public HnswNode(int id, float[] vector, String externalId, int maxLayer, int numLayers) {
        this.id = id;
        this.vector = vector;
        this.externalId = externalId;
        this.maxLayer = maxLayer;
        this.neighbors = new ArrayList<>(numLayers);
        for (int i = 0; i < numLayers; i++)
            this.neighbors.add(new CopyOnWriteArrayList<>());
    }

    public List<Integer> getNeighbors(int layer) { return neighbors.get(layer); }

    public void setNeighbors(int layer, List<Integer> n) {
        List<Integer> l = neighbors.get(layer);
        l.clear();
        l.addAll(n);
    }

    public int getDimension() { return vector.length; }
}
