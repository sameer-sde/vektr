package com.vektr.index;

public record SearchResult(int nodeId, String externalId, float distance, float score)
        implements Comparable<SearchResult> {
    public static SearchResult of(HnswNode node, float distance) {
        return new SearchResult(node.id, node.externalId, distance, 1f - distance);
    }
    @Override
    public int compareTo(SearchResult other) {
        return Float.compare(this.distance, other.distance);
    }
}
