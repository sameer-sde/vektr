package com.vektr.retrieval;

import com.vektr.index.SearchResult;
import java.util.*;
import java.util.stream.Collectors;

public class ReciprocalRankFusion {
    private final int k;
    public ReciprocalRankFusion() { this(60); }
    public ReciprocalRankFusion(int k) { this.k = k; }

    public List<FusedResult> fuse(List<BM25Index.BM25Result> bm25, List<SearchResult> dense, int topN) {
        Map<String, Double> scores = new HashMap<>();
        Map<String, String> texts = new HashMap<>();
        for (int i = 0; i < bm25.size(); i++) {
            scores.merge(bm25.get(i).docId(), 1.0 / (k + i + 1), Double::sum);
            texts.put(bm25.get(i).docId(), bm25.get(i).text());
        }
        for (int i = 0; i < dense.size(); i++)
            scores.merge(dense.get(i).externalId(), 1.0 / (k + i + 1), Double::sum);
        return scores.entrySet().stream()
            .sorted(Map.Entry.<String, Double>comparingByValue().reversed())
            .limit(topN)
            .map(e -> new FusedResult(e.getKey(), e.getValue(), texts.get(e.getKey())))
            .collect(Collectors.toList());
    }

    public record FusedResult(String docId, double rrfScore, String text) {}
}
