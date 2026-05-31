package com.vektr.retrieval;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

public class BM25Index {
    private final float k1;
    private final float b;
    private final Map<String, Map<String, Integer>> invertedIndex = new ConcurrentHashMap<>();
    private final Map<String, Integer> docLengths = new ConcurrentHashMap<>();
    private final Map<String, String> docTexts = new ConcurrentHashMap<>();
    private volatile long totalTokens = 0;

    public BM25Index() { this(1.5f, 0.75f); }
    public BM25Index(float k1, float b) { this.k1 = k1; this.b = b; }

    public synchronized void addDocument(String docId, String text) {
        List<String> tokens = tokenize(text);
        docLengths.put(docId, tokens.size());
        docTexts.put(docId, text);
        totalTokens += tokens.size();
        Map<String, Integer> tf = new HashMap<>();
        for (String token : tokens) tf.merge(token, 1, Integer::sum);
        for (Map.Entry<String, Integer> e : tf.entrySet())
            invertedIndex.computeIfAbsent(e.getKey(), k -> new ConcurrentHashMap<>()).put(docId, e.getValue());
    }

    public List<BM25Result> search(String query, int k) {
        List<String> terms = tokenize(query);
        if (terms.isEmpty() || docLengths.isEmpty()) return Collections.emptyList();
        double avgdl = (double) totalTokens / docLengths.size();
        int N = docLengths.size();
        Map<String, Double> scores = new HashMap<>();
        for (String term : terms) {
            Map<String, Integer> postings = invertedIndex.getOrDefault(term, Collections.emptyMap());
            if (postings.isEmpty()) continue;
            double idf = Math.log((N - postings.size() + 0.5) / (postings.size() + 0.5) + 1);
            for (Map.Entry<String, Integer> p : postings.entrySet()) {
                int tf = p.getValue();
                int dl = docLengths.get(p.getKey());
                double tfNorm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl));
                scores.merge(p.getKey(), idf * tfNorm, Double::sum);
            }
        }
        return scores.entrySet().stream()
            .sorted(Map.Entry.<String, Double>comparingByValue().reversed())
            .limit(k)
            .map(e -> new BM25Result(e.getKey(), e.getValue(), docTexts.get(e.getKey())))
            .collect(Collectors.toList());
    }

    private List<String> tokenize(String text) {
        if (text == null || text.isBlank()) return Collections.emptyList();
        return Arrays.stream(text.toLowerCase().split("[\\s\\p{Punct}]+"))
            .filter(t -> t.length() > 1).collect(Collectors.toList());
    }

    public int size() { return docLengths.size(); }
    public int vocabularySize() { return invertedIndex.size(); }
    public record BM25Result(String docId, double score, String text) {}
}
