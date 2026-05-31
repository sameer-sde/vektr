package com.vektr.index;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import java.util.*;
import static org.assertj.core.api.Assertions.*;

class HnswIndexTest {
    private HnswIndex index;
    private static final int DIM = 128;
    private static final Random RAND = new Random(42);

    @BeforeEach
    void setUp() { index = new HnswIndex(HnswConfig.defaults()); }

    @Test
    void insert_singleVector_canBeFound() {
        float[] vec = unitVector(DIM);
        index.insert("doc-1", vec);
        List<SearchResult> results = index.search(vec, 1);
        assertThat(results).hasSize(1);
        assertThat(results.get(0).externalId()).isEqualTo("doc-1");
        assertThat(results.get(0).distance()).isLessThan(1e-4f);
    }

    @Test
    void insert_duplicateId_throws() {
        index.insert("doc-1", unitVector(DIM));
        assertThatThrownBy(() -> index.insert("doc-1", unitVector(DIM)))
            .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void search_emptyIndex_returnsEmpty() {
        assertThat(index.search(unitVector(DIM), 5)).isEmpty();
    }

    @Test
    void search_kLargerThanSize_returnsAll() {
        for (int i = 0; i < 5; i++) index.insert("doc-" + i, unitVector(DIM));
        assertThat(index.search(unitVector(DIM), 100)).hasSize(5);
    }

    @Test
    void search_resultsOrderedByDistance() {
        for (int i = 0; i < 100; i++) index.insert("doc-" + i, unitVector(DIM));
        List<SearchResult> r = index.search(unitVector(DIM), 10);
        for (int i = 1; i < r.size(); i++)
            assertThat(r.get(i).distance()).isGreaterThanOrEqualTo(r.get(i-1).distance());
    }

    @Test
    void recall_atK10_exceeds90pct() {
        int n = 1000, k = 10, queries = 50;
        float[][] corpus = new float[n][DIM];
        for (int i = 0; i < n; i++) {
            corpus[i] = unitVector(DIM);
            index.insert("doc-" + i, corpus[i]);
        }
        double totalRecall = 0;
        HnswConfig cfg = HnswConfig.defaults();
        for (int q = 0; q < queries; q++) {
            float[] query = unitVector(DIM);
            Set<String> gt = bruteForce(query, corpus, k, cfg.distanceFunction());
            List<SearchResult> res = index.search(query, k);
            Set<String> found = new HashSet<>();
            for (SearchResult r : res) found.add(r.externalId());
            found.retainAll(gt);
            totalRecall += (double) found.size() / k;
        }
        double recall = totalRecall / queries;
        System.out.printf("Recall@%d over %d queries: %.4f%n", k, queries, recall);
        assertThat(recall).as("Recall@10 must exceed 0.90").isGreaterThan(0.90);
    }

    private Set<String> bruteForce(float[] query, float[][] corpus, int k, DistanceFunction df) {
        PriorityQueue<float[]> pq = new PriorityQueue<>((a, b) -> Float.compare(b[1], a[1]));
        for (int i = 0; i < corpus.length; i++) {
            pq.offer(new float[]{i, df.distance(query, corpus[i])});
            if (pq.size() > k) pq.poll();
        }
        Set<String> result = new HashSet<>();
        while (!pq.isEmpty()) result.add("doc-" + (int) pq.poll()[0]);
        return result;
    }

    private float[] unitVector(int dim) {
        float[] v = new float[dim];
        float norm = 0f;
        for (int i = 0; i < dim; i++) { v[i] = (float) RAND.nextGaussian(); norm += v[i]*v[i]; }
        norm = (float) Math.sqrt(norm);
        for (int i = 0; i < dim; i++) v[i] /= norm;
        return v;
    }
}
