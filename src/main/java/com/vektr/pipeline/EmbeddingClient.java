package com.vektr.pipeline;

import com.fasterxml.jackson.databind.ObjectMapper;
import okhttp3.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

public class EmbeddingClient {
    private static final Logger log = LoggerFactory.getLogger(EmbeddingClient.class);
    private static final MediaType JSON = MediaType.get("application/json");
    private final String baseUrl;
    private final OkHttpClient http;
    private final ObjectMapper mapper = new ObjectMapper();

    public EmbeddingClient(String baseUrl) {
        this.baseUrl = baseUrl.endsWith("/") ? baseUrl.substring(0, baseUrl.length()-1) : baseUrl;
        this.http = new OkHttpClient.Builder()
            .connectTimeout(10, TimeUnit.SECONDS)
            .readTimeout(60, TimeUnit.SECONDS)
            .writeTimeout(60, TimeUnit.SECONDS)
            .connectionPool(new ConnectionPool(10, 5, TimeUnit.MINUTES)).build();
    }

    public static EmbeddingClient defaultClient() { return new EmbeddingClient("http://localhost:8001"); }

    public float[] embed(String text) throws IOException { return embedBatch(List.of(text)).get(0); }

    public List<float[]> embedBatch(List<String> texts) throws IOException {
        if (texts.isEmpty()) return List.of();
        String body = mapper.writeValueAsString(Map.of("texts", texts, "normalize", true));
        Request req = new Request.Builder().url(baseUrl + "/embed")
            .post(RequestBody.create(body, JSON)).build();
        try (Response resp = http.newCall(req).execute()) {
            if (!resp.isSuccessful()) throw new IOException("Embedding error: " + resp.code());
            @SuppressWarnings("unchecked")
            Map<String, Object> r = mapper.readValue(resp.body().string(), Map.class);
            @SuppressWarnings("unchecked")
            List<List<Double>> raw = (List<List<Double>>) r.get("embeddings");
            return raw.stream().map(this::toFloat).toList();
        }
    }

    public boolean isHealthy() {
        try {
            Request req = new Request.Builder().url(baseUrl + "/health").get().build();
            try (Response r = http.newCall(req).execute()) { return r.isSuccessful(); }
        } catch (IOException e) { return false; }
    }

    private float[] toFloat(List<Double> d) {
        float[] a = new float[d.size()];
        for (int i = 0; i < d.size(); i++) a[i] = d.get(i).floatValue();
        return a;
    }
}
