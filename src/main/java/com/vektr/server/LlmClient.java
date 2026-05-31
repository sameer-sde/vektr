package com.vektr.server;

import com.fasterxml.jackson.databind.ObjectMapper;
import okhttp3.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Map;
import java.util.concurrent.TimeUnit;

public class LlmClient {
    private static final Logger log = LoggerFactory.getLogger(LlmClient.class);
    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final MediaType JSON = MediaType.get("application/json");
    private final String baseUrl;
    private final String model;
    private final OkHttpClient http;

    public LlmClient(String baseUrl, String model) {
        this.baseUrl = baseUrl;
        this.model = model;
        this.http = new OkHttpClient.Builder()
            .connectTimeout(10, TimeUnit.SECONDS)
            .readTimeout(120, TimeUnit.SECONDS).build();
    }

    public static LlmClient defaultClient() {
        return new LlmClient("http://localhost:11434", "llama3.2");
    }

    public void streamCompletion(String prompt, TokenCallback onToken, Runnable onComplete)
            throws IOException {
        String body = MAPPER.writeValueAsString(Map.of("model", model, "prompt", prompt, "stream", true));
        Request req = new Request.Builder().url(baseUrl + "/api/generate")
            .post(RequestBody.create(body, JSON)).build();
        try (Response resp = http.newCall(req).execute()) {
            if (!resp.isSuccessful()) throw new IOException("LLM error: " + resp.code());
            try (BufferedReader reader = new BufferedReader(
                    new InputStreamReader(resp.body().byteStream()))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    if (line.isBlank()) continue;
                    @SuppressWarnings("unchecked")
                    Map<String, Object> chunk = MAPPER.readValue(line, Map.class);
                    String token = (String) chunk.getOrDefault("response", "");
                    boolean done = Boolean.TRUE.equals(chunk.get("done"));
                    if (!token.isEmpty()) {
                        try {
                            onToken.onToken(token);
                        } catch (IOException e) {
                            throw e;
                        } catch (Exception e) {
                            throw new IOException("Token callback error", e);
                        }
                    }
                    if (done) break;
                }
            }
        }
        onComplete.run();
    }

    @FunctionalInterface
    public interface TokenCallback {
        void onToken(String token) throws IOException;
    }
}
