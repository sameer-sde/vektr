package com.vektr.query;

import com.fasterxml.jackson.databind.ObjectMapper;
import okhttp3.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.io.IOException;
import java.util.Map;
import java.util.concurrent.TimeUnit;

/**
 * HyDE: Hypothetical Document Embeddings.
 *
 * Paper: "Precise Zero-Shot Dense Retrieval without Relevance Labels"
 *         Gao et al., 2022 (Google Brain / CMU)
 * https://arxiv.org/abs/2212.10496
 *
 * The problem with raw query embedding:
 *   Query: "what causes inflation"  (8 tokens, vague)
 *   Documents: long detailed paragraphs about monetary policy
 *   → The query embedding sits far from document embeddings in vector space
 *     because they're stylistically different (question vs answer)
 *
 * HyDE solution:
 *   1. Ask the LLM: "Write a short answer to: what causes inflation"
 *   2. LLM generates: "Inflation is caused by excess money supply,
 *      supply chain disruptions, and rising demand..."
 *   3. Embed THAT hypothetical answer instead of the raw query
 *   4. The embedding now lives in "answer space" — much closer to real docs
 *
 * Result: +8-15% recall improvement on short/vague queries.
 * Works best when: query is short, documents are long and detailed.
 * Costs: one LLM call per query (~50-200ms with local Ollama).
 *
 * This is a real technique used in production RAG systems at Google,
 * Cohere, and others. Knowing it by name in interviews is a signal.
 */
public class QueryRewriter {
    private static final Logger log = LoggerFactory.getLogger(QueryRewriter.class);
    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final MediaType JSON = MediaType.get("application/json");

    private final String llmBaseUrl;
    private final String model;
    private final OkHttpClient http;
    private final boolean enabled;

    public QueryRewriter(String llmBaseUrl, String model, boolean enabled) {
        this.llmBaseUrl = llmBaseUrl;
        this.model = model;
        this.enabled = enabled;
        this.http = new OkHttpClient.Builder()
            .connectTimeout(5, TimeUnit.SECONDS)
            .readTimeout(30, TimeUnit.SECONDS).build();
    }

    public static QueryRewriter defaultRewriter() {
        return new QueryRewriter("http://localhost:11434", "llama3.2", true);
    }

    public static QueryRewriter disabled() {
        return new QueryRewriter("", "", false);
    }

    /**
     * Rewrite a query using HyDE.
     *
     * If Ollama is unavailable or rewriting is disabled, returns the
     * original query unchanged — graceful degradation, not failure.
     *
     * @param originalQuery the raw user query
     * @return hypothetical answer text (to be embedded), or original query on failure
     */
    public String rewrite(String originalQuery) {
        if (!enabled) return originalQuery;

        try {
            String hypothetical = generateHypotheticalAnswer(originalQuery);
            log.debug("HyDE rewrite: '{}' -> '{}'", originalQuery,
                hypothetical.substring(0, Math.min(80, hypothetical.length())));
            return hypothetical;
        } catch (Exception e) {
            log.warn("HyDE rewrite failed, using original query: {}", e.getMessage());
            return originalQuery; // graceful fallback
        }
    }

    /**
     * Call the LLM to generate a hypothetical answer.
     * Non-streaming: we need the full answer before embedding.
     */
    private String generateHypotheticalAnswer(String query) throws IOException {
        String prompt = """
            Write a concise factual paragraph that directly answers this question.
            Do not include the question itself. Answer only, 2-4 sentences max.
            
            Question: %s
            
            Answer:""".formatted(query);

        String body = MAPPER.writeValueAsString(Map.of(
            "model", model,
            "prompt", prompt,
            "stream", false,
            "options", Map.of("num_predict", 150, "temperature", 0.3)
        ));

        Request req = new Request.Builder()
            .url(llmBaseUrl + "/api/generate")
            .post(RequestBody.create(body, JSON)).build();

        try (Response resp = http.newCall(req).execute()) {
            if (!resp.isSuccessful()) throw new IOException("LLM error: " + resp.code());
            @SuppressWarnings("unchecked")
            Map<String, Object> r = MAPPER.readValue(resp.body().string(), Map.class);
            String answer = (String) r.getOrDefault("response", "");
            if (answer.isBlank()) throw new IOException("Empty LLM response");
            return answer.trim();
        }
    }

    public boolean isEnabled() { return enabled; }
}
