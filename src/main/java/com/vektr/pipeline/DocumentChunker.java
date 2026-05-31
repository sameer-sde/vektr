package com.vektr.pipeline;

import java.util.ArrayList;
import java.util.List;

public class DocumentChunker {
    private final int targetTokens;
    private final int overlapSentences;
    private final int minChunkTokens;

    public DocumentChunker() { this(256, 2, 20); }
    public DocumentChunker(int targetTokens, int overlapSentences, int minChunkTokens) {
        this.targetTokens = targetTokens;
        this.overlapSentences = overlapSentences;
        this.minChunkTokens = minChunkTokens;
    }

    public List<Chunk> chunk(String docId, String text) {
        if (text == null || text.isBlank()) return List.of();
        List<String> sentences = splitSentences(text);
        List<Chunk> chunks = new ArrayList<>();
        int si = 0, ci = 0;
        while (si < sentences.size()) {
            List<String> cs = new ArrayList<>();
            int tokens = 0;
            while (si < sentences.size() && tokens < targetTokens) {
                cs.add(sentences.get(si));
                tokens += approxTokens(sentences.get(si));
                si++;
            }
            String txt = String.join(" ", cs).strip();
            if (approxTokens(txt) >= minChunkTokens)
                chunks.add(new Chunk(docId + "_chunk_" + ci++, docId, txt, ci, approxTokens(txt)));
            si = Math.max(0, si - overlapSentences);
            if (si == 0 && ci > 0) si = cs.size();
        }
        return chunks;
    }

    private List<String> splitSentences(String text) {
        List<String> out = new ArrayList<>();
        for (String s : text.split("(?<=[.!?])\\s+"))
            if (!s.isBlank()) out.add(s.strip());
        return out;
    }

    private int approxTokens(String text) {
        if (text == null || text.isBlank()) return 0;
        return (int)(text.split("\\s+").length * 1.3);
    }

    public record Chunk(String id, String sourceDocId, String text, int chunkIndex, int tokenCount) {}
}
