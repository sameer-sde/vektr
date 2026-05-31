package com.vektr.pipeline;

import java.util.ArrayList;
import java.util.List;

public class DocumentChunker {
    private final int targetTokens;
    private final int overlapSentences;
    private final int minChunkTokens;

    public DocumentChunker() { this(256, 2, 5); }
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
                String s = sentences.get(si);
                cs.add(s);
                tokens += wordCount(s);
                si++;
            }
            String txt = String.join(" ", cs).strip();
            if (wordCount(txt) >= minChunkTokens) {
                chunks.add(new Chunk(docId + "_chunk_" + ci, docId, txt, ci, wordCount(txt)));
                ci++;
            }
            if (cs.isEmpty()) break;
            si = Math.max(si - overlapSentences, si - cs.size() + 1);
        }
        return chunks;
    }

    private List<String> splitSentences(String text) {
        List<String> out = new ArrayList<>();
        int start = 0;
        for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            if (c == '.' || c == '!' || c == '?') {
                String s = text.substring(start, i + 1).strip();
                if (!s.isEmpty()) out.add(s);
                start = i + 1;
            }
        }
        String remaining = text.substring(start).strip();
        if (!remaining.isEmpty()) out.add(remaining);
        if (out.isEmpty()) out.add(text.strip());
        return out;
    }

    private int wordCount(String text) {
        if (text == null || text.isBlank()) return 0;
        int count = 0;
        boolean inWord = false;
        for (int i = 0; i < text.length(); i++) {
            if (Character.isWhitespace(text.charAt(i))) {
                inWord = false;
            } else if (!inWord) {
                count++;
                inWord = true;
            }
        }
        return count;
    }

    public record Chunk(String id, String sourceDocId, String text, int chunkIndex, int tokenCount) {}
}
