package com.vektr.index;

public enum DistanceFunction {
    COSINE {
        @Override
        public float distance(float[] a, float[] b) {
            float dot = 0f, normA = 0f, normB = 0f;
            for (int i = 0; i < a.length; i++) {
                dot   += a[i] * b[i];
                normA += a[i] * a[i];
                normB += b[i] * b[i];
            }
            if (normA == 0f || normB == 0f) return 1f;
            return 1f - (dot / (float)(Math.sqrt(normA) * Math.sqrt(normB)));
        }
    },
    EUCLIDEAN {
        @Override
        public float distance(float[] a, float[] b) {
            float sum = 0f;
            for (int i = 0; i < a.length; i++) { float d = a[i]-b[i]; sum += d*d; }
            return (float) Math.sqrt(sum);
        }
    },
    DOT {
        @Override
        public float distance(float[] a, float[] b) {
            float dot = 0f;
            for (int i = 0; i < a.length; i++) dot += a[i] * b[i];
            return 1f - dot;
        }
    };
    public abstract float distance(float[] a, float[] b);
}
