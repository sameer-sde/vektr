package com.vektr.index;

public record HnswConfig(int m, int efConstruction, int efSearch,
                         DistanceFunction distanceFunction, int maxElements) {
    public static final int DEFAULT_M = 16;
    public static final int DEFAULT_EF_CONSTRUCTION = 200;
    public static final int DEFAULT_EF_SEARCH = 50;

    public double mL() { return 1.0 / Math.log(m); }
    public int mMax0() { return 2 * m; }

    public static HnswConfig defaults() {
        return new HnswConfig(DEFAULT_M, DEFAULT_EF_CONSTRUCTION, DEFAULT_EF_SEARCH,
                DistanceFunction.COSINE, 1_000_000);
    }

    public static Builder builder() { return new Builder(); }

    public static class Builder {
        private int m = DEFAULT_M;
        private int efConstruction = DEFAULT_EF_CONSTRUCTION;
        private int efSearch = DEFAULT_EF_SEARCH;
        private DistanceFunction distanceFunction = DistanceFunction.COSINE;
        private int maxElements = 1_000_000;

        public Builder m(int m) { this.m = m; return this; }
        public Builder efConstruction(int ef) { this.efConstruction = ef; return this; }
        public Builder efSearch(int ef) { this.efSearch = ef; return this; }
        public Builder distanceFunction(DistanceFunction df) { this.distanceFunction = df; return this; }
        public Builder maxElements(int max) { this.maxElements = max; return this; }

        public HnswConfig build() {
            if (efConstruction < m)
                throw new IllegalArgumentException("efConstruction must be >= M");
            return new HnswConfig(m, efConstruction, efSearch, distanceFunction, maxElements);
        }
    }
}
