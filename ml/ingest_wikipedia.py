"""
Bulk ingest Wikipedia articles into Vektr.
"""
import urllib.request
import urllib.parse
import json
import time

VEKTR_URL = "http://localhost:8080"

ARTICLES = [
    "Random_forest",
    "K-nearest_neighbors_algorithm",
    "Principal_component_analysis",
    "Attention_(machine_learning)",
    "Reinforcement_learning",
    "Natural_language_processing",
    "Nearest_neighbor_search",
    "Approximate_nearest_neighbor_searching",
    "Vector_database",
    "Semantic_search",
    "Information_retrieval",
    "TF-IDF",
    "Cosine_similarity",
    "Euclidean_distance",
    "Neural_network_(machine_learning)",
    "Deep_learning",
    "Generative_artificial_intelligence",
    "Prompt_engineering",
    "Fine-tuning_(deep_learning)",
    "Embedding_(machine_learning)",
]

def fetch_wikipedia(title):
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(title)}"
    req = urllib.request.Request(url, headers={"User-Agent": "vektr-research/1.0 sameer"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())
                return data.get("extract", "")
        except Exception as e:
            if "429" in str(e):
                wait = 10 * (attempt + 1)
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  Error: {e}")
                return ""
    return ""

def ingest(doc_id, text):
    body = json.dumps({"doc_id": doc_id, "text": text}).encode()
    req = urllib.request.Request(
        f"{VEKTR_URL}/ingest",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read())
    except Exception as e:
        print(f"  Ingest failed: {e}")
        return None

def main():
    print(f"Ingesting {len(ARTICLES)} more Wikipedia articles...\n")
    total_chunks = 0
    total_vectors = 0
    failed = 0

    for i, title in enumerate(ARTICLES):
        print(f"[{i+1}/{len(ARTICLES)}] {title}")
        text = fetch_wikipedia(title)
        if not text or len(text) < 100:
            print(f"  Skipped")
            failed += 1
            time.sleep(3)
            continue

        print(f"  Fetched {len(text)} chars")
        doc_id = "wiki-" + title.lower().replace("(","").replace(")","").replace(" ","-").replace("_","-")
        result = ingest(doc_id, text)

        if result:
            chunks = result.get("chunks_indexed", 0)
            total = result.get("total_vectors", 0)
            total_chunks += chunks
            total_vectors = total
            print(f"  Indexed {chunks} chunks | total vectors: {total}")
        else:
            failed += 1

        time.sleep(2)  # be polite to Wikipedia

    print(f"\nDone! Articles: {len(ARTICLES)-failed}/{len(ARTICLES)}")
    print(f"Total chunks this run: {total_chunks}")
    print(f"Total vectors in index: {total_vectors}")

if __name__ == "__main__":
    main()
