from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import time, logging, os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = FastAPI()
MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME, device="cpu")
DIMENSION = model.get_sentence_embedding_dimension()
log.info(f"Ready. dim={DIMENSION}")

class EmbedRequest(BaseModel):
    texts: list[str]
    normalize: bool = True

@app.post("/embed")
def embed(request: EmbedRequest):
    start = time.perf_counter()
    vecs = model.encode(request.texts, normalize_embeddings=request.normalize, show_progress_bar=False)
    ms = (time.perf_counter() - start) * 1000
    log.info(f"Embedded {len(request.texts)} in {ms:.0f}ms")
    return {"embeddings": vecs.tolist(), "dimension": DIMENSION, "count": len(request.texts), "latency_ms": round(ms,2)}

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "dimension": DIMENSION}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, workers=1)
