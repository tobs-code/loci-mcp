import logging

from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from llm import classify, load_config

logger = logging.getLogger(__name__)

app = FastAPI(title="Inference Server")

embedder = SentenceTransformer(
    "jinaai/jina-embeddings-v5-text-nano-retrieval",
    trust_remote_code=True,
)
llm = load_config()


class EmbedRequest(BaseModel):
    texts: list[str]
    mode: str = "document"  # "query" or "document"

class EmbedResponse(BaseModel):
    embeddings: list[list[float]]
    dimension: int

class NliRequest(BaseModel):
    query: str
    documents: list[str]

class NliResponse(BaseModel):
    contradiction_scores: list[float]
    # Set when the scores are NOT a real judgment (provider down, not
    # configured): the caller must not mistake zeros for "no contradiction".
    error: str | None = None


@app.get("/health")
def health():
    return {
        "status": "ok",
        "llm_provider": llm.provider,
        "llm_model": llm.model,
        "llm_configured": llm.configured,
    }


@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest):
    # jina-embeddings-v5 is an asymmetric retrieval model: queries and
    # documents must be encoded with their own prompt, otherwise the
    # similarity scores collapse (paraphrases land around 0.2 instead of 0.8).
    prompt_name = "query" if req.mode == "query" else "document"
    try:
        vectors = embedder.encode(
            req.texts, normalize_embeddings=True, prompt_name=prompt_name
        )
    except (ValueError, KeyError):
        # Model without prompt presets: fall back to plain encoding.
        vectors = embedder.encode(req.texts, normalize_embeddings=True)
    embeddings = vectors.tolist()
    return EmbedResponse(
        embeddings=embeddings,
        dimension=len(embeddings[0]) if embeddings else 0,
    )


@app.post("/nli", response_model=NliResponse)
def nli(req: NliRequest):
    if not llm.configured:
        logger.warning(
            "NLI disabled: LLM provider %r is not configured", llm.provider,
        )
        return NliResponse(contradiction_scores=[0.0] * len(req.documents),
                           error=f"LLM provider {llm.provider!r} is not configured")

    scores = []
    for idx, doc in enumerate(req.documents):
        try:
            scores.append(classify(llm, req.query, doc))
        except Exception as exc:
            # Provider down / invalid key: don't silently claim "no
            # contradiction" for the remaining documents — abort early and
            # carry the failure back to the caller so it can surface it.
            logger.warning(
                "NLI unavailable, returning 0.0 for remaining documents: %s",
                exc,
            )
            scores.extend([0.0] * (len(req.documents) - idx))
            return NliResponse(contradiction_scores=scores,
                               error=f"NLI unavailable: {exc}")

    return NliResponse(contradiction_scores=scores)
