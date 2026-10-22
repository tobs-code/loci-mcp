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
    # Optional subject hints: which entity each fact is stored under. They
    # help the LLM to judge whether both facts describe the SAME subject -
    # a differing port of two different services is not a contradiction.
    # Omitting them keeps the comparison text-only.
    query_entity: str | None = None
    document_entities: list[str | None] | None = None

class NliResponse(BaseModel):
    contradiction_scores: list[float]
    # Set when NO score is a real judgment (provider down, not configured):
    # the caller must not mistake the zeros for "no contradiction".
    error: str | None = None
    # Set when only SOME documents failed: the successful scores are real,
    # the failed ones are 0.0 and must not be read as "neutral".
    partial_error: str | None = None
    failed: int = 0


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

    scores: list[float] = []
    failures: list[str] = []
    document_entities = req.document_entities or []
    for index, doc in enumerate(req.documents):
        document_entity = (
            document_entities[index] if index < len(document_entities) else None
        )
        try:
            scores.append(classify(
                llm, req.query, doc,
                query_entity=req.query_entity,
                document_entity=document_entity,
            ))
        except Exception as exc:
            # One malformed provider response must not throw away the
            # judgments for the other documents: record the failure, keep
            # the 0.0 placeholder, and continue. A failed score is reported
            # separately so the caller never reads it as "neutral".
            logger.warning("NLI failed for one document: %s", exc)
            scores.append(0.0)
            failures.append(str(exc))

    if failures and len(failures) == len(req.documents):
        # Nothing was judged: this is an outage, not a clean scan.
        return NliResponse(
            contradiction_scores=scores,
            error=f"NLI unavailable: {failures[0]}",
            failed=len(failures),
        )
    partial_error = None
    if failures:
        partial_error = (
            f"NLI unavailable for {len(failures)} of {len(req.documents)} "
            f"document(s): {failures[0]}"
        )
    return NliResponse(contradiction_scores=scores,
                       partial_error=partial_error,
                       failed=len(failures))
