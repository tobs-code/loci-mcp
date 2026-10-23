import logging
import os
import threading

import torch
from fastapi import FastAPI
from llm import classify, load_config
from pydantic import BaseModel
from sentence_transformers import CrossEncoder, SentenceTransformer

logger = logging.getLogger(__name__)

app = FastAPI(title="Inference Server")

RERANK_MODEL = os.environ.get(
    "RERANK_MODEL", "mixedbread-ai/mxbai-rerank-base-v1"
).strip() or "mixedbread-ai/mxbai-rerank-base-v1"
try:
    RERANK_MAX_LENGTH = int(os.environ.get("RERANK_MAX_LENGTH", "256"))
except ValueError:
    RERANK_MAX_LENGTH = 256
RERANK_MAX_LENGTH = max(64, min(512, RERANK_MAX_LENGTH))

EMBED_MODEL = os.environ.get(
    "EMBED_MODEL", "jinaai/jina-embeddings-v5-text-nano-retrieval"
).strip() or "jinaai/jina-embeddings-v5-text-nano-retrieval"
DEVICE_REQUEST = os.environ.get("INFERENCE_DEVICE", "auto").strip().lower() or "auto"


def _resolve_device(request: str) -> torch.device:
    """Pick CUDA when it is usable, otherwise fall back to CPU.

    A CPU-only torch build has no CUDA, a container started without a GPU
    device sees no CUDA, and a host without a driver reports no device either;
    all three take the CPU path. `INFERENCE_DEVICE=cpu` forces it, `cuda`
    requests it explicitly and still falls back if the request cannot be
    fulfilled.
    """
    if request in {"", "auto"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        device = torch.device(request)
    except (RuntimeError, ValueError):
        logger.warning(
            "unknown INFERENCE_DEVICE=%r; using auto (CUDA if available)",
            request,
        )
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        logger.warning(
            "INFERENCE_DEVICE=%r requested but CUDA is unavailable; using CPU",
            request,
        )
        return torch.device("cpu")
    return device


def _prefer_aten_bmm() -> None:
    """Keep CUDA matmul on ATen instead of torch's Triton native kernel.

    torch 2.14 routes batched outer products (used by the Jina RoPE code) to a
    Triton kernel that compiles a C extension on first use. The runtime image
    has no compiler, so the eager ATen/cuBLAS path is both sufficient and
    startup-safe; disabling the override keeps CUDA and CPU behaviour aligned.
    """
    if not torch.cuda.is_available():
        return
    try:
        from torch._native import registry

        registry.deregister_op_overrides(disable_op_symbols=["bmm"])
    except Exception as exc:
        logger.info("native bmm overrides left enabled: %s", exc)


_prefer_aten_bmm()
DEVICE = _resolve_device(DEVICE_REQUEST)


def _load_embedder(device: torch.device) -> SentenceTransformer:
    kwargs = {"trust_remote_code": True, "device": device}
    if device.type == "cuda":
        # The model config asks for bfloat16, which Turing (RTX 20xx) lacks
        # native support for. float32 is exact and still far faster than CPU.
        kwargs["model_kwargs"] = {"dtype": torch.float32}
    try:
        return SentenceTransformer(EMBED_MODEL, **kwargs)
    except Exception as exc:
        if device.type != "cuda":
            raise
        logger.warning(
            "embedder failed on %s (%s); falling back to CPU", device, exc,
        )
        kwargs["device"] = "cpu"
        kwargs.pop("model_kwargs", None)
        return SentenceTransformer(EMBED_MODEL, **kwargs)


embedder = _load_embedder(DEVICE)
ACTIVE_DEVICE = embedder.device
llm = load_config()
logger.info(
    "inference device: requested=%s active=%s cuda_available=%s",
    DEVICE_REQUEST, ACTIVE_DEVICE, torch.cuda.is_available(),
)

# The reranker (184M parameters) is loaded on first use rather than at import:
# an NLI-only deployment should not pay for it, and a broken model download
# must not prevent the embedding and NLI endpoints from starting. The predict
# lock serialises inference because a CrossEncoder is not re-entrant, and CPU
# inference of one batch at a time is what the latency measurements assume.
_reranker: CrossEncoder | None = None
_reranker_error: str | None = None
_reranker_load_lock = threading.Lock()
_reranker_predict_lock = threading.Lock()


def _get_reranker() -> CrossEncoder:
    global _reranker, _reranker_error
    if _reranker is not None:
        return _reranker
    with _reranker_load_lock:
        if _reranker is None:
            try:
                # float32 is deliberate on CPU: sentence-transformers would
                # pick float16 there, which measured ~5x slower. On CUDA it
                # keeps scores identical to the CPU path and still fits easily
                # in 8 GB next to the embedder.
                _reranker = CrossEncoder(
                    RERANK_MODEL,
                    max_length=RERANK_MAX_LENGTH,
                    device=ACTIVE_DEVICE,
                    model_kwargs={"dtype": torch.float32},
                )
                _reranker_error = None
            except Exception as exc:
                if ACTIVE_DEVICE.type != "cuda":
                    _reranker_error = f"{type(exc).__name__}: {exc}"
                    raise
                logger.warning(
                    "reranker failed on %s (%s); falling back to CPU",
                    ACTIVE_DEVICE, exc,
                )
                try:
                    _reranker = CrossEncoder(
                        RERANK_MODEL,
                        max_length=RERANK_MAX_LENGTH,
                        device="cpu",
                        model_kwargs={"dtype": torch.float32},
                    )
                    _reranker_error = None
                except Exception as cpu_exc:
                    _reranker_error = f"{type(cpu_exc).__name__}: {cpu_exc}"
                    raise
    return _reranker


class EmbedRequest(BaseModel):
    texts: list[str]
    mode: str = "document"  # "query" or "document"

class EmbedResponse(BaseModel):
    embeddings: list[list[float]]
    dimension: int


class RerankRequest(BaseModel):
    query: str
    documents: list[str]


class RerankResponse(BaseModel):
    # Raw cross-encoder logits, one per document. They are not probabilities
    # and only meaningful relative to the other documents in this request.
    scores: list[float]
    error: str | None = None


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
    # Indices of the documents whose score is a placeholder (0.0). A caller
    # can name exactly which pairs were NOT judged instead of guessing from
    # the count alone (older clients ignore the field).
    failed_indices: list[int] = []


@app.get("/health")
def health():
    return {
        "status": "ok",
        "llm_provider": llm.provider,
        "llm_model": llm.model,
        "llm_configured": llm.configured,
        "device_requested": DEVICE_REQUEST,
        "device": str(ACTIVE_DEVICE),
        "embedder_device": str(embedder.device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": (torch.cuda.get_device_name(0)
                        if torch.cuda.is_available() else None),
        "torch_version": torch.__version__,
        "rerank_model": RERANK_MODEL,
        "rerank_loaded": _reranker is not None,
        "reranker_device": (str(_reranker.model.device)
                            if _reranker is not None else None),
        "rerank_error": _reranker_error,
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


@app.post("/rerank", response_model=RerankResponse)
def rerank(req: RerankRequest):
    if not req.documents:
        return RerankResponse(scores=[])
    if not req.query.strip():
        return RerankResponse(scores=[], error="query is empty")
    try:
        model = _get_reranker()
        with _reranker_predict_lock:
            scores = model.predict(
                [(req.query, document) for document in req.documents]
            )
    except Exception as exc:
        # A reranker outage must not turn into an empty answer: the caller
        # falls back to the first-stage ranking and surfaces the warning.
        logger.warning("rerank failed: %s", exc)
        return RerankResponse(scores=[], error=f"rerank unavailable: {exc}")
    return RerankResponse(scores=[float(score) for score in scores])


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
    failure_indices: list[int] = []
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
            failure_indices.append(index)

    if failures and len(failures) == len(req.documents):
        # Nothing was judged: this is an outage, not a clean scan.
        return NliResponse(
            contradiction_scores=scores,
            error=f"NLI unavailable: {failures[0]}",
            failed=len(failures),
            failed_indices=failure_indices,
        )
    partial_error = None
    if failures:
        partial_error = (
            f"NLI unavailable for {len(failures)} of {len(req.documents)} "
            f"document(s): {failures[0]}"
        )
    return NliResponse(contradiction_scores=scores,
                       partial_error=partial_error,
                       failed=len(failures),
                       failed_indices=failure_indices)
