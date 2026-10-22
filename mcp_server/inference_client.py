import logging
import os
import requests

logger = logging.getLogger(__name__)

INFERENCE_URL = os.environ.get(
    "MEMORY_INFERENCE_URL",
    "http://localhost:8001",
)

_session = None


def _get_session():
    global _session
    if _session is None:
        _session = requests.Session()
    return _session


def embed(texts: list[str], mode: str = "document") -> list[list[float]]:
    """Embed texts. mode="query" for search queries, "document" for stored content."""
    r = _get_session().post(
        f"{INFERENCE_URL}/embed",
        json={"texts": texts, "mode": mode},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["embeddings"]


def nli_detailed(query: str,
                 documents: list[str]) -> tuple[list[float], str | None]:
    """Return (scores, partial_error).

    Raises when no document could be judged (outage). `partial_error` is set
    when some documents failed: their score is 0.0 and must not be treated
    as a clean "neutral" judgment.
    """
    if not documents:
        return [], None
    resp = _get_session().post(
        f"{INFERENCE_URL}/nli",
        json={"query": query, "documents": documents},
        timeout=60,
    )
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("error"):
        # The scores are zeros from a failed run, not a real judgment:
        # callers must surface this instead of trusting 0.0 as "neutral".
        raise RuntimeError(payload["error"])
    return payload["contradiction_scores"], payload.get("partial_error")


def nli(query: str, documents: list[str]) -> list[float]:
    scores, partial = nli_detailed(query, documents)
    if partial:
        logger.warning("%s", partial)
    return scores


def health() -> bool:
    return health_details() is not None


def health_details() -> dict | None:
    try:
        resp = _get_session().get(f"{INFERENCE_URL}/health", timeout=5)
        if resp.status_code != 200:
            return None
        payload = resp.json()
        return payload if isinstance(payload, dict) else {}
    except requests.RequestException:
        return None


def nli_configured() -> bool | None:
    details = health_details()
    if details is None or "llm_configured" not in details:
        return None
    return bool(details["llm_configured"])
