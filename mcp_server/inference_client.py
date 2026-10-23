import logging
import os
import threading
import time

import requests

logger = logging.getLogger(__name__)

INFERENCE_URL = os.environ.get(
    "MEMORY_INFERENCE_URL",
    "http://localhost:8001",
)

_local = threading.local()


def _get_session():
    """One session per thread: `requests.Session` is not thread-safe, and
    `contradictions_report` calls this client from a small worker pool."""
    session = getattr(_local, "session", None)
    if session is None:
        session = requests.Session()
        _local.session = session
    return session


def embed(texts: list[str], mode: str = "document") -> list[list[float]]:
    """Embed texts. mode="query" for search queries, "document" for stored content."""
    r = _get_session().post(
        f"{INFERENCE_URL}/embed",
        json={"texts": texts, "mode": mode},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["embeddings"]


def rerank(query: str, documents: list[str], *, timeout: float = 60.0) -> list[float]:
    """Score query/document pairs with the local cross-encoder.

    Returns one raw logit per document. The values are not probabilities and
    are not comparable across different candidate sets; callers turn them into
    a relative score for the block they submitted.
    """
    if not documents:
        return []
    resp = _get_session().post(
        f"{INFERENCE_URL}/rerank",
        json={"query": query, "documents": documents},
        timeout=timeout,
    )
    resp.raise_for_status()
    try:
        payload = resp.json()
    except ValueError as e:
        raise RuntimeError(f"rerank response was not valid JSON: {e}") from e
    error = payload.get("error")
    if error:
        raise RuntimeError(str(error))
    scores = payload.get("scores")
    if not isinstance(scores, list) or len(scores) != len(documents):
        raise RuntimeError("rerank response did not contain one score per document")
    try:
        return [float(score) for score in scores]
    except (TypeError, ValueError) as e:
        raise RuntimeError(f"rerank response contained a non-numeric score: {e}") from e


def nli_detailed(query: str,
                 documents: list[str],
                 query_entity: str | None = None,
                 document_entities: list[str | None] | None = None,
                 *,
                 timeout: float = 60.0,
                 deadline: float | None = None,
                 ) -> tuple[list[float], str | None, list[int]]:
    """Return (scores, partial_error, failed_indices).

    Raises when no document could be judged (outage). `partial_error` is set
    when some documents failed: their score is 0.0 and must not be treated
    as a clean "neutral" judgment. `failed_indices` names exactly those
    documents, so a caller can report which pairs were not judged (a stale
    inference container that does not send the field yields an empty list,
    which is why `partial_error` stays authoritative).

    The optional entity labels tell the NLI model which subject each fact
    belongs to (older inference containers ignore them silently).

    `timeout` bounds each HTTP request; `deadline` (a `time.monotonic()`
    value) bounds the whole call, including retries. A caller that scans many
    memories uses it to stay inside its own request budget instead of being
    cut off by the transport timeout.
    """
    if not documents:
        return [], None, []
    body: dict = {"query": query, "documents": documents}
    if query_entity is not None:
        body["query_entity"] = query_entity
    if document_entities is not None:
        body["document_entities"] = document_entities
    last_error: str | None = None
    # A provider occasionally returns unparseable JSON. That is a transient
    # model/format hiccup, not an outage: retry twice instead of reporting
    # the whole contradiction scan as incomplete on the first hiccup.
    for attempt in range(3):
        if attempt:
            # A short pause with linear backoff before the retry: the same
            # immediate resend would most likely hit the very same transient
            # state again.
            time.sleep(0.5 * attempt)
        request_timeout = timeout
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("NLI time budget exhausted")
            request_timeout = max(0.5, min(timeout, remaining))
        resp = _get_session().post(
            f"{INFERENCE_URL}/nli",
            json=body,
            timeout=request_timeout,
        )
        resp.raise_for_status()
        try:
            payload = resp.json()
        except ValueError as e:
            last_error = f"response was not valid NLI JSON: {e}"
            if attempt < 2:
                logger.warning("NLI response invalid, retrying: %s",
                               last_error)
                continue
            raise RuntimeError(last_error) from e
        error = payload.get("error")
        scores = payload.get("contradiction_scores")
        if not error:
            if (isinstance(scores, list) and len(scores) == len(documents)):
                failed = payload.get("failed_indices") or []
                # Keep only usable indices: a broken or stale payload must not
                # make a caller name a document that does not exist.
                if not isinstance(failed, list):
                    failed = []
                failed_indices = [i for i in failed
                                  if isinstance(i, int)
                                  and 0 <= i < len(documents)]
                return scores, payload.get("partial_error"), failed_indices
            error = ("response did not contain one contradiction score per "
                     "document")
        last_error = str(error)
        retryable = "json" in last_error.lower() or "parse" in last_error.lower()
        if attempt < 2 and retryable:
            logger.warning("NLI response invalid, retrying: %s", last_error)
            continue
        # The scores are zeros from a failed run, not a real judgment:
        # callers must surface this instead of trusting 0.0 as "neutral".
        raise RuntimeError(last_error)
    raise RuntimeError(last_error or "NLI failed")


def nli(query: str, documents: list[str],
        query_entity: str | None = None,
        document_entities: list[str | None] | None = None) -> list[float]:
    scores, partial, _failed = nli_detailed(
        query, documents, query_entity, document_entities)
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
