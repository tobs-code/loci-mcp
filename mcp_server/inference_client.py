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


def nli_detailed(query: str,
                 documents: list[str],
                 query_entity: str | None = None,
                 document_entities: list[str | None] | None = None,
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
    # model/format hiccup, not an outage: retry once instead of reporting the
    # whole contradiction scan as incomplete.
    for attempt in range(2):
        if attempt:
            # A short pause before the retry: the same immediate resend
            # would most likely hit the very same transient state again.
            time.sleep(0.5)
        resp = _get_session().post(
            f"{INFERENCE_URL}/nli",
            json=body,
            timeout=60,
        )
        resp.raise_for_status()
        try:
            payload = resp.json()
        except ValueError as e:
            last_error = f"response was not valid NLI JSON: {e}"
            if attempt == 0:
                logger.warning("NLI response invalid, retrying once: %s",
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
        if attempt == 0 and retryable:
            logger.warning("NLI response invalid, retrying once: %s",
                           last_error)
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
