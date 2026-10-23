"""One interface for every inference call.

Every module that needs an embedding, a rerank score or an NLI verdict asks
this gateway instead of importing the transport functions. The gateway is the
seam: ``HttpInference`` talks to the inference service, ``FakeInference``
answers deterministically and never opens a socket. A caller that accepts a
gateway can be exercised without Postgres, without the inference container and
without patching an import path.

``inference_client`` stays the HTTP transport. It is an implementation detail
of ``HttpInference`` and is still the module the retry and timeout policy
lives in.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Callable, Protocol, Sequence, runtime_checkable

from mcp_server import inference_client

logger = logging.getLogger(__name__)

NLIScores = tuple[list[float], str | None, list[int]]


@runtime_checkable
class InferenceGateway(Protocol):
    """The inference surface the tools depend on."""

    def embed(self, texts: Sequence[str], mode: str = "document") -> list[list[float]]:
        ...

    def rerank(self, query: str, documents: Sequence[str], *,
               timeout: float = 60.0) -> list[float]:
        ...

    def nli(self, query: str, documents: Sequence[str],
            query_entity: str | None = None,
            document_entities: Sequence[str | None] | None = None) -> list[float]:
        ...

    def nli_detailed(self, query: str, documents: Sequence[str],
                     query_entity: str | None = None,
                     document_entities: Sequence[str | None] | None = None,
                     *, timeout: float = 60.0,
                     deadline: float | None = None) -> NLIScores:
        ...

    def health(self) -> bool:
        ...

    def health_details(self) -> dict | None:
        ...

    def nli_configured(self) -> bool | None:
        ...


class HttpInference:
    """The production adapter: delegates to the HTTP transport."""

    def embed(self, texts: Sequence[str], mode: str = "document") -> list[list[float]]:
        return inference_client.embed(list(texts), mode=mode)

    def rerank(self, query: str, documents: Sequence[str], *,
               timeout: float = 60.0) -> list[float]:
        return inference_client.rerank(query, list(documents), timeout=timeout)

    def nli(self, query: str, documents: Sequence[str],
            query_entity: str | None = None,
            document_entities: Sequence[str | None] | None = None) -> list[float]:
        return inference_client.nli(
            query, list(documents), query_entity, document_entities)

    def nli_detailed(self, query: str, documents: Sequence[str],
                     query_entity: str | None = None,
                     document_entities: Sequence[str | None] | None = None,
                     *, timeout: float = 60.0,
                     deadline: float | None = None) -> NLIScores:
        return inference_client.nli_detailed(
            query, list(documents), query_entity, document_entities,
            timeout=timeout, deadline=deadline)

    def health(self) -> bool:
        return inference_client.health()

    def health_details(self) -> dict | None:
        return inference_client.health_details()

    def nli_configured(self) -> bool | None:
        return inference_client.nli_configured()


class FakeInference:
    """A second adapter: deterministic embeddings, scripted NLI, no network.

    ``vectors`` is the embedding width (8 by default). ``nli_scores`` is
    either a fixed list of scores, one score or a callable
    ``(query, documents) -> scores``; a callable may also return the full
    ``(scores, partial_error, failed_indices)`` tuple for ``nli_detailed``.
    Every call is recorded in ``calls`` so a test can assert what was asked.
    """

    def __init__(self, *, vectors: int = 8,
                 nli_scores: float | Sequence[float] | Callable = 0.0,
                 healthy: bool = True,
                 nli_configured: bool = True) -> None:
        self.vectors = vectors
        self.nli_scores = nli_scores
        self.healthy = healthy
        self._nli_configured = nli_configured
        self.calls: list[tuple[str, tuple]] = []

    def embed(self, texts: Sequence[str], mode: str = "document") -> list[list[float]]:
        self.calls.append(("embed", (tuple(texts), mode)))
        return [self._vector(str(text)) for text in texts]

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [b / 255.0 for b in digest[: self.vectors]]

    def rerank(self, query: str, documents: Sequence[str], *,
               timeout: float = 60.0) -> list[float]:
        self.calls.append(("rerank", (query, tuple(documents))))
        return [self._score(query, document, index)
                for index, document in enumerate(documents)]

    def _score(self, query: str, document: str, index: int) -> float:
        shared = len(set(query.lower().split()) & set(document.lower().split()))
        return float(shared) + (0.001 * index)

    def nli(self, query: str, documents: Sequence[str],
            query_entity: str | None = None,
            document_entities: Sequence[str | None] | None = None) -> list[float]:
        return self.nli_detailed(
            query, documents, query_entity, document_entities)[0]

    def nli_detailed(self, query: str, documents: Sequence[str],
                     query_entity: str | None = None,
                     document_entities: Sequence[str | None] | None = None,
                     *, timeout: float = 60.0,
                     deadline: float | None = None) -> NLIScores:
        self.calls.append(("nli", (query, tuple(documents), query_entity,
                                   tuple(document_entities)
                                   if document_entities is not None else None)))
        scores = self.nli_scores
        if callable(scores):
            produced = scores(query, list(documents))
            if (isinstance(produced, tuple) and len(produced) == 3):
                return produced
            scores = produced
        if isinstance(scores, (int, float)):
            return [float(scores)] * len(documents), None, []
        scores = list(scores)
        if len(scores) != len(documents):
            raise ValueError("FakeInference needs one score per document")
        return [float(score) for score in scores], None, []

    def health(self) -> bool:
        return self.healthy

    def health_details(self) -> dict | None:
        if not self.healthy:
            return None
        return {"llm_configured": self._nli_configured}

    def nli_configured(self) -> bool | None:
        return self._nli_configured if self.healthy else None


_gateway: InferenceGateway = HttpInference()


def configure_gateway(gateway: InferenceGateway | None = None) -> InferenceGateway:
    """Install the process-wide gateway (tests pass a ``FakeInference``)."""
    global _gateway
    _gateway = gateway if gateway is not None else HttpInference()
    return _gateway


def get_gateway() -> InferenceGateway:
    return _gateway


def embed(texts: list[str], mode: str = "document") -> list[list[float]]:
    return _gateway.embed(texts, mode=mode)


def rerank(query: str, documents: list[str], *, timeout: float = 60.0) -> list[float]:
    return _gateway.rerank(query, documents, timeout=timeout)


def nli(query: str, documents: list[str],
        query_entity: str | None = None,
        document_entities: list[str | None] | None = None) -> list[float]:
    return _gateway.nli(query, documents, query_entity, document_entities)


def nli_detailed(query: str, documents: list[str],
                 query_entity: str | None = None,
                 document_entities: list[str | None] | None = None,
                 *, timeout: float = 60.0,
                 deadline: float | None = None) -> NLIScores:
    return _gateway.nli_detailed(
        query, documents, query_entity, document_entities,
        timeout=timeout, deadline=deadline)


def health() -> bool:
    return _gateway.health()


def health_details() -> dict | None:
    return _gateway.health_details()


def nli_configured() -> bool | None:
    return _gateway.nli_configured()
