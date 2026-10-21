import hashlib
import re
import uuid
from datetime import datetime, timezone

from mcp_server.config import (
    DECAY_ACCESS_CAP,
    DECAY_HALF_LIFE_DAYS,
    DECAY_RAW_MAX,
    DECAY_WEIGHT,
)

_FTS_TOKEN_RE = re.compile(r"[^\s]+")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _gen_id() -> str:
    return uuid.uuid4().hex[:16]


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()


def _decay_score(created_at: datetime, updated_at: datetime | None,
                 access_count: int, importance: float) -> float:
    """Raw recency/frequency score in [0, DECAY_RAW_MAX]."""
    ref = updated_at or created_at
    age_days = (_now() - ref).total_seconds() / 86400
    decay = 0.5 ** (age_days / DECAY_HALF_LIFE_DAYS)
    access_boost = 1.0 + min(access_count, DECAY_ACCESS_CAP) * 1.5 / DECAY_ACCESS_CAP
    return importance * decay * access_boost


def _decay_norm(created_at: datetime, updated_at: datetime | None,
                access_count: int, importance: float) -> float:
    """Normalized decay score in [0, 1], comparable to a normalized RRF score."""
    raw = _decay_score(created_at, updated_at, access_count, importance)
    return max(0.0, min(1.0, raw / DECAY_RAW_MAX))


def _blend_scores(rrf_norm: float, decay_norm: float) -> float:
    """Blend relevance and decay so decay can reorder but never dominate."""
    return (1.0 - DECAY_WEIGHT) * rrf_norm + DECAY_WEIGHT * decay_norm


def _fts_query(text: str | None) -> str | None:
    """Build an OR-joined websearch query.

    websearch_to_tsquery() ANDs bare terms, so multi-word natural language
    queries almost never matched. Joining the terms with OR keeps the FTS
    retriever useful for any query containing a distinctive word.
    """
    if not text:
        return None
    tokens = []
    for raw in _FTS_TOKEN_RE.findall(text):
        token = re.sub(r'[():"&|!<>*\\]', " ", raw).strip()
        token = re.sub(r"\s+", " ", token)
        if token and token.upper() != "OR":
            tokens.append(token)
    if not tokens:
        return None
    return " OR ".join(tokens)
