import os
import re


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, value))


def _env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, value))


MEMORY_TYPES = {
    "fact", "decision", "learning", "preference",
    "pattern", "incident", "runbook", "context",
    "observation", "convention", "command", "note",
}


# The kind of a memory decides how it is used later: a `decision` is read
# when "why did we do it?" is asked, a `convention` when "how do we work
# here?" is asked. The names alone do not tell an agent that, so the tool
# descriptions spell out the difference.
MEMORY_TYPE_HELP = (
    "Kind of the memory. "
    "fact: a verifiable statement about the system. "
    "decision: a choice that was made, ideally with its reason. "
    "learning: a lesson drawn from experience. "
    "preference: what the user wants. "
    "pattern: a recurring practice or behaviour. "
    "incident: something that happened, usually a failure. "
    "runbook: step-by-step instructions for a task. "
    "context: background that explains the surroundings. "
    "observation: a noticed or measured fact. "
    "convention: a rule the team follows. "
    "command: a command to run. "
    "note: anything that does not fit the others."
)


VALID_SCOPES = {"global", "project"}


DECAY_HALF_LIFE_DAYS = 90


DECAY_ACCESS_CAP = 20


DECAY_WEIGHT = 0.3


RRF_K = 60


FTS_WEIGHT = 0.6


# Minimum share of the query lexemes a full-text-only hit must cover.
#
# The full-text leg is OR-based: a document that contains one common word of
# a four-word query is a hit. The coverage factor already scales its score
# down (1/4 -> 0.25), but without a floor such a hit is still returned - and
# if nothing else matches, it is the *only* result, which reads like an
# answer. A hit that the vector retriever did not find (so it has no
# similarity to vouch for it) must therefore cover at least this share of
# the query. 0.3 keeps a single-word match of a three-word query (0.33) and
# drops the one-word matches of four or more query words - that case is
# only ever reached via the dictionary's leftover function words ("per" is
# not a stopword for the english dictionary), which the retrieval
# measurement showed to be pure noise.
FTS_MIN_COVERAGE = 0.3


# Text search configuration for the full-text retriever. The dictionary has to
# match the language of the stored content: stemming and stop words differ per
# language, so a mismatched dictionary silently degrades recall quality.
#
# The value is interpolated into SQL (index and query side), so it is
# restricted to a plain identifier. An empty or malformed value falls back to
# English instead of breaking startup with a syntax error.
_fts_language = os.getenv("FTS_LANGUAGE", "english").strip() or "english"
FTS_LANGUAGE = (_fts_language
                if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", _fts_language)
                else "english")


# Similarity threshold used by the "semantic" dedupe mode.
#
# Calibrated against the live embedding model (jina-embeddings-v5): a
# paraphrase of a stored fact scores 0.88, an intentional rewrite of one
# attribute (rate limit 60 -> 120) 0.94, unrelated facts below 0.37. The
# old 0.9 hid the paraphrase pair - it was stored as a "new" fact next to
# the old one - so the default sits at 0.85 now.
DEFAULT_DUPLICATE_THRESHOLD = 0.85


# A write that the dedupe check did NOT match still gets flagged when it is
# at least this close to an active memory: between this value and the dedupe
# threshold the content reads as a reworded copy, not as a new fact. The
# caller stores the memory but warns with the id of the near-duplicate, so
# the agent can supersede deliberately instead of forking silently.
NEAR_DUPLICATE_THRESHOLD = 0.75


# Avoid returning clearly unrelated vector hits unless callers opt out.
#
# Calibrated against the embedding model's own model card: its noise floor is
# 0.288 and its balanced cutoff is 0.678. The old 0.40 value came from the
# assumption that unrelated facts score below 0.30; the card's official
# example scores unrelated documents at 0.50-0.61. That assumption silently
# dropped true paraphrases: "sign-off before shipping code to live customers"
# scored 0.325 against the two-approvals fact. 0.30 keeps those hits just
# above the measured noise floor; full-text matches are never filtered by
# this value, so exact identifiers stay reachable. Callers that want the old
# strictness can still pass min_similarity=0.4 explicitly.
DEFAULT_MIN_SIMILARITY = 0.30


# How strongly a result set answers the query.
#
# A caller (especially an agent) needs to know whether a hit is an answer or
# merely the least-bad row of an unrelated query: `recall` used to return
# 0.30-0.51 noise hits with no warning. The thresholds follow the measured
# relevance bands: natural-language matches score 0.6-0.95, vocabulary-
# shifting paraphrases 0.3-0.55. `low` therefore still returns the rows but
# says that they may not answer the question.
CONFIDENCE_HIGH = 0.60
CONFIDENCE_MEDIUM = 0.45


# A write's `check_contradictions` used to supersede EVERY pair the NLI
# flagged, so one reworded fact could silently invalidate several correct
# memories. One confirmed replacement is applied by default; further pairs
# are reported as `kept` and need an explicit decision.
CONTRADICTION_AUTO_MIN_CONFIDENCE = 0.70
CONTRADICTION_AUTO_MAX_SUPERSEDES = 1


# Cross-encoder reranking of the first-stage candidates.
#
# The bi-encoder scores a memory and a query independently, so a paraphrase
# that changes the vocabulary can fall below the similarity floor even when a
# human would call it the obvious answer. A cross-encoder reads both texts
# together and ranked the hard eval pair first. It runs locally (Apache-2.0,
# no API key) but costs ~2s per 10 pairs on the current CPU, so it is only
# applied when the first stage is ambiguous: the best relevance is below
# RERANK_TRIGGER_SCORE and the margin to the runner-up is below
# RERANK_TRIGGER_MARGIN. Set RERANK_ALWAYS=true to rerank every query, or
# RERANK_ENABLED=false (the test suite does) to skip it.
RERANK_ENABLED = _env_bool("RERANK_ENABLED", True)
RERANK_MODEL = (
    os.getenv("RERANK_MODEL", "mixedbread-ai/mxbai-rerank-base-v1").strip()
    or "mixedbread-ai/mxbai-rerank-base-v1"
)
RERANK_TOP_K = _env_int("RERANK_TOP_K", 10, 2, 50)
RERANK_TRIGGER_SCORE = _env_float("RERANK_TRIGGER_SCORE", 0.85, 0.0, 1.0)
RERANK_TRIGGER_MARGIN = _env_float("RERANK_TRIGGER_MARGIN", 0.10, 0.0, 1.0)
RERANK_ALWAYS = _env_bool("RERANK_ALWAYS", False)
RERANK_MAX_LENGTH = _env_int("RERANK_MAX_LENGTH", 256, 64, 512)


# HTTP transports expose every tool, including the destructive maintenance
# ones. A shared bearer token is required for a non-loopback bind; see
# mcp_server.http_auth. `MCP_ALLOW_INSECURE_HTTP=true` is the explicit opt-out
# for a trusted sandbox (it is never a good default).
MCP_AUTH_TOKEN = (os.getenv("MCP_AUTH_TOKEN") or "").strip()
MCP_ALLOW_INSECURE_HTTP = _env_bool("MCP_ALLOW_INSECURE_HTTP", False)


PROJECT_SCOPE_BOOST = 0.05


# `supersede_last` replaces the best-matching memory of an entity. Only do
# that when the new fact is actually about the same topic: below this cosine
# similarity the two facts are unrelated and replacing would destroy the old one.
SUPERSEDE_SIMILARITY_MIN = 0.5


# Dedupe modes for remember/remember_batch.
DEDUPE_MODES = {"exact", "semantic", "off"}


CACHE_ACCESS_BOOST_DAYS = 7


SENTINEL_FORGOTTEN = '__forgotten__'


_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
