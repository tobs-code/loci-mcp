import re

MEMORY_TYPES = {
    "fact", "decision", "learning", "preference",
    "pattern", "incident", "runbook", "context",
    "observation", "convention", "command", "note",
}


VALID_SCOPES = {"global", "project"}


DECAY_HALF_LIFE_DAYS = 90


DECAY_ACCESS_CAP = 20


DECAY_RAW_MAX = 1.5


DECAY_WEIGHT = 0.3


RRF_K = 60


FTS_WEIGHT = 0.6


# Similarity threshold used by the "semantic" dedupe mode.
DEFAULT_DUPLICATE_THRESHOLD = 0.9


# Avoid returning clearly unrelated vector hits unless callers opt out.
DEFAULT_MIN_SIMILARITY = 0.3


PROJECT_SCOPE_BOOST = 0.05


# Dedupe modes for remember/remember_batch.
DEDUPE_MODES = {"exact", "semantic", "off"}


CACHE_ACCESS_BOOST_DAYS = 7


SENTINEL_FORGOTTEN = '__forgotten__'


_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
