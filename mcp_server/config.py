import re

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
    "Kind of the memory (default: fact). "
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


# Text search configuration for the full-text retriever. The dictionary has to
# match the language of the stored content: stemming and stop words differ per
# language, so a mismatched dictionary silently degrades recall quality.
FTS_LANGUAGE = "english"


# Similarity threshold used by the "semantic" dedupe mode.
DEFAULT_DUPLICATE_THRESHOLD = 0.9


# Avoid returning clearly unrelated vector hits unless callers opt out.
#
# Measured with jina-embeddings-v5-text-nano-retrieval: a natural-language
# question that matches a fact scores 0.62-0.95, a bare keyword that occurs
# verbatim in the fact only scores 0.30-0.61 (single tokens dilute the query
# vector). 0.4 therefore cut off roughly every third true keyword hit, so the
# default sits lower; full-text matches are never filtered by this value.
# 0.35 is the compromise: the bottom tail of the keyword range (0.30-0.35) is
# mostly noise, everything above it is kept.
DEFAULT_MIN_SIMILARITY = 0.35


PROJECT_SCOPE_BOOST = 0.05


# `supersede_last` replaces the newest memory of an entity. Only do that when
# the new fact is actually about the same topic: below this cosine similarity
# the two facts are unrelated and replacing would destroy the old one.
SUPERSEDE_SIMILARITY_MIN = 0.5


# Dedupe modes for remember/remember_batch.
DEDUPE_MODES = {"exact", "semantic", "off"}


CACHE_ACCESS_BOOST_DAYS = 7


SENTINEL_FORGOTTEN = '__forgotten__'


_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
