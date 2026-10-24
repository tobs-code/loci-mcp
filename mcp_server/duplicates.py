"""One place for "is this the same fact?" decisions.

Duplicate policy used to live in five modules: the exact/semantic lookup in
``storage``, the cross-scope winner in ``search``, two differently shaped
functions both named ``_shares_subject`` in the tools, and a text-overlap
dedupe in ``projection``. Each caller had to know which of those pieces to
combine.

This module owns the pieces and names them:

- ``find_duplicate`` / ``find_near_duplicate``: exact and vector matching
  inside one write target.
- ``same_topic`` / ``shared_content_word``: explicitly different subject
  predicates (a strict topic guard and a cheap NLI safety net).
- ``is_cross_scope_duplicate`` / ``dedupe_across_scopes``: the read-side
  rule that a project row wins over an identical global row.
- ``dedupe_similar_rows``: the bootstrap's local wording dedupe.

Tools call the policy, not its pieces. The behaviour is unchanged.
"""

from __future__ import annotations

import re
from typing import Any

from mcp_server.config import (
    DEFAULT_DUPLICATE_THRESHOLD,
    NEAR_DUPLICATE_THRESHOLD,
)
from mcp_server.lifecycle import ACTIVE_ROW_SQL, is_active
from mcp_server.scope_catalog import _mem_table
from mcp_server.scoring import _content_hash


def find_duplicate(
    cur,
    content: str,
    project: str | None,
    embedding_vec: list[float] | None = None,
    fuzzy_threshold: float | None = None,
    dedupe: str = "exact",
    exclude_id: str | None = None,
) -> tuple[str, str, float | None] | None:
    """Return the id of an existing memory that says the same thing.

    `dedupe` selects the mode: "exact" (content hash), "semantic" (hash plus
    vector similarity) or "off" (no check at all).

    Returns a tuple (memory id, mode, similarity). The mode tells the caller
    *why* the memory was seen as a duplicate, and the similarity is the
    measured cosine distance of a semantic match (None for an exact one).
    """
    if dedupe == "off":
        return None

    c_hash = _content_hash(content)
    table = _mem_table(project)
    exclude_sql = " AND id <> %s" if exclude_id else ""
    exclude_params = [exclude_id] if exclude_id else []
    cur.execute(
        f"SELECT id FROM {table} WHERE content_hash = %s "
        f"AND {ACTIVE_ROW_SQL}{exclude_sql} LIMIT 1",
        [c_hash] + exclude_params,
    )
    row = cur.fetchone()
    if row:
        return row[0], "exact", None

    if embedding_vec is not None and dedupe == "semantic":
        threshold = (fuzzy_threshold if fuzzy_threshold is not None
                     else DEFAULT_DUPLICATE_THRESHOLD)
        max_dist = 1.0 - threshold
        cur.execute(
            f"SELECT id, embedding <=> %s::vector AS dist "
            f"FROM {table} WHERE {ACTIVE_ROW_SQL} AND embedding IS NOT NULL"
            f"{exclude_sql} "
            f"ORDER BY dist ASC LIMIT 1",
            [embedding_vec] + exclude_params,
        )
        row = cur.fetchone()
        if row and float(row[1]) <= max_dist:
            return row[0], "semantic", round(1.0 - float(row[1]), 4)

    return None


def find_near_duplicate(
    cur,
    content: str,
    project: str | None,
    embedding_vec: list[float] | None = None,
    exclude_id: str | None = None,
) -> dict | None:
    """The active memory that is closest to `content` above the near-dup floor.

    Runs only when `find_duplicate` found no match: the dedupe thresholds say
    "different fact", but a similarity of at least `NEAR_DUPLICATE_THRESHOLD`
    still reads like a reworded copy of an existing memory. Storing it anyway
    is allowed (the caller may want both versions) - this function only
    reports the match so the response can warn instead of forking silently.

    The match carries its `entity` and `content` so the caller can decide
    whether the two facts really describe the same subject. A high vector
    similarity between two different entities (staging vs. production, two
    services) is not a duplicate and must never trigger an automatic
    supersede.

    Returns {"id", "similarity", "entity", "content"} or None. Exact content
    hashes never land here: `find_duplicate` already reports them before this
    check runs.
    """
    if embedding_vec is None:
        return None
    c_hash = _content_hash(content)
    table = _mem_table(project)
    exclude_sql = " AND id <> %s" if exclude_id else ""
    params: list[Any] = [embedding_vec, c_hash]
    if exclude_id:
        params.append(exclude_id)
    cur.execute(
        f"SELECT id, embedding <=> %s::vector AS dist, entity, content "
        f"FROM {table} "
        f"WHERE content_hash <> %s "
        f"AND {ACTIVE_ROW_SQL} "
        # A relation-context sentence describes a graph edge, not a fact of
        # the entity it mentions. Comparing a new fact against it produced a
        # "near-duplicate across entities" warning for every entity name that
        # also appeared in an edge sentence, which is noise, not a fork.
        f"AND COALESCE(metadata->>'relation_context', 'false') <> 'true' "
        f"AND embedding IS NOT NULL{exclude_sql} "
        f"ORDER BY dist ASC LIMIT 1",
        params,
    )
    row = cur.fetchone()
    if not row:
        return None
    similarity = round(1.0 - float(row[1]), 4)
    if similarity >= NEAR_DUPLICATE_THRESHOLD:
        return {"id": row[0], "similarity": similarity,
                "entity": row[2], "content": row[3]}
    return None


# --- Subject predicates ----------------------------------------------------

_WORD_RE = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*")
_CAMEL_WORD_RE = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+")
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "for", "from",
    "has", "have", "in", "is", "it", "its", "of", "on", "or", "that", "the",
    "this", "to", "was", "were", "with", "now", "still", "also",
    "gets", "get", "new", "old",
    # Generic predicates describe the template, not the subject: two facts
    # about a port and a backup job both "run" on the same service.
    "run", "runs", "using", "use", "uses", "used", "supports", "support",
    "listens", "listening", "serves", "provides", "includes", "allows",
    "contains", "stores",
    # Generic containers are template words too: a port fact and a backup-job
    # fact about "the service" share them without sharing a subject.
    "service", "application", "app", "system", "platform", "component",
    "der", "die", "das", "ein", "eine", "einer", "und", "oder", "ist", "sind",
    "auf", "fuer", "für", "mit", "von", "zu", "im", "in", "den", "dem",
    "des", "wird", "werden", "hat", "haben",
}

# Attribute *values* change when a fact is updated, so they must not count as
# the shared subject: "runs on friday" -> "runs on saturday" is the same
# topic, "runs on port 8766" -> "runs a nightly backup" is not.
_VALUE_WORDS = {
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
    "sunday", "mon", "tue", "wed", "thu", "fri", "sat", "sun",
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
    "daily", "weekly", "monthly", "yearly", "annual", "annually",
    "hourly", "today", "tomorrow", "yesterday", "tonight", "utc", "gmt",
    # Environment qualifiers are values too: "in staging" vs "in production"
    # is an environment update, and the same words must not make a port fact
    # and a backup-job fact look like the same topic.
    "staging", "production", "prod", "development", "dev", "qa", "uat",
    "sandbox", "local", "localhost", "postgres", "postgresql", "mysql",
    "redis", "sqlite", "docker", "kubernetes", "k8s",
} | _STOPWORDS

_NEGATIONS = {
    "not", "no", "never", "without", "none", "cannot", "disabled", "false",
    "nicht", "kein", "keine", "keinen", "ohne", "nie", "niemals",
}

# A second, deliberately simpler predicate used by the contradiction resolver.
# It only needs "do the two sentences share a content word at all?", so it
# keeps its own stopword list instead of reusing the topic guard above.
_SUBJECT_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "are", "was",
    "were", "has", "have", "had", "not", "but", "its", "his", "her", "their",
    "der", "die", "das", "und", "oder", "ist", "sind", "war", "waren", "ein",
    "eine", "einer", "eines", "mit", "von", "für", "fuer", "auf", "aus", "bei",
    "nicht", "kein", "keine", "über", "ueber", "unter", "wird", "werden",
    "wurde", "wurden", "sich", "auch", "noch", "nur", "als", "wie", "dass",
    "nach", "vor", "zum", "zur", "im", "in", "an", "am", "es", "sie", "er",
}


def _entity_tokens(entity: str | None) -> set[str]:
    """Content-free tokens of an entity name (CamelCase is split).

    The entity name is not a subject term: two facts about the same service
    always share it, even when one describes its port and the other its
    backup job. Removing it is what makes the topic check meaningful.
    """
    if not entity:
        return set()
    tokens: set[str] = set()
    for chunk in re.split(r"[^A-Za-z0-9]+", str(entity)):
        if not chunk:
            continue
        # The content tokenizer keeps "Round19Svc" as one token, so the
        # camel-split parts alone would not remove it from the topic words.
        tokens.add(chunk.lower())
        for match in _CAMEL_WORD_RE.findall(chunk):
            tokens.add(match.lower())
    return tokens


def topic_tokens(text: str, entity: str | None) -> set[str]:
    """Subject words of a fact: no stopwords, values or the entity name."""
    tokens = {token.lower() for token in _WORD_RE.findall((text or "").lower())}
    ignored = _VALUE_WORDS | _entity_tokens(entity)
    return {token for token in tokens
            if token not in ignored and not token.isdigit() and len(token) > 1}


def same_topic(new_content: str, old_content: str,
               entity: str | None,
               similarity: float | None = None) -> bool:
    """Whether two facts talk about the same non-entity subject.

    `supersede_last` used to accept any pair above a similarity floor. Two
    sentences about the same service share their template ("the QA service
    runs ..."), so a port fact could replace a backup-job fact. This compares
    the remaining *attribute* words after removing the entity, stopwords and
    attribute values (numbers, days, intervals). A port update keeps the
    words "port"/"runs"; a backup job does not.

    Containment (overlap / smaller token set), not Jaccard, is what makes an
    attribute *update* pass: the new sentence often adds detail words ("up to
    five times with a 30 second timeout") that inflate the union and used to
    push a legitimate replacement below the Jaccard floor. Requiring at least
    two shared terms keeps a generic verb such as "runs" from proving a topic
    match on its own. A very high similarity still accepts a short reworded
    fact, so a legitimate one-word update is not blocked.
    """
    new_tokens = topic_tokens(new_content, entity)
    old_tokens = topic_tokens(old_content, entity)
    if new_tokens == old_tokens:
        # Only attribute values (day, number, interval, environment) changed:
        # the subject words are identical, so the fact is an update even when
        # the entity leaves just one word (or none) behind.
        return True
    if not new_tokens or not old_tokens:
        return similarity is not None and similarity >= 0.92
    overlap = len(new_tokens & old_tokens)
    if overlap >= 2:
        containment = overlap / min(len(new_tokens), len(old_tokens))
        if containment >= 0.5:
            return True
    return overlap >= 1 and similarity is not None and similarity >= 0.92


def content_conflict(new_content: str, old_content: str) -> bool:
    """Cheap signal that two similar texts differ in a number or a negation.

    Such a pair is usually an attribute update or an outright contradiction,
    not a reworded copy. The near-duplicate warning labels it accordingly so
    an agent does not blindly accept `on_near_duplicate='reject'` advice for a
    fact that is actually meant to replace the old value.

    This is deliberately not an NLI call: writes stay cheap. It only decides
    how the warning reads; `check_contradictions=true` remains the real test.
    """
    new_numbers = set(_NUMBER_RE.findall((new_content or "").lower()))
    old_numbers = set(_NUMBER_RE.findall((old_content or "").lower()))
    if new_numbers and old_numbers and new_numbers != old_numbers:
        return True
    new_words = set(_WORD_RE.findall((new_content or "").lower()))
    old_words = set(_WORD_RE.findall((old_content or "").lower()))
    return bool(new_words & _NEGATIONS) != bool(old_words & _NEGATIONS)


def _subject_tokens(text: str) -> set[str]:
    """Lower-cased content words of a fact (stopwords removed)."""
    tokens = re.findall(r"[0-9A-Za-zÄÖÜäöüß][0-9A-Za-zÄÖÜäöüß_.-]*",
                        text or "")
    return {t.lower() for t in tokens
            if t.lower() not in _SUBJECT_STOPWORDS and len(t) > 1}


def shared_content_word(a: str, b: str) -> bool:
    """True when two facts share at least one content word.

    A recency-based auto-resolve is only defensible for facts that clearly
    talk about the same thing. Two unrelated statements can still be scored
    as a contradiction by a noisy NLI model; requiring a shared content word
    keeps that mistake from silently invalidating a correct memory.
    """
    return bool(_subject_tokens(a) & _subject_tokens(b))


# --- Cross-scope winner ----------------------------------------------------

def is_cross_scope_duplicate(row: dict, table: str,
                             seen_active: dict[str, str]) -> bool:
    """Cross-scope dedup for a stream of rows, project tables first.

    Only an *active* row may hide an *active* copy in another scope. History
    (superseded and soft-forgotten rows) is never dropped: a superseded fact
    used to disappear from `include_superseded` because an active copy with
    the same content lived in the other scope. Copies inside one scope stay
    visible too - `dedupe="off"` deliberately allows them.
    """
    c_hash = row.get("content_hash") or ""
    if not c_hash or not is_active(row):
        return False
    previous = seen_active.get(c_hash)
    if previous is not None and previous != table:
        return True
    seen_active[c_hash] = table
    return False


def _scope_of_row(row: dict) -> str:
    table = row.get("_table")
    if table:
        return "global" if table == "memories" else table
    return row.get("_project") or "global"


def dedupe_across_scopes(
    rows: list[dict],
    include_duplicates: bool = False,
) -> tuple[list[dict], int]:
    """Drop active cross-scope copies of the same content.

    Same rule as `is_cross_scope_duplicate`, but for a complete row set:
    inactive rows survive, and copies inside the winning scope survive
    (`dedupe="off"` deliberately allows them). An active project table wins
    over the global one; when several project tables hold the same content the
    first one wins, so a row set must be ordered project tables first.

    Returns `(kept, hidden_count)`. The count matters: a browse that silently
    lists one of two identical rows looks complete while a whole scope is
    missing from it.
    """
    if include_duplicates:
        return list(rows), 0
    groups: dict[str, list[dict]] = {}
    kept: list[dict] = []
    for row in rows:
        c_hash = row.get("content_hash") or ""
        if not c_hash or not is_active(row):
            kept.append(row)
            continue
        groups.setdefault(c_hash, []).append(row)
    hidden = 0
    for group in groups.values():
        project_rows = [r for r in group if _scope_of_row(r) != "global"]
        winner = (_scope_of_row(project_rows[0]) if project_rows
                  else _scope_of_row(group[0]))
        for row in group:
            if _scope_of_row(row) == winner:
                kept.append(row)
            else:
                hidden += 1
    return kept, hidden


# --- Bootstrap wording dedupe ---------------------------------------------

def _stem(word: str) -> str:
    """Crude plural stemming so 'monday' and 'mondays' count as the same."""
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _tokens(content: str) -> set[str]:
    text = (content or "").lower()
    chars = [c if c.isalnum() or c.isspace() else " " for c in text]
    return {_stem(w) for w in "".join(chars).split() if w}


def normalize_content(content: str) -> str:
    """Key for near-duplicate detection: case, punctuation and plurals
    insensitive."""
    return " ".join(sorted(_tokens(content)))


def _similarity(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def dedupe_similar_rows(rows: list[dict],
                        threshold: float = 0.9) -> list[dict]:
    """Drop rows that say the same thing in slightly different words.

    Keeps the first occurrence, so ordering (by score or recency) decides
    which variant survives.
    """
    kept: list[dict] = []
    kept_tokens: list[set[str]] = []
    for row in rows:
        tokens = _tokens(str(row.get("content", "")))
        if not tokens:
            kept.append(row)
            kept_tokens.append(tokens)
            continue
        if any(_similarity(tokens, other) >= threshold for other in kept_tokens
               if other):
            continue
        kept.append(row)
        kept_tokens.append(tokens)
    return kept
