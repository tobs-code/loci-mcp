"""Regression tests for the eleventh round of fixes (audit follow-up).

Covers:
- `supersede_last` without an entity resolves the best-matching memory
  across the scope instead of erroring (same-topic guard still applies);
- `auto_supersede_near_duplicate` replaces an active near-duplicate instead
  of storing a reworded copy (remember + remember_batch, per-fact);
- the near-dup write that matches its superseded target is not warned as a
  copy anymore.
"""

from mcp_server.server import (
    list_memories,
    remember,
    remember_batch,
)


def _count(content_sub: str) -> int:
    listed = list_memories(contains=content_sub, project="auto-sd-probe")
    return listed["count"]


# supersede_last without an entity: scope-wide fallback ----------------------


def test_supersede_last_without_entity_replaces_scope_best_match():
    first = remember("staging service runs on port 9443",
                     project="auto-sd-probe", dedupe="off")
    old_id = first["id"]
    content_b = "deploy scripts live in the ci-pipelines repository"
    remember(content_b, project="auto-sd-probe", dedupe="off")

    result = remember("staging service runs on port 9445 now",
                      project="auto-sd-probe", dedupe="off",
                      supersede_last=True)

    assert result["ok"] is True
    assert result["status"] == "stored"
    assert old_id in result["superseded"]
    assert any(
        "supersede_last without an entity" in w for w in result["warnings"])
    assert not any("near-duplicate" in w for w in result["warnings"])


def test_supersede_last_without_entity_refuses_off_topic_match():
    remember("staging service runs on port 9443",
             project="auto-sd-probe", dedupe="off")

    result = remember(
        "completely unrelated: the office parking garage opens at six",
        project="auto-sd-probe", dedupe="off", supersede_last=True)

    assert result["ok"] is True
    assert result["superseded"] == []
    assert any("supersede_last skipped" in w for w in result["warnings"])


# auto supersede near-duplicate ----------------------------------------------


def test_auto_supersede_near_duplicate_replaces_old_memory():
    first = remember("The staging service runs on port 9443 internally",
                     project="auto-sd-probe", dedupe="off")
    old_id = first["id"]

    result = remember("The staging service runs on port 9445 internally",
                      project="auto-sd-probe", dedupe="off",
                      auto_supersede_near_duplicate=True)

    assert result["ok"] is True
    assert old_id in result["superseded"]
    assert any("auto-supersede" in w for w in result["warnings"])
    assert not any("near-duplicate:" in w for w in result["warnings"])
    # Only the new copy is active now.
    listed = list_memories(project="auto-sd-probe", include_superseded=False)
    active = [r["content"] for r in listed["results"]]
    assert all("9443" not in c for c in active)


def test_auto_supersede_near_duplicate_off_keeps_warning():
    remember("The staging service runs on port 9443 internally",
             project="auto-sd-probe", dedupe="off")

    result = remember("The staging service runs on port 9444 internally",
                      project="auto-sd-probe", dedupe="off",
                      auto_supersede_near_duplicate=False)

    assert result["ok"] is True
    assert result["superseded"] == []
    # dedupe='off' explicitly asks for every write to land: no check, no
    # warning.
    assert not any("near-duplicate" in w for w in result["warnings"])


def test_near_duplicate_warning_with_default_dedupe():
    remember("The staging service runs on port 9443 internally",
             project="auto-sd-probe", dedupe="exact")

    result = remember("The staging service runs on port 9444 internally",
                      project="auto-sd-probe", dedupe="exact")

    assert result["ok"] is True
    assert result["superseded"] == []
    assert any("near-duplicate:" in w for w in result["warnings"])


def test_batch_auto_supersede_near_duplicate_per_fact():
    first = remember("The delivery queue holds up to 10000 jobs",
                     project="auto-sd-probe", dedupe="off")
    old_id = first["id"]

    result = remember_batch(
        project="auto-sd-probe",
        facts=[{
            "content": "The delivery queue holds up to 12000 jobs",
            "auto_supersede_near_duplicate": True,
        }, {
            "content": "The delivery queue holds at most 15000 jobs",
            "auto_supersede_near_duplicate": True,
        }])

    assert result["ok"] is True
    superseded = [id for r in result["results"]
                  for id in r.get("superseded", [])]
    assert first["id"] in result["results"][0].get("superseded", []) \
        or first["id"] in superseded
    # The first entry replaced the original, the second one can only
    # supersede anything else or store on top - no forgotten ids, the old
    # memory is invalidated exactly once.
    listed = list_memories(project="auto-sd-probe")
    active = [r["content"] for r in listed["results"]]
    assert len([c for c in active if "delivery queue" in c]) >= 1
    assert old_id not in [r["id"] for r in listed["results"]]


def test_neardup_targeting_hint_names_supersede_option():
    remember("The reporting jobs run nightly on worker-1",
             project="auto-sd-probe", dedupe="exact")

    result = remember("The reporting jobs run nightly on worker-2",
                      project="auto-sd-probe", dedupe="exact")

    assert any(
        "near-duplicate: active memory" in w and "supersede=" in w
        for w in result["warnings"])
