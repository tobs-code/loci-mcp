"""Regression tests for the twentieth round of fixes (agent audit follow-up).

Covers:
- `forget(entity=...)` names only the scopes it actually touched, not every
  scope that happened to be searched;
- `supersede_last` refuses a same-template fact about another attribute;
- a near-duplicate that differs in a number/negation is labelled as a
  possible contradiction instead of a reworded copy;
- `update_memory` metadata merge / key removal;
- a duplicate update answers with the blocking memory id;
- bulk id reads through `list_memories(memory_ids=[...])`.
"""

from mcp_server.db import get_conn, ivfflat_lists
from mcp_server.server import (
    list_memories,
    remember,
    update_batch,
    update_memory,
    forget,
)


def test_small_scopes_keep_exact_vector_search():
    assert ivfflat_lists(2) is None
    assert ivfflat_lists(10_000) == 10
    assert ivfflat_lists(400_000) == 400

    remember("Round19 index probe", project="round19-index")
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT indexdef FROM pg_indexes "
            "WHERE schemaname = 'public' AND tablename LIKE 'mem\\_%' "
            "AND indexdef LIKE '%%ivfflat%%'"
        )
        assert cur.fetchall() == []
    finally:
        conn.close()


def test_forget_entity_reports_only_the_scopes_it_touched():
    unrelated_project = "round19-unrelated"
    project = "round19-forget"
    entity = "Round19ForgetEntity"
    remember("Round19 unrelated scope holder",
             project=unrelated_project, entity="Round19Other")
    remember("Round19 forget fact", entity=entity, project=project)

    result = forget(entity=entity, soft=True)

    assert result["ok"] is True
    assert result["scopes_affected"] == [project]
    assert not any("touched multiple scopes" in w for w in result["warnings"])


def test_supersede_last_refuses_a_same_template_other_attribute():
    project = "round19-topic"
    entity = "Round19Svc"
    old = remember(
        "The Round19Svc runs on port 8766 using postgresql 16 in staging",
        project=project, entity=entity, dedupe="off")

    # Shares most of the sentence template (and therefore a high vector
    # similarity) but describes a different attribute.
    result = remember(
        "The Round19Svc runs a nightly backup job using postgresql 16 in staging",
        project=project, entity=entity, dedupe="off", supersede_last=True)

    assert result["ok"] is True
    assert result["superseded"] == []
    assert any("shares no subject term" in w for w in result["warnings"])
    active = list_memories(memory_id=old["id"], project=project)
    assert active["ok"] is True
    assert active["memory"]["superseded_by"] is None


def test_near_duplicate_conflict_is_labeled_as_possible_contradiction():
    first = remember("Round19 listener supports TLS on port 443")

    result = remember("Round19 listener does not support TLS on port 443")

    assert result["status"] == "stored"
    assert any("near-duplicate" in w and "contradiction" in w
               for w in result["warnings"])
    listed = list_memories(memory_id=first["id"])
    assert listed["memory"]["superseded_by"] is None


def test_auto_supersede_of_a_conflict_names_the_destructive_step():
    project = "round19-conflict"
    first = remember("Round19 relay listens on port 9443", project=project,
                     dedupe="off")

    result = remember("Round19 relay listens on port 9445", project=project,
                      dedupe="off", auto_supersede_near_duplicate=True)

    assert first["id"] in result["superseded"]
    assert any("auto-supersede" in w and "number" in w
               for w in result["warnings"])


def test_update_metadata_can_merge_and_remove_keys():
    project = "round19-metadata"
    stored = remember(
        "Round19 metadata probe", project=project,
        metadata={"env": "staging", "port": 8766, "owner": "ops"})

    merged = update_memory(stored["id"], project=project,
                           metadata={"env": "prod"}, metadata_mode="merge")
    assert merged["ok"] is True
    row = list_memories(memory_id=stored["id"], project=project)["memory"]
    assert row["metadata"] == {"env": "prod", "port": 8766, "owner": "ops"}

    cleared = update_memory(stored["id"], project=project,
                            clear_metadata_keys=["owner"])
    assert cleared["ok"] is True
    row = list_memories(memory_id=stored["id"], project=project)["memory"]
    assert row["metadata"] == {"env": "prod", "port": 8766}


def test_update_batch_metadata_merge_and_clear():
    project = "round19-batch-meta"
    stored = remember("Round19 batch metadata", project=project,
                      metadata={"env": "staging", "port": 8766})

    result = update_batch([{
        "memory_id": stored["id"], "project": project,
        "metadata": {"env": "prod"}, "metadata_mode": "merge",
        "clear_metadata_keys": ["port"],
    }])

    assert result["ok"] is True
    row = list_memories(memory_id=stored["id"], project=project)["memory"]
    assert row["metadata"] == {"env": "prod"}


def test_update_duplicate_names_the_blocking_memory():
    project = "round19-update-dup"
    first = remember("Round19 duplicate alpha", project=project)
    second = remember("Round19 duplicate beta", project=project)

    result = update_memory(second["id"], project=project,
                           content="Round19 duplicate alpha")

    assert result["status"] == "duplicate"
    assert result["id"] == first["id"]
    assert result["memory_id"] == second["id"]
    assert result["duplicate_of"] == first["id"]


def test_list_memories_reads_several_ids_and_names_the_missing_ones():
    project = "round19-bulk"
    first = remember("Round19 bulk one", project=project)
    second = remember("Round19 bulk two", project=project)

    result = list_memories(project=project,
                           memory_ids=[first["id"], second["id"], "missing"])

    assert result["ok"] is True
    assert result["count"] == 2
    assert {row["id"] for row in result["results"]} == {
        first["id"], second["id"]}
    assert result["missing"] == ["missing"]
    assert any("not found" in w for w in result["warnings"])
