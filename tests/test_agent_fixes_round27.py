"""Regression tests for the round-27 live-audit fixes.

1. `forget` with an explicit `project="global"` is bounded: the reserved
   name maps to `None` internally, and the plain `if not project` test
   widened every id lookup to ALL scopes - copies in unrelated projects
   were deleted (or refused as "ambiguous") together with the global row.
2. Unknown fields inside a `links` edge object are rejected, instead of
   being silently ignored while the caller believes the edge matches.
3. `parse_nli_json` tolerates trailing commas (direct and embedded), the
   harmless formatting slip that made whole NLI passes fail.
4. `recall` accepts a `min_relevance` floor on the combined
   `_relevance_score` with its own counter and empty-result reason.
5. `get_entity` warns when the requested project does not exist instead of
   posing as an empty scope.
"""

from inference.llm import parse_nli_json
from mcp_server.backup import export_memories, import_memories
from mcp_server.tools.memory import forget, remember, remember_batch
from mcp_server.tools.read import get_entity, list_memories, recall


def test_forget_bulk_global_only_touches_the_shared_table():
    stored = remember(content="The release pipeline signs artifacts with "
                              "key kappa.", scope="global")
    assert stored["ok"] is True, stored.get("error")
    mid = stored["id"]
    data = export_memories(project="global")["data"]
    imported = import_memories(data=data, format="json", project="forget-q27")
    assert imported["ok"] is True, imported.get("error")

    # The dry run sees BOTH copies when no project is given ...
    preview = forget(memory_ids=[mid], dry_run=True)
    assert preview["ok"] is True
    assert preview["bulk_results"][0]["status"] == "would_soft_forget"
    assert set(preview["bulk_results"][0]["scopes"]) == {
        "global", "forget-q27"}

    # ... and an explicit project="global" removes ONLY the shared row:
    # 'global' normalizes to None, and the bulk path must not treat that
    # as "no scope given".
    deleted = forget(memory_ids=[mid], project="global", soft=False)
    assert deleted["ok"] is True, deleted.get("error")
    assert deleted["counts"]["memories"] == 1
    assert deleted["bulk_results"][0]["scopes"] == ["global"]
    surviving = list_memories(project="forget-q27", memory_id=mid)
    assert surviving["ok"] is True
    assert surviving["results"], "the project copy must survive"


def test_forget_single_global_id_is_not_ambiguous():
    stored = remember(content="The audit bot reads only append-only logs.",
                      scope="global")
    assert stored["ok"] is True, stored.get("error")
    mid = stored["id"]
    data = export_memories(project="global")["data"]
    imported = import_memories(data=data, format="json", project="forget-r27")
    assert imported["ok"] is True, imported.get("error")

    # Same id in two scopes; the explicit global scope must pin the removal
    # to the shared table instead of refusing the call as ambiguous.
    result = forget(memory_id=mid, project="global", soft=True)
    assert result["ok"] is True, result.get("error")
    # Exactly one row was hidden - the shared one.
    assert result["counts"]["memories"] == 1
    assert result["scopes"] == ["forget-r27", "global"]  # where the id exists
    surviving = list_memories(project="forget-r27", memory_id=mid)
    assert surviving["ok"] is True
    assert surviving["results"], "the project copy must survive"


def test_links_unknown_field_is_rejected():
    result = remember(
        content="The STRAY fact carries a malformed link.",
        entity="Stray27",
        links=[{"to_entity": "Target27", "relation_type": "USES",
                "token_endpoint": "https://oauth.example/token"}])
    assert result["ok"] is False
    assert "unknown field" in result["error"]
    assert "token_endpoint" in result["error"]
    browsed = list_memories(contains="STRAY fact")
    assert browsed["count"] == 0, "nothing may be stored alongside the error"


def test_batch_links_unknown_field_is_a_per_fact_failure():
    result = remember_batch(
        project="stray-batch-27",
        facts=[
            {"content": "Batch fact with a clean edge.", "entity": "B27",
             "links": [{"to_entity": "Clean27", "relation_type": "USES"}]},
            {"content": "Batch fact with a stray edge field.",
             "entity": "B27",
             "links": [{"to_entity": "Other27", "relation_type": "USES",
                        "bogus": 1}]},
        ],
    )
    assert result["ok"] is True
    good, bad = result["results"][0], result["results"][1]
    assert good["status"] == "stored"
    assert bad["status"] == "error"
    assert "unknown field" in bad["error"]
    assert bad["id"] is None


def test_all_link_fields_stay_accepted():
    result = remember(
        content="The WIRED fact carries a complete edge.",
        entity="Wired27",
        links=[{"from_entity": "Wired27", "to_entity": "Peer27",
                "relation_type": "LINKS_TO", "weight": 0.5,
                "context": "round 27"}])
    assert result["ok"] is True, result.get("error")


def test_parse_nli_json_tolerates_trailing_commas():
    parsed = parse_nli_json(
        '{"label": "contradiction", "score": 0.9,}')
    assert parsed == {"label": "contradiction", "score": 0.9}
    # Same slip inside prose, caught by the substring path.
    embedded = parse_nli_json(
        'Here you go: {"label": "entailment", "score": 0.4,} hope this helps')
    assert embedded == {"label": "entailment", "score": 0.4}
    # A structurally broken score is still not faked into a judgment.
    broken = parse_nli_json('{"label": "contradiction", "score": }')
    assert broken is None


def test_recall_min_relevance_filters_and_counts():
    stored = remember(content="The API service stores its sessions in the "
                              "Redis cache.", entity="MinRel27",
                      project="minrel-27")
    assert stored["ok"] is True, stored.get("error")
    base = recall("How does the API service store its sessions?",
                  project="minrel-27", min_similarity=None)
    assert base["results"], base.get("warnings")
    scores = [r["_relevance_score"] for r in base["results"]
              if isinstance(r, dict) and r["_relevance_score"] is not None]
    assert scores
    weakest = min(scores)

    # A floor exactly at the weakest score keeps every hit.
    kept = recall("How does the API service store its sessions?",
                  project="minrel-27", min_similarity=None,
                  min_relevance=max(0.0, weakest - 1e-6))
    assert kept["count"] == base["count"]
    assert "dropped_by_relevance" not in kept

    # An unreachable floor (1.0) drops every query hit and is reported.
    dropped = recall("How does the API service store its sessions?",
                     project="minrel-27", min_similarity=None,
                     min_relevance=1.0)
    assert dropped["count"] == 0
    assert dropped.get("dropped_by_relevance", 0) == len(scores)
    assert any("min_relevance" in w for w in dropped["warnings"])

    invalid = recall("any query", min_relevance=1.5)
    assert invalid["ok"] is False
    assert "min_relevance" in invalid["error"]


def test_get_entity_warns_about_unknown_project():
    remember(content="The auth service rotates the signing key nightly.",
             entity="Auth", scope="global")
    remember(content="The auth tokens live in Redis.", entity="Auth",
             project="gp-27")
    hit = get_entity(entity="Auth", project="gp-27-ghost")
    assert hit["ok"] is True, hit.get("error")
    assert any("gp-27-ghost" in w for w in hit["warnings"])


def test_get_entity_unknown_project_names_the_fallback_on_a_miss():
    miss = get_entity(entity="NoSuch27", project="gp-27-ghost")
    assert miss["ok"] is False
    assert miss["status"] == "not_found"
    assert any("gp-27-ghost" in w for w in miss["warnings"])
