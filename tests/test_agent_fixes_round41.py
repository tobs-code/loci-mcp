"""Round-41 feature: `idempotency_key` makes a retry a no-op replay."""
from mcp_server.server import forget, list_memories, remember, remember_batch

PROJECT = "round41-idempotency"


def test_replay_returns_the_first_id_and_stores_nothing():
    first = remember("Round41 deploy runs at noon", project=PROJECT,
                     idempotency_key="deploy-1")
    assert first["ok"] is True
    assert first["status"] == "stored"

    again = remember("Round41 deploy runs at noon", project=PROJECT,
                     dedupe="off", idempotency_key="deploy-1")

    assert again["ok"] is True
    assert again["status"] == "duplicate"
    assert again["id"] == first["id"]
    assert again["duplicate_of"] == first["id"]
    assert again["duplicate_mode"] == "idempotency_key"
    assert again["idempotent_replay"] is True
    assert any("idempotent replay" in w for w in again["warnings"])
    rows = list_memories(project=PROJECT, contains="Round41 deploy runs")
    assert rows["total"] == 1


def test_replay_does_not_apply_supersede_side_effects():
    target = remember("Round41 old port is 8000", project=PROJECT)["id"]
    first = remember("Round41 new port is 9000", project=PROJECT,
                     idempotency_key="port-change")
    assert first["status"] == "stored"

    replay = remember("Round41 new port is 9000", project=PROJECT,
                      supersede=target, idempotency_key="port-change")

    assert replay["status"] == "duplicate"
    assert replay["id"] == first["id"]
    assert replay["idempotent_replay"] is True
    assert not replay.get("superseded")
    still_active = list_memories(project=PROJECT, memory_id=target)
    assert still_active["ok"] is True


def test_same_key_in_another_scope_is_not_a_replay():
    first = remember("Round41 shared wording", project=PROJECT,
                     idempotency_key="shared-key")
    other = remember("Round41 shared wording", project="round41-other",
                     idempotency_key="shared-key")

    assert other["ok"] is True
    assert other["status"] == "stored"
    assert other["id"] != first["id"]


def test_replay_with_different_content_names_the_stored_row():
    first = remember("Round41 first payload", project=PROJECT,
                     idempotency_key="changed-payload")

    replay = remember("Round41 second payload", project=PROJECT,
                      idempotency_key="changed-payload")

    assert replay["status"] == "duplicate"
    assert replay["id"] == first["id"]
    assert any("different content" in w for w in replay["warnings"])


def test_batch_replays_share_the_key_space_with_remember():
    first = remember("Round41 batch collision", project=PROJECT,
                     idempotency_key="batch-1")

    result = remember_batch([{"content": "Round41 batch collision",
                              "idempotency_key": "batch-1"},
                             {"content": "Round41 fresh fact",
                              "idempotency_key": "batch-2"}],
                            project=PROJECT, dedupe="off")

    assert result["ok"] is True
    assert result["succeeded"] == 2
    entry = result["results"][0]
    assert entry["status"] == "duplicate"
    assert entry["id"] == first["id"]
    assert entry["idempotent_replay"] is True
    assert result["results"][1]["status"] == "stored"


def test_invalid_idempotency_key_is_an_argument_error():
    for value in ("", "   ", 7, "x" * 129):
        result = remember("Round41 bad key", project=PROJECT,
                          idempotency_key=value)
        assert result["ok"] is False, (value, result)
        assert "idempotency_key" in result["error"]


def test_invalid_batch_key_only_fails_its_entry():
    result = remember_batch([
        {"content": "Round41 good entry", "idempotency_key": "ok-key"},
        {"content": "Round41 bad entry", "idempotency_key": 42},
    ], project=PROJECT)

    assert result["ok"] is True
    assert result["succeeded"] == 1
    assert result["failed"] == 1
    assert "idempotency_key" in result["results"][1]["error"]


def test_expired_row_frees_its_key():
    first = remember("Round41 temporary fact", project=PROJECT,
                     expires_at="2000-01-01T00:00:00Z",
                     idempotency_key="ttl-key")
    assert first["status"] == "stored"

    second = remember("Round41 replacement fact", project=PROJECT,
                      idempotency_key="ttl-key")

    assert second["status"] == "stored"
    assert second["id"] != first["id"]

def test_superseded_row_frees_its_key():
    old = remember("Round41 old retry fact", project=PROJECT,
                   idempotency_key="supersede-key")
    replacement = remember("Round41 new retry fact", project=PROJECT,
                           supersede=old["id"])
    assert replacement["status"] == "stored"

    reused = remember("Round41 third retry fact", project=PROJECT,
                      idempotency_key="supersede-key")

    assert reused["status"] == "stored"
    assert reused["id"] not in (old["id"], replacement["id"])


def test_forgotten_row_frees_its_key():
    first = remember("Round41 forgotten retry fact", project=PROJECT,
                     idempotency_key="forget-key")
    assert first["status"] == "stored"
    assert forget(memory_id=first["id"], project=PROJECT)["ok"] is True

    second = remember("Round41 restored retry fact", project=PROJECT,
                      idempotency_key="forget-key")

    assert second["status"] == "stored"
    assert second["id"] != first["id"]
