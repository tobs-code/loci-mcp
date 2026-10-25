"""Round-38 feature: `compile_context` assembles a token-bounded block."""
import mcp_server.server as server
from mcp_server.server import compile_context, remember


PROJECT = "round38-compile"


def test_compile_context_packs_ranked_hits():
    remember("Round38 deploy runbook: build, migrate, restart the workers",
             project=PROJECT, entity="Round38Deploy")
    remember("Round38 unrelated note about the coffee machine",
             project=PROJECT)

    result = compile_context("deploy runbook workers", project=PROJECT)

    assert result["ok"] is True
    assert result["count"] >= 1
    assert result["truncated"] is False
    assert result["used_tokens"] <= result["max_tokens"] == 4000
    assert "build, migrate" in result["context"]
    assert result["items"][0]["id"]
    assert result["items"][0]["_relevance_score"] is not None
    assert result["scopes_searched"]


def test_compile_context_respects_a_tight_budget():
    for index in range(12):
        remember(f"Round38 long fact {index}: " + "detail " * 400,
                 project=PROJECT)

    result = compile_context("Round38 long fact", project=PROJECT,
                             max_tokens=256)

    assert result["ok"] is True
    assert result["used_tokens"] <= 256
    assert result["truncated"] is True
    assert result["count"] >= 1
    assert result["items"][0]["content"]
    assert result["context"]


def test_compile_context_empty_result_warns():
    result = compile_context("no-such-round38-token-anywhere",
                             project=PROJECT)

    assert result["ok"] is True
    assert result["count"] == 0
    assert result["context"] == ""
    assert result["items"] == []
    assert any("no memories matched" in w for w in result["warnings"])


def test_compile_context_validates_arguments():
    empty = compile_context("   ", project=PROJECT)
    assert empty["ok"] is False
    assert "query" in empty["error"]

    small = compile_context("deploy", project=PROJECT, max_tokens=10)
    assert small["ok"] is False
    assert "max_tokens" in small["error"]

    bad_type = compile_context("deploy", project=PROJECT, max_tokens="big")
    assert bad_type["ok"] is False

    bad_limit = compile_context("deploy", project=PROJECT, limit=0)
    assert bad_limit["ok"] is False


def test_compile_context_is_exposed_in_both_toolsets():
    assert compile_context in server._ALL_TOOLS
    assert "compile_context" in server._CORE_TOOL_NAMES
