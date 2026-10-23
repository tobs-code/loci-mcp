"""The inference gateway is a seam with two real adapters."""

import mcp_server.inference_client as inference_client
from mcp_server import inference_gateway as gw


def test_fake_inference_embeds_deterministically_without_network():
    fake = gw.FakeInference(vectors=4)

    first = fake.embed(["alpha", "beta"])
    second = fake.embed(["alpha", "beta"])

    assert first == second
    assert len(first) == 2 and all(len(vector) == 4 for vector in first)
    assert first[0] != first[1]
    assert fake.calls[0] == ("embed", (("alpha", "beta"), "document"))


def test_fake_inference_rerank_scores_shared_words_and_keeps_order():
    fake = gw.FakeInference()

    scores = fake.rerank("deploy the relay", ["deploy relay", "unrelated"])

    assert scores[0] > scores[1]


def test_fake_inference_scripts_nli_and_detailed_failures():
    fake = gw.FakeInference(nli_scores=lambda query, documents: (
        [0.9] * len(documents), "partial", [0]))

    scores, partial, failed = fake.nli_detailed("A", ["B", "C"])

    assert scores == [0.9, 0.9]
    assert partial == "partial"
    assert failed == [0]
    assert fake.nli("A", ["B"]) == [0.9]


def test_configure_gateway_routes_the_module_functions(monkeypatch):
    fake = gw.FakeInference(vectors=3)
    monkeypatch.setattr(gw, "_gateway", fake)

    vectors = gw.embed(["hello"])
    assert len(vectors[0]) == 3
    assert fake.calls[-1][0] == "embed"
    assert gw.nli("A", ["B"]) == [0.0]
    assert gw.health_details() == {"llm_configured": True}
    assert gw.get_gateway() is fake

    assert isinstance(gw.configure_gateway(), gw.HttpInference)
    monkeypatch.setattr(gw, "_gateway", fake)


def test_http_inference_delegates_to_the_transport(monkeypatch):
    captured = {}

    def fake_embed(texts, mode="document"):
        captured["embed"] = (texts, mode)
        return [[0.1, 0.2]]

    monkeypatch.setattr(inference_client, "embed", fake_embed)

    assert gw.HttpInference().embed(["x"], mode="query") == [[0.1, 0.2]]
    assert captured["embed"] == (["x"], "query")
