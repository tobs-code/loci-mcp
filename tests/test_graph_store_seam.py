"""Cypher and mirror-row writes stay behind the graph seam."""

from pathlib import Path

import mcp_server

SOURCE_ROOT = Path(mcp_server.__file__).parent


def test_only_graph_store_speaks_cypher():
    offenders = []
    for path in SOURCE_ROOT.rglob("*.py"):
        if path.name == "graph_store.py":
            continue
        if "cypher(" in path.read_text(encoding="utf-8"):
            offenders.append(path.relative_to(SOURCE_ROOT).as_posix())

    assert offenders == []
