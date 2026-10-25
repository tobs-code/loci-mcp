"""Build information for the running server.

`code_fingerprint` is computed from the Python files that are actually
running, not from a build argument. That makes a stale container visible: the
fingerprint of the server can be compared with the workspace.

    python -c "from mcp_server.version import code_fingerprint; print(code_fingerprint())"
"""

from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

VERSION = os.environ.get("LOCI_VERSION", "0.6.0")
STARTED_AT = datetime.now(timezone.utc).isoformat(timespec="seconds")


@lru_cache(maxsize=1)
def code_fingerprint() -> str:
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()[:12]
