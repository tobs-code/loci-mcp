"""The MCP application: the FastMCP instance and its tool registration.

Every tool answers with the same envelope, so an agent never has to guess the
shape of an answer. That promise also holds for *invalid* calls: a wrong
argument type or a missing required argument must not escape as a raw
validation error, it has to come back as `{"ok": false, "error": ...}`.
"""

from __future__ import annotations

import logging

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.tools.base import Tool
from mcp.server.fastmcp.tools.tool_manager import ToolManager
from pydantic import ValidationError

from mcp_server.envelope import STATUS_ERROR, fail

logger = logging.getLogger(__name__)


def _format_validation_error(exc: ValidationError) -> str:
    """Readable one-liner for a pydantic validation error."""
    parts: list[str] = []
    for error in exc.errors():
        location = ".".join(str(item) for item in error.get("loc", ())) or "arguments"
        message = str(error.get("msg", "invalid value"))
        parts.append(f"{location}: {message}")
    return "invalid arguments - " + "; ".join(parts)


class EnvelopeTool(Tool):
    """A tool that answers with the envelope, also when the call is invalid.

    The base class lets a pydantic `ValidationError` escape as a `ToolError`.
    The caller would then see "1 validation error for recallArguments ..."
    instead of the documented envelope, which breaks the contract that a
    failed call never raises.
    """

    async def run(self, arguments, context=None, convert_result: bool = False):
        result = None
        try:
            parsed = self.fn_metadata.pre_parse_json(arguments)
            self.fn_metadata.arg_model.model_validate(parsed)
        except ValidationError as exc:
            result = fail(STATUS_ERROR, _format_validation_error(exc))
        except Exception as exc:  # pragma: no cover - defensive
            result = fail(STATUS_ERROR, f"invalid arguments: {exc}")
        if result is None:
            try:
                return await super().run(arguments, context, convert_result)
            except Exception as exc:
                result = fail(STATUS_ERROR, _error_message(exc))
        if convert_result:
            return self.fn_metadata.convert_result(result)
        return result


def _error_message(exc: Exception) -> str:
    """Message of a failed tool call, without the SDK's tool-name prefix."""
    cause = exc.__cause__ or exc
    message = str(cause) or exc.__class__.__name__
    prefix = "Error executing tool "
    text = str(exc)
    if text.startswith(prefix):
        text = text[len(prefix):].split(":", 1)[-1].strip()
        return text or message
    return message


class EnvelopeToolManager(ToolManager):
    """Tool manager that registers the envelope-aware tool class."""

    def add_tool(self, fn, **kwargs) -> Tool:
        tool = EnvelopeTool.from_function(fn, **kwargs)
        existing = self._tools.get(tool.name)
        if existing is not None:
            if self.warn_on_duplicate_tools:
                logger.warning("Tool already exists: %s", tool.name)
            return existing
        self._tools[tool.name] = tool
        return tool


class LociFastMCP(FastMCP):
    """FastMCP that uses the envelope-aware tool manager."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        manager = self._tool_manager
        self._tool_manager = EnvelopeToolManager(
            warn_on_duplicate_tools=manager.warn_on_duplicate_tools,
            tools=list(manager._tools.values()),
        )


app = LociFastMCP(
    name="loci",
    instructions="""Persistent memory tools for agents.

TOOLS BY TASK
- Store/change/remove: `remember`, `remember_batch`, `update_memory`,
  `update_batch`, `forget`.
- Search/browse/bootstrap: `recall`, `list_memories`, `get_entity`,
  `list_entities`, `init_context`.
- Graph: `relate`, `relate_batch`, `unrelate`, `find_connections`, `find_paths`.
- Maintenance/backup: `server_health`, `contradictions_report`,
  `export_memories`, `import_memories`, `list_projects`, `drop_project`.
- Repair/refresh: `touch_memory` (bump `updated_at`, no content change),
  `restore_memory` (undo a soft delete).

COMMON FLOWS
- Start/search: `init_context(project="api")` → `recall(query="deploy steps", project="api")`.
- Store/link: `remember(content="Service uses Postgres", entity="API", project="api")` →
  `relate(from_entity="API", to_entity="Postgres", relation_type="USES", project="api")`.
- Explore: `find_connections(entity="API", project="api")` or
  `find_paths(from_entity="API", to_entity="Postgres", project="api")`.
- Backup/restore: pass `export_memories(format="json").data` to
  `import_memories(data=..., format="json")`.

Read-scope defaults differ by tool; check each tool's description and pass
`include_global=false` when a project-only read is required. Collection tools
return items in `results`; export and import payloads use `data`.
Every call answers with the same envelope: {"ok", "status", "id", "error",
"warnings"}. A failed call has `ok: false` and never raises.

A write reports the `scope`/`project` it really used, and a read reports the
`scopes` it searched: trust those names for the next call instead of
guessing. An empty `recall` names every filter that emptied the result, so
read the `warnings` before concluding that a memory does not exist.
""",
    streamable_http_path="/",
    json_response=True,
)


__all__ = ["app", "EnvelopeTool", "EnvelopeToolManager", "LociFastMCP"]
