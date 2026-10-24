"""One read-scope request, resolved once.

Every read tool used to re-branch on ``project``, ``"*"``, ``"global"`` and
``include_global`` and then re-derive the table list, the scope labels and the
"which other scopes exist" warning. The rules are subtle (a project also sees
global rows, an omitted project means different things per tool, an unknown
project silently falls back to global), so each tool drifted a little.

``resolve_scope`` turns the raw arguments into a ``ScopeSelection``. The
selection owns the table choice (per read mode), the ``include_global``
narrowing, the readable scope labels and the unknown-project warning. Tools
keep their own warning wording - only the facts come from here.
"""

from __future__ import annotations

from dataclasses import dataclass

from mcp_server.scope_catalog import (
    _all_mem_tables,
    _is_global_scope,
    _norm_project,
    _retrieval_tables,
    _scopes_of_tables,
    _unknown_project_warning,
)

MODE_PROJECT = "project"
MODE_BROWSE = "browse"
MODE_BOOTSTRAP = "bootstrap"
MODE_LOOKUP = "lookup"

_MODES = frozenset({MODE_PROJECT, MODE_BROWSE, MODE_BOOTSTRAP, MODE_LOOKUP})


@dataclass(frozen=True)
class ScopeSelection:
    """A normalized read-scope request."""

    raw: str | None
    project: str | None
    include_global: bool
    all_projects: bool
    global_only: bool

    @property
    def explicit(self) -> bool:
        return bool(self.raw and str(self.raw).strip())

    def tables(self, cur, mode: str = MODE_PROJECT) -> list[str]:
        """Memory tables this selection covers in the given read mode.

        The modes mirror the documented scope defaults:

        - ``project``: a project (plus global), otherwise global only.
        - ``browse`` / ``bootstrap``: an omitted project covers every scope
          (browse) or only global (bootstrap); ``"*"`` covers every scope.
        - ``lookup``: an id without a project is searched in every scope.
        """
        if mode not in _MODES:
            raise ValueError(f"unknown scope read mode {mode!r}")
        if mode == MODE_LOOKUP:
            if self.global_only:
                tables = ["memories"]
            elif self.project:
                tables = _retrieval_tables(cur, self.project)
            else:
                tables = _all_mem_tables(cur)
        elif mode == MODE_BROWSE:
            if self.all_projects:
                tables = _all_mem_tables(cur)
            elif self.project:
                tables = _retrieval_tables(cur, self.project)
            elif self.global_only:
                tables = ["memories"]
            else:
                tables = _all_mem_tables(cur)
        elif mode == MODE_BOOTSTRAP:
            if self.all_projects:
                tables = _all_mem_tables(cur)
            elif self.project:
                tables = _retrieval_tables(cur, self.project)
            else:
                tables = ["memories"]
        else:
            tables = (_all_mem_tables(cur) if self.all_projects
                      else _retrieval_tables(cur, self.project))
        if (self.project or self.all_projects) and not self.include_global:
            tables = [table for table in tables if table != "memories"]
        return tables

    def scopes(self, cur, tables: list[str]) -> list[str]:
        """Readable scope names of ``tables`` in the same order."""
        return _scopes_of_tables(cur, tables)

    def other_scopes(self, cur, tables: list[str]) -> list[str]:
        """Scopes that exist but are not covered by ``tables``."""
        covered = set(_scopes_of_tables(cur, tables))
        return [scope for scope in _scopes_of_tables(cur, _all_mem_tables(cur))
                if scope not in covered]

    def unknown_warning(self, cur) -> str | None:
        """The fallback warning for a project that holds no memories."""
        if self.all_projects:
            return None
        return _unknown_project_warning(cur, self.project)

    def graph_filter(self) -> tuple[str, list[str]]:
        """SQL condition and parameters for the entity/relation mirror."""
        if self.project and self.include_global:
            return "(project = %s OR project = 'global')", [self.project]
        if self.project:
            return "project = %s", [self.project]
        if self.global_only:
            return "project = 'global'", []
        if self.all_projects and not self.include_global:
            return "project <> 'global'", []
        return "TRUE", []


def resolve_scope(project: str | None,
                  include_global: bool = True) -> ScopeSelection:
    """Normalize a raw ``project`` argument into a ``ScopeSelection``.

    ``"*"`` is recognized before normalization (otherwise it would become a
    project literally named ``"*"``). A blank name raises ``ValueError`` with
    the argument-naming message of ``_norm_project``.
    """
    raw = project
    all_projects = project == "*"
    normalized = None if all_projects else _norm_project(project)
    return ScopeSelection(
        raw=raw,
        project=normalized,
        include_global=include_global,
        all_projects=all_projects,
        global_only=not all_projects and _is_global_scope(raw),
    )
