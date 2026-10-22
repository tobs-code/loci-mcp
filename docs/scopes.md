# Scopes & Projects

Every memory belongs to exactly one scope. There is one global scope and any
number of project scopes.

## The two kinds of scope

- The **global scope** lives in the `memories` table. It is shared by every
  project and holds the facts that are true everywhere (team conventions, user
  preferences).
- A **project scope** lives in its own table `mem_<suffix>`. It holds the facts
  of one project.

A project search sees the project scope **and** the global scope, so global
facts stay visible from inside a project. When the same content exists in
both, the project copy wins.

## Project names

Project names may contain any characters. Plain lower-case identifiers up to
40 characters are used verbatim as the table suffix (backwards compatible).
All other names are lower-cased, truncated and get a short SHA-1 of the
*original* name appended:

```
Project Alpha-2  ->  project_alpha_2_e6a52393
Team             ->  team
team             ->  team_9f2c1ab4     (different table)
```

This keeps table and index names below PostgreSQL's 63-character limit, so
long names and case variants cannot share a table.

`global` is reserved: `project="global"` means "no project" (the global
scope) and never creates a project of that name.

The `projects` registry maps suffix → original name, so results show the name
the caller passed. Existing plain-identifier tables keep working unchanged.

## The graph per scope

The graph holds one Apache AGE label per scope (`Entity` for global,
`Project_<suffix>` for a project) plus a `project` property. The SQL mirror
`entities` has `UNIQUE(name, project)`, so identical entity names do not
collide across projects.

Graph tools use the global graph when `project` is omitted.

## Which tools accept `project`

`get_entity`, `list_entities`, `list_memories`, `recall`, `init_context`,
`relate`, `relate_batch`, `unrelate`, `find_connections`, `find_paths`,
`forget`, `restore_memory`, `export_memories`, `import_memories`,
`drop_project`.

The read defaults differ per tool and are listed in
[Retrieval & Scoring](retrieval.md#read-scope-defaults). For `init_context`,
use `project="*"` to include all scopes; `recall` understands `project="*"`
as well and then searches every scope at once.

A project that does not exist is never silently ignored: `recall`,
`list_memories`, `list_entities` and `init_context` fall back to the global
scope and say so in `warnings`.

## Memory ids are unique per scope

A memory id is unique **per scope, not globally**: an import copies the id
into the target scope, and then the same id exists in two scopes.

`update_memory`, `restore_memory` and `forget(memory_id=...)` therefore refuse
an id that exists in several scopes and name the scopes in the error:

```json
{"ok": false, "status": "error",
 "error": "ambiguous memory id 980937cb790c45f1: it exists in the scopes
           ['api', 'api-copy']; pass project=<name> to say which one to update",
 "scopes": ["api", "api-copy"]}
```

With `project` the call is unambiguous, and the answer reports the scope that
was changed. `forget(entity=...)` keeps removing the entity from every scope.

## The lifecycle of a project

| Tool | Effect |
|------|--------|
| `list_projects` | Lists every scope with `memories`, `total_memories`, `superseded_memories`, `forgotten_memories`, `entities` and `relations`. |
| `forget(entity="*", project=…)` | **Empties** the scope: memories, graph nodes and mirror rows are removed, the scope itself survives. The shared scope is addressed with `project="global"`. |
| `drop_project(project=…)` | **Removes** the scope completely: memory table, registry entry, graph nodes and mirror rows. `dry_run=true` previews the counts. `global` is refused. |

`forget(entity="*", project="global")` empties the shared global scope the
same way (soft or hard); the wildcard without any `project` is refused, so a
stray call cannot wipe the whole database.

`drop_project` is permanent and refuses `global`. Use it when a project is
finished and its scope should disappear from every listing.

## Storing outside a project

`remember(scope="global")` writes to the shared `memories` table even when a
project is given — for facts that are true everywhere.
