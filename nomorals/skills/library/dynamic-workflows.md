---
id: dynamic-workflows
title: Dynamic workflows — saved, reviewable agent scripts
summary: How multi-agent work runs as code: `/workflow <task>` writes a synchronous Python script (phase, log, agent, parallel, pipeline), saves it durably, shows it in chat, and runs it on the scheduler clock — no review gate, schema-typed handoffs, per-agent worktrees, multi-turn handles.
topics: workflows, agents, scripts, delegation, parallel, review
triggers: workflow, run workflow, agent script, multi-agent script
layer: agents
version: 1
---

# Dynamic workflows — saved, reviewable agent scripts

A workflow is a short synchronous Python script that coordinates
roster agents. `/workflow <task>` writes one, saves it durably,
shows it in chat, and runs it on the scheduler clock. A saved script starts
in the `queued` state; the runner picks it up on the next
tick and it runs in the background. The workflow
is CODE, not a chat transcript: reviewable, versionable,
re-runnable with `/workflow run <name>`.

**No review-before-run gate** (owner order 2026-11-16): a script
runs as soon as it is written. It is saved durably and shown in
chat so the owner can look if he wants — nothing blocks on
approval.

## The script form (synchronous — no `await`)

Devon scripts are SYNCHRONOUS Python. There is no `await`:
`agent(...)` returns the result directly (spawn + bounded wait).
Anything else is the Antigravity shape.

```python
phase("Discover")
files = agent("List the Python files changed since main. Paths only.",
              schema={"type": "array", "items": {"type": "string"}})
log("Found", len(files), "files")
phase("Review")
reviews = parallel(files, lambda f: agent(
    "Review " + f + " for unhandled exceptions (read files only), "
    "and summarize the riskiest issue.",
    schema={"type": "string"}, role="reviewer"))
for review in reviews:
    log(review)
print("Done")
```

## Primitives

- `phase(name)` — starts a named phase; logs and agent results
  group under it in the durable journal. The previous open phase
  is closed.
- `log(*args)` — one line in the current phase AND in the result
  the owner receives. `print(...)` does the same.
- `agent(prompt, schema=None, *, role="planner", input=None,
  workspace="inherit", name="")` — runs one roster agent and
  returns its schema-validated result. `schema` is a JSON Schema;
  omit it for a free-form string (`{"type": "string"}`).
  A failure (unknown role, missing required input, timeout,
  worker error, schema failure) is returned as
  `{"ok": False, "error": ...}` — the script decides what to do
  with it. `agent(role="reviewer")` with NO prompt returns a
  multi-turn handle instead: `handle(prompt, schema)` runs the
  next turn of the same agent, carrying earlier turns as context.
- `parallel(items, fn)` — runs `fn(item)` for every item at the
  same time, results in item order. One item failing does NOT
  stop the others — the script receives an error dict for that
  item.
- `pipeline(items, fn)` — runs `fn(item)` one item at a time,
  results in item order, same per-item failure isolation.
- `trajectory(limit=10, kind=None)` — earlier steps of the
  current conversation, oldest first; `kind` filters by message
  role when given.
- `args` — dict of arguments passed to the script.
- `CONVERSATION_ID`, `STEP_INDEX`, `WORKSPACES` — the origin
  chat, the run id, the workspace folders (primary repo first).

## Roster — the only roles a script may name

`planner`, `implementer`, `reviewer`, `tester`, `refactorer`,
`dep_hunter`, `api_designer`. An eighth role fails, naming the
roster. Required inputs (a missing one fails with the field
named):

- `planner`: `task`
- `implementer`: `path`, `instruction`
- `reviewer`: `diff`
- `tester`: (none — `focus` is accepted)
- `refactorer`: `path`, `goal`
- `dep_hunter`: `name`
- `api_designer`: `feature`

A bare goal alone can spawn planner / reviewer / tester /
dep_hunter / api_designer. `implementer` and `refactorer` need a
`path` — pass it via `input={"path": ..., ...}`.

## Workspaces

- `workspace="inherit"` (default) — the agent works in the repo
  itself.
- `workspace="branch"` — the agent gets its own isolated git
  worktree (base = current HEAD); the result reports the
  worktree path.
- `workspace="share"` — a fresh working copy (clone) instead.

## Rules a script must follow

- No imports, no `open` / `exec` / `eval` / `__import__` /
  `compile`, no dunder access, no `while` loops (finite work
  uses `for` / `parallel` / `pipeline` over a list). A script
  breaking these is rejected before it runs, with the line
  named. Scripts coordinate agents; they never touch the
  filesystem, the network, or Devon's internals directly.
- Agents do NOT see the script or each other — put everything
  an agent needs in its prompt and ask for a short,
  well-defined result.
- Keep prompts self-contained and results small: the script
  receives the parsed value, and the owner receives the log.
- A run's record survives a restart (durable store + phase
  journal); a live thread does not. An interrupted run is
  re-run from the top with `/workflow resume <name>`.
