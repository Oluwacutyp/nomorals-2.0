"""``nm repo`` — the Wave K code workspace: repo/branch/worktree/patch/test/build."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def _ws(args: Any):
    from ...codews import CodeWorkspace

    root = str(Path(getattr(args, "root", "") or ".").expanduser().resolve())
    return CodeWorkspace(root)


def _cmd_repo(args: Any, context: Any) -> int:
    """Route ``nm repo <status|branches|branch|switch|diff|log|worktree|patch|test|build|commit|push|pull|fetch|stash>``."""
    from ...codews import WorkspaceError

    words = list(getattr(args, "task", None) or [])
    if not words:
        print("usage: nm repo status|branches|diff [ref]|log [n] [--root DIR]\n"
              "       nm repo branch <name> | nm repo switch <name>\n"
              "       nm repo worktree <add|list|remove> [path] [branch] [--force]\n"
              "       nm repo patch review|apply|preview|record <file> [path] [--root DIR]\n"
              "       nm repo test [selector] | nm repo build [target] [--root DIR]\n"
              "       nm repo commit -m \"msg\" [paths...] | nm repo push|pull|fetch [remote] [branch]\n"
              "       nm repo stash <push|pop|list> [-m \"msg\"]",
              file=sys.stderr)
        return 2
    verb = words[0]
    try:
        if verb == "status":
            return _repo_status(args)
        if verb == "branches":
            return _repo_branches(args)
        if verb == "branch":
            return _repo_branch(args, words[1:])
        if verb == "switch":
            return _repo_switch(args, words[1:])
        if verb == "diff":
            return _repo_diff(args, words[1:])
        if verb == "log":
            return _repo_log(args, words[1:])
        if verb == "worktree":
            return _repo_worktree(args, words[1:])
        if verb == "patch":
            return _repo_patch(args, context, words[1:])
        if verb == "test":
            return _repo_test(args, words[1:])
        if verb == "build":
            return _repo_build(args, words[1:])
        if verb == "commit":
            return _repo_commit(args, words[1:])
        if verb == "push":
            return _repo_push_pull(args, "push", words[1:])
        if verb == "pull":
            return _repo_push_pull(args, "pull", words[1:])
        if verb == "fetch":
            return _repo_fetch(args, words[1:])
        if verb == "stash":
            return _repo_stash(args, words[1:])
        print(f"unknown repo verb: {verb}", file=sys.stderr)
        return 2
    except WorkspaceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _repo_status(args: Any) -> int:
    st = _ws(args).status()
    if getattr(args, "json", False):
        print(json.dumps(st, indent=2))
        return 0
    print(f"branch: {st['branch']}  ahead {st['ahead']} / behind {st['behind']}")
    for label in ("staged", "unstaged", "untracked"):
        for path in st[label]:
            print(f"  {label[0].upper()}: {path}")
    if not any(st[k] for k in ("staged", "unstaged", "untracked")):
        print("clean")
    return 0


def _repo_branches(args: Any) -> int:
    branches = _ws(args).branches()
    if getattr(args, "json", False):
        print(json.dumps(branches, indent=2))
        return 0
    for b in branches:
        mark = "*" if b["current"] else " "
        print(f"{mark} {b['name']}")
    return 0


def _repo_branch(args: Any, rest: list[str]) -> int:
    if not rest:
        print("usage: nm repo branch <name>", file=sys.stderr)
        return 2
    _ws(args).create_branch(rest[0])
    print(f"created branch {rest[0]}")
    return 0


def _repo_switch(args: Any, rest: list[str]) -> int:
    if not rest:
        print("usage: nm repo switch <name>", file=sys.stderr)
        return 2
    _ws(args).switch_branch(rest[0])
    print(f"switched to {rest[0]}")
    return 0


def _repo_diff(args: Any, rest: list[str]) -> int:
    ref = rest[0] if rest else ""
    print(_ws(args).diff(ref), end="")
    return 0


def _repo_log(args: Any, rest: list[str]) -> int:
    n = int(rest[0]) if rest and rest[0].isdigit() else 10
    entries = _ws(args).log(n)
    if getattr(args, "json", False):
        print(json.dumps(entries, indent=2))
        return 0
    for e in entries:
        print(f"{e['sha'][:8]}  {e['author'][:20]:20}  {e['message'][:70]}")
    return 0


def _repo_worktree(args: Any, rest: list[str]) -> int:
    ws = _ws(args)
    if not rest or rest[0] == "list":
        items = ws.worktree_list()
        if getattr(args, "json", False):
            print(json.dumps(items, indent=2))
            return 0
        for item in items:
            print(f"{item['path']}  [{item['branch']}]  {item['sha'][:8]}")
        return 0
    if rest[0] == "add":
        if len(rest) < 3:
            print("usage: nm repo worktree add <path> <branch>", file=sys.stderr)
            return 2
        ws.worktree_add(rest[1], rest[2])
        print(f"worktree {rest[1]} on branch {rest[2]}")
        return 0
    if rest[0] == "remove":
        if len(rest) < 2:
            print("usage: nm repo worktree remove <path> [--force]", file=sys.stderr)
            return 2
        ws.worktree_remove(rest[1], force=getattr(args, "force", False))
        print(f"removed worktree {rest[1]}")
        return 0
    print(f"unknown worktree verb: {rest[0]}", file=sys.stderr)
    return 2


def _repo_patch(args: Any, context: Any, rest: list[str]) -> int:
    from ...codews import apply_patch, preview_patch, record_patch, review_patch

    if len(rest) < 2:
        print("usage: nm repo patch review|apply|preview|record <diff-file> [path] [--root DIR]",
              file=sys.stderr)
        return 2
    verb, path = rest[0], rest[1]
    root = str(Path(getattr(args, "root", "") or ".").expanduser().resolve())
    diff_text = Path(path).expanduser().read_text(encoding="utf-8")
    if verb == "review":
        report = review_patch(diff_text)
        if getattr(args, "json", False):
            print(json.dumps(report, indent=2))
            return 0
        t = report["totals"]
        print(f"{t['files']} file(s): +{t['additions']} -{t['deletions']}")
        for f in report["files"]:
            print(f"  {f['status']:8} {f['path']} +{f['additions']} -{f['deletions']}")
        return 0
    if verb == "apply":
        results = apply_patch(diff_text, dry_run=not getattr(args, "yes", False),
                              root=root)
        failed = [r for r in results if not r["ok"]]
        if getattr(args, "json", False):
            print(json.dumps(results, indent=2))
        else:
            for r in results:
                status = "OK" if r["ok"] else f"FAILED: {r['error']}"
                print(f"  {r['path']}: {status}")
        return 1 if failed else 0
    if verb == "preview":
        if len(rest) < 3:
            # preview the first file in the diff
            files = review_patch(diff_text)["files"]
            if not files:
                print("no files in diff", file=sys.stderr)
                return 2
            target = files[0]["path"]
        else:
            target = rest[2]
        snip = preview_patch(diff_text, target, root=root)
        print("--- before ---")
        print(snip["before"])
        print("--- after ---")
        print(snip["after"])
        return 0
    if verb == "record":
        db = getattr(context, "db", None)
        if db is None:
            print("error: no database context for artifact recording", file=sys.stderr)
            return 1
        from ...storage.artifacts import ArtifactStore
        from ...storage.blob import BlobStore

        db_path = getattr(db, "path", None)
        blob_dir = Path(db_path).parent / "blobs" if db_path else Path("data/blobs")
        uri = record_patch(ArtifactStore(db, BlobStore(db, blob_dir)), diff_text)
        print(uri)
        return 0
    print(f"unknown patch verb: {verb}", file=sys.stderr)
    return 2


def _repo_test(args: Any, rest: list[str]) -> int:
    from ...codews import run_tests

    root = str(Path(getattr(args, "root", "") or ".").expanduser().resolve())
    result = run_tests(root, rest[0] if rest else "")
    if getattr(args, "json", False):
        print(json.dumps({k: v for k, v in result.items() if k != "output"},
                         indent=2))
    else:
        print(f"runner: {result['runner']}  "
              f"passed={result['passed']} failed={result['failed']}  "
              f"{'OK' if result['ok'] else 'FAILED'}")
    if not getattr(args, "json", False) and not result["ok"]:
        print(result["output"][-4000:])
    return 0 if result["ok"] else 1


def _repo_build(args: Any, rest: list[str]) -> int:
    from ...codews import WorkspaceError, run_build

    root = str(Path(getattr(args, "root", "") or ".").expanduser().resolve())
    try:
        result = run_build(root, rest[0] if rest else "")
    except WorkspaceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if getattr(args, "json", False):
        print(json.dumps({k: v for k, v in result.items() if k != "output"},
                         indent=2))
    else:
        print("OK" if result["ok"] else "FAILED")
    if not result["ok"]:
        print(result["output"][-4000:])
    return 0 if result["ok"] else 1


def _repo_commit(args: Any, rest: list[str]) -> int:
    message = getattr(args, "message", "") or ""
    paths = list(rest)
    if not message:
        # allow `nm repo commit "message" [paths...]` as well
        if not rest:
            print('usage: nm repo commit -m "message" [paths...]', file=sys.stderr)
            return 2
        message, paths = rest[0], rest[1:]
    res = _ws(args).commit(message, paths or None)
    if getattr(args, "json", False):
        print(json.dumps(res, indent=2))
    else:
        print(f"committed {res['sha'][:8]}: {res['message'][:70]}")
    return 0


def _repo_push_pull(args: Any, which: str, rest: list[str]) -> int:
    ws = _ws(args)
    remote = rest[0] if len(rest) > 0 else "origin"
    branch = rest[1] if len(rest) > 1 else ""
    res = ws.push(remote, branch) if which == "push" else ws.pull(remote, branch)
    if getattr(args, "json", False):
        print(json.dumps(res, indent=2))
    else:
        print(f"{which}ed {res['remote']}" + (f" {res['branch']}" if res["branch"] else ""))
    return 0


def _repo_fetch(args: Any, rest: list[str]) -> int:
    remote = rest[0] if rest else "origin"
    res = _ws(args).fetch(remote)
    if getattr(args, "json", False):
        print(json.dumps(res, indent=2))
    else:
        print(f"fetched {res['remote']}")
    return 0


def _repo_stash(args: Any, rest: list[str]) -> int:
    ws = _ws(args)
    if not rest or rest[0] == "list":
        items = ws.stash_list()
        if getattr(args, "json", False):
            print(json.dumps(items, indent=2))
            return 0
        for item in items:
            print(f"stash@{{{item['index']}}}: {item['message']}")
        if not items:
            print("no stashes")
        return 0
    if rest[0] == "push":
        message = getattr(args, "message", "") or ""
        res = ws.stash_push(message)
        if getattr(args, "json", False):
            print(json.dumps(res, indent=2))
        else:
            print("stashed" + (f": {message}" if message else ""))
        return 0
    if rest[0] == "pop":
        ws.stash_pop()
        print("popped stash")
        return 0
    print("unknown stash verb: " + rest[0], file=sys.stderr)
    return 2
