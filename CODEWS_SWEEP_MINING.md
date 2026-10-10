# codews sweep — external mining

Module: `nomorals/codews/` — `patch.py` (diff review/apply/preview/record),
`run.py` (test/build runners), `workspace.py` (`CodeWorkspace` git handle).
Mined 2026-10-10. Every technique below comes from a real implementation;
nothing invented.

## 1. patch.py — diff parsing, fuzzy application, edit blocks

### unidiff (matiasb/python-unidiff) — what a diff parser should expose
- Sources: https://github.com/matiasb/python-unidiff, https://pypi.org/project/unidiff/0.7.1
- Verified behaviors: `PatchSet` parses unified diffs into `PatchedFile`
  objects with `is_added_file` / `is_removed_file`, `added` / `removed`
  counts, per-`Hunk` objects carrying `source_line_no` / `target_line_no`
  per line, `source_mode` / `target_mode` and `is_symlink` (v1.0.0), git
  mnemonic prefixes (`i/ w/ c/ o/`, `1/ 2/`), and `diff_line_no` to locate
  hunkless (binary/mode-only) entries. `str(patched_file)` round-trips one
  file's diff.
- Gold taken: `review_patch` gains per-file `is_added`/`is_removed`,
  `old_mode`/`new_mode`, `is_symlink`, rename similarity, and per-hunk
  headers (old/new start+length, section heading). New `split_patch()`
  returns per-file diff texts (unidiff's `str()` round-trip idea).

### GNU patch fuzz factor — documented inexact-match semantics
- Source: http://www.gnu.org/software/diffutils/manual/html_node/Inexact.html
- Verified behaviors: first guess = hunk line number ± previous offset;
  then scan forward AND backward for full-context match; if max fuzz ≥ 1,
  re-scan ignoring the first and last context lines; fuzz 2 ignores the
  first two and last two; default fuzz is 2; larger fuzz raises the odds of
  a faulty patch.
- Gold taken: new `fuzz: int = 0` parameter on `apply_patch` /
  `preview_patch`. Default 0 stays exact (like `git apply`, which requires
  all context to match — see drush below). With `fuzz > 0`, on
  `DiffApplyError` the section is rewritten per GNU semantics (up to
  `fuzz` leading/trailing context lines dropped per hunk, `@@` headers
  recomputed) and retried through the existing engine.

### drush issue #4193 — fuzzy applies must be VISIBLE
- Source: https://github.com/drush-ops/drush/issues/4193
- Verified behavior/lesson: `git apply` needs exact context; the GNU
  `patch` fallback applies with fuzz=2 silently, "a patch applied fuzzily
  appears to be every bit as successful as a patch applied with no fuzz —
  the user has no indication that there was anything amiss."
- Gold taken: every `apply_patch` result dict gains `"fuzzy": bool`
  (and `"fuzz"` level used). A fuzzy success is never reported as a clean
  apply.

### aider edit formats — SEARCH/REPLACE cascade
- Sources: https://github.com/derric01/clutchcode/blob/HEAD/research/repos/aider.md
  (studied aider source: `editblock_coder.py:127-240` — cascade
  `perfect_replace` → whitespace-drift repair → blank-line drop →
  `try_dotdotdots`; `replace_closest_edit_distance` deliberately DISABLED —
  "fuzzy apply silently misplaces edits"),
  https://github.com/clemens865/lazy-fetch/blob/HEAD/research/frameworks/aider.md
- Verified behaviors: SEARCH/REPLACE block format
  (`<<<<<<< SEARCH` / `=======` / `>>>>>>> REPLACE`), empty SEARCH =
  create file, first-match-only, context control via files added to chat.
- Gold taken: new `apply_edit_blocks()` — the edit format LLMs actually
  emit reliably — with aider's ladder: exact → trailing-whitespace
  normalized → uniform leading-whitespace drift → spurious leading blank
  line → `difflib.SequenceMatcher` fuzzy (threshold-gated, strategy
  reported). Ambiguous (multi-match) blocks are refused, never
  first-match-guessed.

### apply-edit-block (pjdurden) — strategy-ladder library
- Source: https://github.com/pjdurden/apply-edit-block/blob/HEAD/python/README.md
- Verified behaviors: `apply_edit` returns the winning strategy name
  (`'trailing-ws'`, …), `parse_blocks` parses fenced blocks agents emit,
  `similarity()` scoring exposed.
- Gold taken: per-block result reports the winning `"strategy"`, so the
  agent sees how loose the match was.

### llm-patch (trickl) — fuzzy unified-diff application for LLM output
- Source: https://github.com/trickl/llm-patch
- Verified behaviors: `PatchApplier(similarity_threshold=0.8)`,
  `FuzzyMatcher.find_best_match(source_lines, pattern_lines)`.
- Gold taken: corroborates the fuzz approach above; threshold concept
  reused for the SequenceMatcher ladder in edit blocks.

### llm-search-replace (renatmursalimov) — why unified diff fails for LLMs
- Source: https://github.com/renatmursalimov/llm-search-replace
- Verified behavior/argument: a unified diff is "a description of an edit
  *plus arithmetic*" — the `@@ -47,7 +47,9 @@` header is what models get
  wrong (off-by-one), so `patch` rejects output that was 95% right.
- Gold taken: justification for adding `apply_edit_blocks` alongside
  unified-diff application rather than unified-diff only.

## 2. run.py — test/build runners

### junitparser — JUnit XML as the lingua franca of test results
- Source: https://github.com/weiwei/junitparser/blob/HEAD/README.rst
- Verified behaviors: parses pytest/maven/surefire JUnit XML
  (`pytest --junitxml`), `TestCase.result` is a list of
  `Failure`/`Error`/`Skipped` (v2+ fixed pytest multi-result support),
  per-case `classname`, `time`, merge support, CLI + `python -m`.
- Gold taken: `run_tests` runs pytest with `--junitxml` into a temp file
  and parses it with stdlib `xml.etree` (zero new deps) into structured
  `failures: [{nodeid, classname, duration_s, message}]`,
  plus `skipped`, `duration_s`, and a `junit_path` echo. New public
  `parse_junit_xml(path)` for any JUnit file (surefire, jest-junit, …).

### CI problem matchers / short summary — cheap failure extraction
- Verified behavior: pytest's `-q` "short test summary info" section
  prints `FAILED <nodeid>` lines; this is what GitHub Actions
  problem-matchers scrape.
- Gold taken: when no JUnit XML is available (non-pytest runners),
  `run_tests` scrapes `FAILED <nodeid>` lines into `failed_tests`.

### detect-stack skill (aethries/dotfiles) + spicashield tech signatures
- Sources: https://github.com/aethries/dotfiles/blob/HEAD/resources/skills/detect-stack/SKILL.md,
  https://github.com/hostspicaindia/spicashield/blob/HEAD/skills/spicashield/reference/tech-detection-signatures.md
- Verified approach: marker files first (`package.json`, `Cargo.toml`,
  `go.mod`, `pyproject.toml`, `pom.xml`, `build.gradle`, `Makefile`),
  then manifest dependencies, then directory conventions; multiple stacks
  can co-exist.
- Gold taken: new `detect_stack(root)` — marker-file → command table:
  tests: `package.json` scripts.test (jest/vitest/mocha via devDeps) →
  `go test ./...` → `cargo test` → pytest config markers →
  `make test` → `test_*.py` layout → `unittest`;
  builds: `make` → `npm run build` → `cargo build` → `go build ./...` →
  `gradle build` → `mvn package` → PEP 517 `python -m build`
  (reads `[build-system]` per PEP 518: https://www.python.org/dev/peps/pep-0517).
  `run_tests`/`run_build` now dispatch through it instead of the old
  pytest-or-nothing chain.

### ineersa/agent-core — LLM patches need higher fuzz
- Source: https://github.com/ineersa/agent-core/commit/c2957c67b43ad32f8a5fffe71759679587f01122
- Verified behavior: bumped GNU patch fuzz to 5 for LLM "plain @@ hunks"
  after normalizing headers; fuzz anchors correctly at computed offset.
- Gold taken: corroborates exposing a tunable `fuzz` knob (not a fixed
  default) on the patch side.

## 3. workspace.py — git handle

### GitPython — the reference Python git object model
- Source: https://gitpython.readthedocs.io/en/2.1.6/index.html
- Verified API surface mined: `Repo.remotes` / `create_remote` /
  `delete_remote`, `TagReference` create/delete, `repo.is_dirty()`,
  `index.diff`, `untracked_files`, `head.reset`, `archive`, refs and
  `tracking_branch` with ahead/behind.
- Gold taken (as fail-fast subprocess methods, keeping this module's own
  runner): `remotes()`, `remote_add/remove/set_url`, `tags()`,
  `create_tag/delete_tag`, `is_dirty()`, `merge_branch()` with conflict
  detection, `abort_merge()`, `cherry_pick()`, `revert()`,
  `delete_branch()`, `rename_branch()`, `add()`, `restore()`,
  `clean()` (dry-run default), `stash_apply()`/`stash_drop()`,
  `show()`, `worktree_prune()`, per-branch `upstream`/`ahead`/`behind`
  (via `git for-each-ref`, the `git branch -vv` data source), `conflicts()`
  from porcelain v2 `u` lines, renames parsed from porcelain v2 `2` lines,
  `blame()` from `git blame --porcelain`, `file_history()` from
  `git log --follow`, `diff_stat()` from `git diff --numstat`,
  `diff(ref_a, ref_b, paths)`, `log(..., paths, stats)` with numstat.

### git porcelain formats (verified against git documentation/behavior)
- `git status --porcelain=v2`: `2 <XY> … <to>\t<from>` rename lines;
  `u <XY> … <path>` unmerged lines (already partially parsed).
- `git worktree list --porcelain`: `locked` lines, `prunable` marker —
  parsed into `worktree_list()`; `git worktree prune` added.
- `git branch -vv` data comes from `git for-each-ref` with
  `%(upstream:short)`, `%(ahead)`, `%(behind)`.

## Deliberately NOT taken
- aider's disabled `replace_closest_edit_distance` for unified diffs —
  silent misplacement; our fuzzy path is opt-in, threshold-gated, and
  always reported (drush lesson).
- GitPython itself as a dependency — this module's standing design is a
  small subprocess runner with `WorkspaceError` semantics (faster,
  no new dep, works on Termux); only the API surface was mined.
- junitparser as a dependency — stdlib `xml.etree` covers parsing; the
  *format* (JUnit XML via `pytest --junitxml`) was the gold.
