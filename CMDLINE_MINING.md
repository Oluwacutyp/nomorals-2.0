# CMDLINE_MINING.md — `nomorals/cmdline/` external mining

Module: the `nm` command-line interface — `parser.py` (argparse tree, alias
map), `dispatch.py` (`main()`, dispatch, session/timeline attach, log config),
`emit.py` (JSON/prose output helper), `commands/` (79 domain command modules),
`commands/meta.py` (`nm help` / `nm commands` overview surfaces).

Mined 2026-10-10. Sources: public GitHub research docs, skill catalogs, and
framework docs (links in §6). Best AND trash were both mined — the "don't do
this" column is as important as the gold.

## 1. CLI framework choice: argparse vs click vs typer

**Best-in-class says:** typer wins for new CLIs (type-hint-driven, built-in
rich output, shell completion, `typer.Exit()` for clean exits); click for
deeply nested trees and plugin ecosystems; argparse when zero-dependency is a
constraint. One project (lucadeleo/gdoc research) deliberately stayed on
argparse: flat subcommand structure, agent-first consumers, plain output is
more token-efficient, no dependency weight.

**What nm does:** stdlib argparse, ~100 flat top-level subcommands, hand-built
`_parser()` (~3400 lines). This is the *correct* call for this codebase:
- Zero mandatory deps is a standing user order; typer pulls click+rich+shellingham.
- Flat (not nested) command tree — argparse's subparser pattern is sufficient.
- Agent-first consumers: plain `--json` output beats rich color for token efficiency.
- Startup latency matters on Termux/phone: importing typer/rich/click on every
  `nm` invocation is pure cost.

**Verdict: keep argparse. Steal from typer/click:**
- typer's `Exit()` semantics → nm already returns int exit codes from
  `main()`; keep that contract, document the code table (0/1/2/130).
- click's `@click.group()` + lazy command loading → nm's `dispatch.py` is a
  ~110-branch `if args.command == ...` chain. Best practice (click groups,
  OpenMMLab-style registries) is a **data-driven dispatch table**: one dict
  canonical-name → handler, single source of truth that also powers help,
  completion, and plugin registration.
- click's `CliRunner` in-process testing → nm already tests via real parser +
  `redirect_stdout`; keep, add table-driven dispatch coverage.

**Trash mined:** decorator-magic registries that only work if the module was
imported (dev.to "Python Registry Pattern" critique) — global state, import
side effects, debugging mess. nm will use an *explicit* table, not magic.

## 2. Command dispatch: if-chain → registry

**Best-in-class (dev.to "Dynamic Command Loading", OpenMMLab registry
pattern):** commands discovered/registered in one place; adding a command =
adding one entry, never touching the dispatcher. The decorator-registry
critique warns: registration must be explicit, not import-side-effect magic.

**What nm does:** `dispatch.py::_dispatch` is ~110 sequential `if`
comparisons, plus special-case branches (`models` sub-branches on
`model_action`, `skill` branches on `args.action`, settings-only commands
run outside `build_context`). Works, but every new command edits the
dispatcher; the parser, the alias map, and the dispatcher can drift.

**Gold to take:**
- `_COMMANDS: dict[str, _CommandSpec]` — canonical name → (handler,
  needs_context). `_dispatch` becomes: canonicalize → table lookup → run.
  Settings-only commands (`snapshot`, `recover`, `update`, `config`, `setup`)
  keep their no-context path via `needs_context=False`.
- Public `register_command()` so the plugin system can add CLI commands at
  runtime without editing dispatch (explicit, documented, test-covered).
- `difflib.get_close_matches` "did you mean?" on unknown commands — every
  good CLI (git, pip, gh) does this; nm prints a bare `unknown command: X`.
- Keep exact behavior: same handlers, same exit codes, same context lifecycle
  (`_attach_cli_session`, `_attach_timeline` inside the context block).

## 3. Terminal output: the missing style layer

**Best-in-class (rich, textual, dev.to "How to Build Beautiful TUIs"):**
- Centralized theme: one class owns colors, icons, text styles
  (`TngTheme`-style: `Colors` / `Icons` / `TextStyles` nested classes).
- Semantic styling over raw ANSI: `success`/`error`/`warning`/`info`, not
  scattered escape codes.
- TTY detection: strip color when piped (`Console(file=StringIO())` pattern),
  honor `NO_COLOR`, `TERM=dumb`.
- rich tables for tabular data; plain text stays readable without a TTY.

**What nm does:** `emit.py` is 16 lines — `_emit(args, payload, text)`:
`--json` → `json.dumps(payload)`; else `print(text)`. All prose is
hand-formatted strings scattered across 79 command modules. No theme, no
color, no table renderer. Honest but flat.

**Gold to take (zero-dep — rich is NOT installed here and the standing order
is zero mandatory deps):**
- `cmdline/style.py`: a `Theme` with semantic styles (title/success/error/
  warning/info/dim/accent), icons (✓ ✗ ⚠ ℹ →), auto TTY detection,
  `NO_COLOR`/`NM_NO_COLOR` support, global `--no-color` flag.
- `Table` renderer: auto-sized columns, truncation, header styling — used by
  `nm commands`, `nm help cli`, status surfaces. Degrades to plain aligned
  text when not a TTY (tests capture non-TTY → byte-identical output).
- `kv()` panel for key: value dumps (config, status sections).
- Spinner/progress context for long ops (stderr, zero-dep threading).
- **Never restyle existing `_emit` prose paths** — styling is opt-in via new
  helpers; existing tests and `--json` contracts stay byte-identical.
- Design the theme so a rich backend can be plugged in later without
  changing call sites.

**Trash mined:** color-only status indicators (accessibility failure —
always pair color with an icon/word); random accent colors per screen
(limit to one accent + semantic colors).

## 4. Shell completion: the biggest missing feature

**Best-in-class (shtab, edumatcher design doc, xylar/swage):**
- shtab: generate *static* completion scripts from the real argparse parser
  (bash/zsh/fish/tcsh/powershell), committed or printed on demand. Rejected
  argcomplete (runs the whole program on every Tab — import cost).
- edumatcher's decision: generate from the parsers as source of truth, never
  from hand-curated data (drift); expose as `<cli> completion bash|zsh`.
- swage: `completion bash` prints the hook; import the heavy deps only when
  the shell asks.

**What nm does:** nothing. ~100 commands × aliases × options, zero completion.
With this many commands this is the highest-value missing feature.

**Gold to take:**
- New `nm completion {bash,zsh,fish,powershell}` command walking the real
  `_parser()` tree: subcommands, aliases, options, `choices=` values —
  zero new dependencies, hand-rolled generator (shtab's approach, stdlib
  only). Parser stays the single source of truth → completion can never drift.
- Print install instructions with the script.

## 5. Help system

**What nm does (good bones):** `nm help cli` renders every command + alias
from the live parser (generated, never stale — exactly the edumatcher
"parsers are the source of truth" rule); `nm help <cmd>` is alias-aware;
`nm commands [filter]` groups the chat catalog.

**Gaps / gold:**
- `_cli_overview` renders as flat text; upgrade to the new table renderer
  (command | aliases | one-liner) when on a TTY — same content, better scan.
- Unknown-command path should suggest `nm help cli` / `nm help <closest>`.
- Add per-command example lines? The parser already carries long
  descriptions for the complex commands (studio, media, code); surface them
  — no new data needed.

## 6. Output contracts: JSON + exit codes

**Best-in-class (clig.dev guidance, jq-era conventions):** machine output
must be stable, parseable, and complete; errors go to stderr; exit codes are
documented (0 ok / 1 error / 2 usage / 130 interrupted).

**What nm does:** `_emit` dual-mode is already the right pattern
(`--json` → payload, else prose; tests use bare Namespaces without `--json`).
`main()` already maps SystemExit/KeyboardInterrupt/Exception → 2/130/1.

**Gold to take:**
- Document the exit-code contract in `--help` epilog and in `main()`'s
  docstring (it's currently tribal knowledge).
- Keep `--json` payloads untouched (jq-friendly: dicts/lists, no prose
  mixed in — already true).
- Add `--no-color` global flag (formatting option; pairs with `NO_COLOR`).

## 7. Sources

- Framework comparison: https://github.com/ironbellyorg/ironclaude/blob/HEAD/docs/research/research_installer_improvements_20251017.md
- CLI skill catalog: https://github.com/tyler-r-kendrick/agent-skills/blob/HEAD/skills/python/cli/AGENTS.md
- Click vs Typer vs argparse matrix: https://github.com/jaigouk/altoiddd/blob/HEAD/docs/research/20260222_cli_framework_comparison.md
- Deliberate argparse choice (agent-first): https://github.com/lucadeleo/gdoc/blob/HEAD/.planning/research/SUMMARY.md
- TUI architecture + theme centralization: https://github.com/neuralblitz/mito/blob/HEAD/.opencode/skills/cli-tui-development/SKILL.md
- Terminal UI best practices (clig.dev-derived): https://github.com/imjonezz/aimakerspace9/blob/HEAD/07_Deep_Agents/TUI_ENHANCEMENT_PLAN.md
- Typer+Textual+Rich integration: https://github.com/aeyeops/elysiactl/blob/HEAD/kb/latest-features-textual-rich-typer.md
- Centralized theme pattern: https://dev.to/dev-tngsh/how-to-build-beautiful-terminal-user-interfaces-in-python-bo6
- Dynamic command loading: https://Dev.To/d1d4c/making-python-clis-more-maintainable-a-journey-with-dynamic-command-loading-113
- Registry pattern + decorator critique: https://dev.to/dentedlogic/stop-writing-giant-if-else-chains-master-the-python-registry-pattern-ldm
- shtab (static completion from argparse): https://github.com/tqdm/shtab/blob/HEAD/docs/index.md
- Shell-completion design decision: https://github.com/johan162/edumatcher/blob/HEAD/docs-design/EduMatcher-Shell-completion.md
- argcomplete hook pattern: https://github.com/xylar/swage/commit/46b6d7dfdf2a006876c69102ef419b12345f53c6

## 8. What gets built (traceability)

| # | Gold | Lands in |
|---|------|----------|
| 1 | Data-driven dispatch table, explicit (no magic) | `dispatch.py`: `_COMMANDS`, `_dispatch` rewrite |
| 2 | `register_command()` plugin API | `dispatch.py` |
| 3 | did-you-mean on unknown commands | `dispatch.py::_suggest_command` |
| 4 | Zero-dep semantic theme (colors/icons/TTY/NO_COLOR) | `cmdline/style.py` (new) |
| 5 | Table + key/value renderers (TTY-styled, plain otherwise) | `cmdline/style.py` |
| 6 | Spinner for long operations | `cmdline/style.py` |
| 7 | `--no-color` global flag + `NM_NO_COLOR` env | `parser.py`, `style.py` |
| 8 | `nm completion {bash,zsh,fish,powershell}` from the live parser | `commands/completion.py` (new), `parser.py`, `CLI_ALIASES` |
| 9 | Exit-code contract documented | `dispatch.py::main` docstring, parser epilog |
| 10 | `nm help cli` overview via table renderer | `commands/meta.py::_cli_overview` (TTY only) |
| 11 | Dispatch-table coverage + completion + theme tests | `tests/test_cmdline_sweep.py` |

Non-goals (deliberate): no typer/click/rich migration (dependency order +
agent-first output); no decorator-magic auto-registration (explicit table);
no restyling of existing prose output (byte-identical non-TTY contracts).
