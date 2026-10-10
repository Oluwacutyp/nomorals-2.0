# TUI Sweep — Mining Notes

Module: `nomorals/tui/` (model.py, app.py, __init__.py) — a pure-model + thin-curses-driver chat TUI.
Mined before any code was written. Every technique below is taken from a real implementation, not invented.

## 1. Input editing — GNU readline emacs mode + prompt_toolkit + sml-readline

Sources:
- GNU readline emacs cheat sheets: https://catonmat.net/ftp/readline-emacs-editing-mode-cheat-sheet.pdf and https://zhenbangcheng.com/assets/readline-emacs-editing-mode-cheat-sheet.pdf
- prompt_toolkit feature list: https://pypi.org/project/prompt_toolkit/
- willibrandon/stroke emacs spec: https://github.com/willibrandon/stroke/blob/HEAD/specs/042-emacs-key-bindings/spec.md
- sjqtentacles/sml-readline (pure SML line-editor state machine): https://github.com/sjqtentacles/sml-readline

Gold taken:
- Kill ring: `C-k` kill-to-end, `C-u` kill-back-to-start, `M-d`/`M-Backspace` kill word, `C-w` unix-word-rubout (whitespace boundary), `C-y` yank, `M-y` yank-pop (rotate on repeat). Consecutive kills append to the same ring entry (readline behaviour).
- Movement: `C-a`/`C-e` line ends (already present via \x01/\x05), `C-b`/`C-f` char, word motion.
- Editing: `C-t` transpose-chars, `C-_`/`C-z` undo.
- prompt_toolkit: reverse/forward incremental search, autosuggestions (fish-like), completion while typing, syntax highlighting of input, multiple buffers, no global state.
- sml-readline: the API shape worth copying — pure `step : state -> key -> state * action`, pure `render`, injectable pure completer hook, decode of ANSI sequences. Our model.py already follows this; the sweep extends it rather than breaking it.

What we implement: kill ring (`C-k`, `C-u`, `C-w`, `C-y` with yank-pop on repeat), word motion (`C-b`/`C-f`), transpose (`C-t`), bounded undo (`C-z`), tab completion of `/` commands (prompt_toolkit's completer idea, injected as a pure function over the command table).

## 2. Command palette — Textual + francois

Sources:
- Textual docs: https://github.com/textualize/textual/blob/HEAD/docs/guide/command_palette.md
- antoine-gmnz/francois command-palette doc: https://github.com/antoine-gmnz/francois/blob/HEAD/docs/guide/command-palette.md
- ADH CLI ADR on Textual providers: https://github.com/allenhutchison/adh-cli/blob/HEAD/docs/adr/011-command-palette-integration.md
- Command-palette challenge requirements: http://dev.to/reactchallenges/new-react-challenge-command-palette-24jn

Gold taken:
- Launch on `Ctrl-P`; single input; up/down navigate; Enter runs; Esc closes; focus trapped while open.
- Matching is **ordered-subsequence, case-insensitive** (not Levenshtein): every query char must appear in order. Ranking = position where the first query char was greedily consumed (ascending), ties alphabetical; empty query returns registration order. (francois doc — simple, fast, no scoring tables.)
- Challenge requirements checklist we honour: opens empty, typing filters, no-match shows `No results for "<q>"`, clearing restores full list, first item highlighted on open/query change, arrows wrap, Enter runs + closes, Esc closes.
- Providers pattern (adh-cli): commands come from a registry; ours registers slash commands plus internal state commands (clear, toggles, scroll-to-top/bottom, quit).

## 3. History — atuin

Sources:
- atuin skill docs: https://github.com/faahim/openclaw-skills/blob/HEAD/atuin-history/SKILL.md
- modern toolkit notes: https://github.com/josephsanjaya/skills/blob/HEAD/shell-expert/references/modern-toolkit.md
- atuin reviewed skill: https://github.com/hinvec/security-scanned-skills/blob/HEAD/skills/atuin-shell-history-database-sync/SKILL.md

Gold taken:
- `Ctrl-R` opens an interactive fuzzy search over full history (not just up/down stepping).
- History is metadata-rich (atuin stores exit code/duration/cwd/host/session); we keep the in-memory list but add **persistence to a file**, dedup (already had), and the Ctrl-R interactive picker reusing the palette machinery.
- Search modes (fuzzy/prefix/substring): our picker uses the same ordered-subsequence matcher as the palette, plus substring fallback for history (short strings).

## 4. Markdown rendering — Rich + agentculture/culture + madcatter

Sources:
- thuja PRD-024 (Rich reference analysis): https://github.com/fidelityframework/thuja/blob/HEAD/docs/PRDs/PRD-024-markdown-rendering.md
- agentculture/culture commit (chat panel markdown): https://github.com/agentculture/culture/commit/cc20363f7b980e9126dc0297c75eb7ae2330a478
- madcatter (Rich-based mdcat): https://github.com/rekursiv-ai/madcatter

Gold taken:
- Rich's mapping: headings → styled rules, code blocks → fenced panels w/ syntax highlight, lists nested, blockquotes → left border, tables, `---` → Rule, inline code styled.
- agentculture/culture's exact pattern for chat: header line (`[ts] icon nick:`) + body parsed as CommonMark — and the footgun fix: agent text must not be reinterpreted as markup, so pass renderables not interpolated strings. We apply block-level markdown **only to assistant lines**, never re-parsing user text as markup.
- madcatter: left-justify headings (Rich centers H1 — wrong for chat), strip fences and indent code blocks, keep it readable without true colour.

What we implement (pure, curses-free): block-level markdown for assistant lines — `#` headings (bold kind), fenced code blocks (own kind, fences stripped, indented), `>` quotes (own kind), `-`/`1.` lists (indented), `---` → rule line. Toggleable (`state.markdown`), because plain output containing `#` must not be mangled.

## 5. Elm architecture + composable widgets — charm (bubbletea/bubbles)

Sources:
- mochi tui-primitives skill: https://github.com/xanstomper/mochi/blob/HEAD/skills/tui-primitives/17_charmbracelet_coding_agents.md
- charm-tui skill: https://github.com/danielxxomg/charm-tui-skills-v2/blob/HEAD/building-glamorous-tuis/references/go-tui.md
- bubbletea skill: https://github.com/devdaveframe/.dotfiles/blob/HEAD/claude/.claude/skills/bubbletea/SKILL.md
- super-simple-software-factory charm skill: https://github.com/iksnae/super-simple-software-factory/blob/HEAD/.claude/skills%20copy/charm-tui/SKILL.md

Gold taken:
- Model/Update/View: state in one struct, Update pure `(model, msg) -> (model, cmd)`, View pure `model -> string`. Our model.py/app.py split already is this; sweep keeps it.
- Bubbles widget set as the checklist of what a chat TUI should have: **spinner** (named frame sets: Dot, MiniDot, Line — ASCII-safe variants), viewport (scrollback with offset), progress, list (filterable), help (short/full key help), key (binding definitions).
- Help bar: `key.Binding` + `help.Model` with short vs full help — ours: status bar shows short hints, `?` overlay shows full.
- Spinner ticks only while work is in flight; the app loop already polls at 120ms — we advance frames there, not in render (keeps render deterministic).
- Tick/poll pattern: keep ticking only while busy, then stop.

## 6. Status bar — powerline / vim-airline / tmux

Gold taken (common knowledge, verified pattern): multi-segment bar — left: mode/status + working spinner with elapsed time; right: focus panel, scroll position %, key hints. Ours becomes: `ready  ⠋ working 3.2s   |  input  |  scroll 42%  |  ? help`. Scroll % like `less`.

## 7. Scrollback search — less / tmux copy-mode

Gold taken: `/` opens incremental search, `n`/`N` (or Enter/Up) step through matches, Esc exits, current match highlighted, match counter `3/12` in the bar. Implemented as pure `SearchState` over line indices; the matched row gets its own kind so the driver can highlight it.

## 8. Themes

Gold taken: btop/Textual theme variables; lipgloss named colours. Ours: `TuiApp(theme={kind: curses_color})` override map merged over defaults; new kinds registered in `_KINDS` (`code`, `md_head`, `md_quote`, `search`, `search_match`, `palette`, `blank`). The ninja "vrede peace" electric-blue accent preference from USER.md is honoured via the theme hook, not hardcoded.

## Deliberately NOT taken

- Full Rich/Textual as a dependency: the repo's standing order is zero mandatory deps and a pure testable model; we re-implement the needed pieces in stdlib-only Python.
- Vi modal editing (prompt_toolkit offers it; readline default is emacs — emacs wins for a chat box).
- Mouse support (prompt_toolkit/bubbletea have it; low value in a chat TUI, adds driver complexity).
- Atuin's encrypted sync server: out of scope; file persistence only.
- Inline ANSI/markup parsing of assistant text: the agentculture/culture footgun — we never reinterpret `[bold]`-style markup, only markdown blocks.
