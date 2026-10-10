# CONSOLE SWEEP — External Mining Report

**Module:** `nomorals/console/` (10 files) · **Date:** 2026-10-10
**Rule:** stdlib-only (owner's standing order: zero mandatory deps, Termux-safe),
no black backgrounds, no red text — in every theme.

For every significant class, the best implementations OUTSIDE the repo were
mined, then the "how should it LOOK/FEEL?" and "what's missing?" questions
were answered per class.

---

## 1. TUI architecture — how the best do full-screen apps

| Source | Gold mined |
|---|---|
| **Textual** (textualize) | Reactive attributes (`reactive`/`watch_`/`compute_`): state that affects rendering triggers refresh automatically; `var` for state that doesn't. Layout discipline: dock top/bottom bars, `1fr` fill regions, scrollable child pane. Widget review checklist: widgets focused and single-purpose, keyboard shortcuts for everything. |
| **Rich** (textualize, 40k★) | `Panel` (titled bordered boxes), `Table` (rounded box styles), `Live` (atomic in-place refresh), `console.status()` spinners, `Progress` with `SpinnerColumn + TextColumn + BarColumn`, `Rule` with title, `Columns` layout, `Bar` widget. Quick-ref map: Styled output→`rich.console`, Tables→`rich.table`, Trees→`rich.tree`, Progress→`rich.progress`, Spinners→`console.status`, Panels→`rich.panel`, Live→`rich.live`. |
| **prompt_toolkit** (IPython/AWS CLI/cmd2) | `PromptSession`: completion menus, persistent `FileHistory`, Ctrl-R history search, bracketed paste, emacs/vi bindings, autosuggest-from-history. cmd2 migrated from readline to prompt_toolkit purely for these UX wins. Lesson: the input line is a PRODUCT, not a `input()` call. |
| **btop/htop** | Widget-grid dashboard (CPU/mem/net boxes, not a wall of numbers). Per-core live graphs. htop's permanent **function-key bar** at the bottom = discoverability without docs. Sort toggling (P=CPU, M=mem), 1/8-cell sub-precision horizontal bars (`█…▉▊▋▌▍▎▏`). |
| **k9s/lazydocker/lazygit** | Colon-command mode (`:resource`) with auto-completion; `/` filter current view; number keys for tabs/favorites (lazygit 1–5); contextual shortcut hints in the top-right; lazydocker's five live panes where selection drives the detail pane. |
| **tqdm** | Display-rate limiting (avoid slowdown from excessive updates); EMA-smoothed rate for stable ETA; unit scaling (1.2k, 3.4M); `bar_format` templating; postfix stats; unknown-total mode. |
| **lnav** | Semantic log understanding: level-aware coloring, identifier highlighting (IP/PID), regex search, filters, histograms of messages over time, "jump to next error" hotkeys, Gantt view of operations, time-offset display. |
| **chartli / terminal-charts** | Chart idioms: `▁▂▃▄▅▆▇█` sparklines, braille line charts (each cell = 2×4 dots, line-joined consecutive points, interpolated to width, 15% headroom + min/max labels), vertical `columns` charts, `heatmap`. |
| **pyfiglet / ascii-art** | 500+ figlet fonts (slant, shadow, doom, mini); banner styles: block / slant / mini / pixel (`░▒▓█` gradient) / framed vintage. |
| **TUI design-system research** | Panels keep fixed positions (spatial memory). Proportional splits + min/max, not hardcoded layout. Minimum size (80×24); never crash on resize; handle SIGWINCH. Design for the smallest screen first, scale up. Terminal = grid of identically sized cells; emphasis via bold + color, never size. |
| **inspect-rs / Catppuccin / Tokyo Night / Nord** | 21-token **semantic contract** (not palette): backgrounds, borders, text ramp, accent, functional signals. Classic palettes: Tokyo Night `#7aa2f7/#bb9af7/#7dcfff/#9ece6a`, Catppuccin Mocha `#cba6f7/#89b4fa/#a6e3a1`, Nord arctic blues. Theme = fixed role→accent mapping; users pick a theme then override groups. |

### Per-class comparison: "How does the best X do it?"

**palette.py vs Rich/inspect-rs:** Best use semantic ROLES (type/field/key/string/number/bool/punctuation), not raw codes; themes fill a token contract. Ours has raw constants only.
→ ADD: truecolor hex→ANSI helper, text attributes (italic/underline/strike), box-drawing glyph sets, semantic-role layer.

**themes.py vs CC-GUI/Catppuccin suite:** Best ship 6–19 named themes on one semantic contract, with previews and per-group overrides. Ours has 3 themes × 11 roles.
→ ADD: tokyo-night, nord, dracula, catppuccin (all mapped onto the no-red/no-black rule: error→magenta/pink family), extended role set (border, muted, hl, sel, link, chart), `theme_preview()` swatch cards, `describe_theme()`.

**banner.py vs pyfiglet/corporate-launcher:** Best ship 6 styles (block/slant/mini/pixel/vintage/tech) and gradient tinting. Ours has 1 block font.
→ ADD: slant, mini, pixel styles (hand-tuned stdlib fonts for "DEVON"), optional gradient logo, centered/framed variants, `list_banner_styles()`.

**avatar.py vs ascii-art braille/pixel renderers:** Best render at multiple densities (braille 2-line micro, pixel shaded). Ours has 2 sizes, fixed art.
→ ADD: braille micro-avatar, pixel-shaded avatar, wide cinematic avatar, framed presentation, `render_avatar(style=)`. KEEP the user's chosen mini/full art untouched ("do not redesign").

**commands.py vs cmd2/prompt-toolkit:** Best REPLs have history, completion, per-command help, command modes, async-safe output. Ours has 8 commands, flat help.
→ ADD: `banner`, `palette` (theme swatch card), `errors` (DebugHub error tail), `slow`, `llm`, `log <level>` (lnav-lite tail), `uptime`, `export <file>`, `history`, `alias` map, per-command `help <cmd>`, `complete(prefix)` for tab-completion, in-memory command history + `history` command.

**dashboard.py vs btop/Rich:** Best show gauges not numbers (1/8-cell precision meters), `┤ label ├` panel captions, responsive width, rich tables with rounded boxes, per-adapter drill-down. Ours is a fixed 58-wide list.
→ ADD: `gauge()` meter, `table()` rounded-box renderer, `panel()` titled box, `render_adapters_view`, `render_health_view` (btop-style), `render_top_view` (activity top), responsive `width` params, min/max axis on sparklines, panel captions.

**debug.py vs lnav:** Best understand logs semantically: histograms over time, jump-to-error, search/filter, exception trace capture, rate metrics.
→ ADD: `histogram()` (per-minute level counts), `search(pattern)` regex over buffer, `exceptions(n)` traceback capture, `log_rate()` msgs/min, `errors_since(ts)`, level filter in `recent()`.

**widgets.py vs tqdm/chartli/Rich:** Best progress bars smooth the rate (EMA), scale units, adapt width, and never update faster than ~10Hz. Best charts: braille lines, columns, heatmaps. Ours: basic bar, basic sparkline.
→ UPGRADE `ProgressBar`: EMA-smoothed rate, unit scaling, adaptive width, postfix, `bar_format` template, 1/8-cell sub-precision fill, quiet/update-throttle.
→ ADD: `gauge()`, `columns_chart()`, `braille_chart()`, `heatmap()`, `table()`, `panel()`, `rule()` with caption, `Spinner`/`status()` context manager.
→ UPGRADE `sparkline`: min/max labels, color-by-value, fixed scale.
→ UPGRADE `GodScreen`: k9s-style `:` command mode, `/` feed filter, `?` help overlay, `t` theme cycle, `+`/`-` interval, feed scroll (j/k), per-view counts in tabs.
→ UPGRADE `MessageFeed`: `search()`, `by_platform()`, level filter.
→ UPGRADE `GodConsole` input: history (↑/↓), tab completion with popup, emacs bindings (Ctrl-A/E/K/U/W), Ctrl-C clears line, bracketed-paste multiline support, input syntax highlight (command vs args).

**godconsole.py vs prompt_toolkit/simorgh:** Best REPLs keep one live buffer, async-safe output above the prompt, history file, completion menus, no lost sessions on Ctrl-C.
→ See widgets/godconsole upgrades above; use `truncate_visible` everywhere (kill the naive truncation).

---

## 2. What each class SHOULD have (feature gaps → now filled)

- **palette**: hex colors, attributes, semantic roles, box glyphs — MISSING → added.
- **themes**: more themes, more roles, previews, validation — MISSING → added.
- **banner**: styles beyond block, gradient, framing — MISSING → added.
- **avatar**: density variants, framing — MISSING → added (art preserved).
- **commands**: history, completion API, debug commands (errors/slow/llm/log), export — MISSING → added.
- **dashboard**: gauges, tables, panels, drill-down views, responsive width — MISSING → added.
- **debug**: histogram, search, exceptions, rate — MISSING → added.
- **widgets**: tqdm-grade progress, braille/columns/heatmap charts, table/panel/rule primitives, spinner, GodScreen command mode + help + filter — MISSING → added.
- **godconsole**: readline-grade input editing — MISSING → added.

## 3. Style/feel target

btop's alive-grid + htop's permanent key bar + lazydocker's selection-driven
detail + lnav's semantic log color + Tokyo Night/Catppuccin-grade palettes —
all in stdlib ANSI, Termux-safe, no red, no black backgrounds. Watch mode
gets k9s colon-commands and `?` discoverability; the god console gets a real
input line (history, completion, emacs keys) instead of raw `read(1)` chars.
