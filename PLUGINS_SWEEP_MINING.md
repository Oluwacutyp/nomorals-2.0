# PLUGINS sweep — external mining report

Module: `nomorals/plugins/` (errors, manifest, loader, registry, wiring).
Method: mine the best implementations outside the repo, per significant
class, then merge the gold into the real classes. No parallel systems.

Sources are summarized in my own words; links below are the exact
result URLs returned by search (verbatim, unmodified).

---

## 1. pluggy — the pytest hook engine (pytest-dev/pluggy)

- https://github.com/pytest-dev/pluggy/
- https://github.com/pytest-dev/pluggy/blob/HEAD/README.rst
- https://github.com/fatpigeorz/agentix/blob/HEAD/docs/plugin-system-research.md
- https://github.com/chernistry/bernstein/blob/HEAD/docs/decisions/007-pluggy-plugin-system.md

**How the best does it.** A `PluginManager` owns the contract. The host
declares *hook specifications* (`@hookspec`) — the canonical signature and
docstring for each hook. Plugins provide *implementations* (`@hookimpl`).
The project name links specs, impls, and manager, so plugins written for a
different project are ignored. Key semantics:

- `firstresult=True` — stop at the first implementation returning non-None
  (override/resolution hooks).
- `tryfirst` / `trylast` markers — deterministic call ordering.
- Hook *wrappers* — code runs around all implementations (tracing, timing).
- **Validation at register time** — an impl whose signature doesn't match
  the spec raises `PluginValidationError` immediately, not at call time.
- Error isolation is a thin `_safe_call` wrapper away.

**What our classes lack.** `loader.py` has *entry points* (one caller, one
callee) but **no hook system at all** — there is no way for N plugins to
extend one host behavior (e.g. "every plugin gets a say on an incoming
message"). No ordering, no firstresult, no registration-time validation of
hook impls. The agentix survey also notes the counter-lesson: don't
over-engineer — pluggy's full marker machinery is heavy; what Devon needs
is the *semantics* (spec, impl, ordering, firstresult, error isolation),
not the decorator framework.

**Gold to merge:** a `PluginHookBus` in `loader.py` — manifest declares
`hooks: {name: entry | {entry, priority}}`; the bus attaches loaded
plugins, validates impls at attach time (callable, accepts `(caps, …)`),
emits in priority order with per-plugin error isolation, and offers
`emit_first()` for resolution-style hooks.

## 2. Home Assistant integration manifests

- https://github.com/laurentlemercier/remora-ng/blob/HEAD/.github/instructions/blueprint.manifest.instructions.md
- https://github.com/aavdberg/ha-petkit/blob/HEAD/.agents/skills/home-assistant-integration-developer/references/manifest-guide.md
- https://github.com/adlaws/ha-oncue-scheduler/blob/HEAD/.agents/skills/ha-integration-developer/SKILL.md

**How the best does it.** `manifest.json` is a rich, validated contract:
`domain`, `name`, `version` (SemVer, enforced), `codeowners`,
`config_flow` (declares UI setup), `documentation`, `issue_tracker`,
`integration_type` (device/hub/service/helper/…), `iot_class`,
`requirements` (PyPI deps installed at setup), `dependencies` (other
integrations that must load first), `after_dependencies` (ordering only).
Validation is schema-driven (hassfest); version must be bumped on every
release.

**What our class lacks.** `manifest.py` knows `name`, `version`,
`description`, `author`, `entry_points`, `permissions`. Missing: engine
compatibility (`requires_devon` — cf. WP "Requires at least", VS Code
`engines`), plugin-to-plugin `dependencies` with version constraints,
validated `config_schema` + defaults (HA's config_flow), `requirements`
(pip deps), discoverability metadata (`tags`, `homepage`, `license`,
`icon`, `display_name`).

**Gold to merge:** extend `PluginManifest` with `display_name`,
`requires_devon` (parsed version spec, checked at install), `dependencies`
(`"name>=1.2"` strings, resolved against the registry at install),
`requirements` (declared; install warns when pip deps are listed since
Devon plugins run in-process), `config_schema` + `default_config`
(validated at manifest load), `tags`, `license`, `homepage`, `icon`, and
forward-compatible `extra` passthrough for unknown keys (VS Code ignores
unknowns; we stash them instead of dropping them).

## 3. VS Code extension manifests — declarations, not execution

- https://dev.to/karrade7/vs-code-extensions-basic-concepts-architecture-b17
- https://github.com/xsyetopz/skills/blob/HEAD/skills/vscode-extension-development/references/hosts-trust-and-packaging.md
- https://github.com/juliusolsson05/agent-code/blob/HEAD/docs/superpowers/plans/2026-07-20-extension-platform.md

**How the best does it.** `package.json` has three jobs: identity
(name/publisher/version/icon/license/repository/bugs/homepage/keywords),
`engines` (minimum host API version), and a **`contributes` block** —
static declarations of commands, configuration schemas, views, menus.
Plus **`activationEvents`** (`onCommand`, `onLanguage`,
`onStartupFinished`) so the host loads code lazily, only when needed.
The decisive insight (from the agent-code extension-platform plan): the
host must know what an extension adds **before the extension runs** —
otherwise every extension loads at startup just to populate a command
list, and a broken extension silently vanishes instead of showing greyed
out with a reason.

**What our classes lack.** Nothing in the manifest declares what a plugin
*adds* to Devon — the host must load and run code to discover anything.
No activation/laziness story, no engine floor, no static command/tool
declarations.

**Gold to merge:** a `contributes` block in the manifest:
`commands: [{name, title, description}]` (surfaced by `nm plugin list`
/ future command palette without loading code) and
`tools: [{name, description, schema, entry}]` — plugins contributing
**agent tools with JSON schemas**, the MCP-equivalent surface for an
agent host. `InstalledPlugin.to_dict()` exposes the contribution summary
so the host can render "what this plugin adds" without importing it.

## 4. WordPress — headers + lifecycle discipline

- https://github.com/eslam-dev/ai-dev-kit/blob/HEAD/source/skills/build-wp-plugin/SKILL.md
- https://github.com/jgreys/claude-wordpress-skills/blob/HEAD/skills/wp-plugin-development/SKILL.md
- https://developer--wordpress--org.proxy.hfzk.net.cn/plugins/plugin-basics/activation-deactivation-hooks/

**How the best does it.** Header fields (`Plugin Name`, `Version`,
`Requires at least`, `Requires PHP`) are enforced. Lifecycle is split
three ways: `register_activation_hook` (schema, defaults — runs on
install/enable), `register_deactivation_hook` (unschedule cron, clear
*temp* data — **never delete user data**), and `uninstall.php` (full,
permanent cleanup, guarded). Deactivation ≠ uninstall is a hard rule.

**What our classes lack.** `registry.py` has install/enable/disable/remove
but **zero lifecycle hooks** — a plugin cannot seed defaults on install,
flush caches on disable, or clean up on remove. No deactivation-vs-
uninstall data distinction.

**Gold to merge:** manifest `lifecycle: {on_install, on_enable,
on_disable, on_uninstall, on_upgrade}` entry points (also accepted as
plain `entry_points` keys for back-compat). Registry calls them around
state changes: install/upgrade failures in lifecycle hooks fail the
operation (WordPress activation semantics); enable/disable/uninstall hook
failures are captured, emitted as events, and swallowed (a broken
`on_disable` must not trap a plugin in the enabled state). Manifest flag
`purge_data_on_remove` (default false) — when true, `remove()` also
deletes the plugin's `plugin_kv` rows: the deactivation-keeps-data /
uninstall-purges-data distinction, enforced in code.

## 5. Sandboxing untrusted code — defense in depth

- https://github.com/portofcontext/pctx-py-sandbox
- https://github.com/aquillm/aquillm/blob/HEAD/docs/roadmap/plans/pending/2025-03-16-sandboxed-math-integration.md
- https://github.com/dkarthi1973/ai-software-factory/blob/HEAD/SECURITY.md
- https://dev.to/yarkhan02/running-untrusted-code-safely-with-aws-lambda-and-keeping-my-vps-out-of-it-2ehe

**How the best does it.** Nobody trusts in-process Python. The standard
ladder: hard wall-clock **timeouts**, **output size caps**, **minimal
environment** (no ambient secrets/credentials leak into the sandbox),
**resource limits** (`RLIMIT_CPU`/`RLIMIT_AS`/`RLIMIT_NPROC`/`RLIMIT_FSIZE`,
process-group kill on timeout), and an **audit trail** of everything the
sandbox did (the ai-software-factory uses a hash-chained audit log so
"why did this code ship" stays answerable).

**What our classes lack.** `loader.py`'s docstring is honest that the
sandbox is cooperative — but it stops there. No timeout on entry-point
calls (a plugin can hang `nm plugin run` forever), no audit of capability
use (the docstring *promises* "every capability use explicit and
auditable" yet nothing records it), no per-run budgets (a plugin can
`fetch`/`chat` in an infinite loop; only a 10 MB per-fetch cap exists).

**Gold to merge (honest, in-process, no new deps):**
- `call_entry(loaded, key, caps, timeout=…)` — runs the entry in a worker
  thread; on expiry raises `PluginTimeout`. Documented as
  cooperative-abandon (the thread can't be killed; the *caller* gets
  control back). Manifest `limits.entry_timeout_s` default.
- **Capability audit**: `PluginCapabilities` accepts an `audit` sink;
  every capability method logs `(plugin, permission, action, detail)`.
  `wiring.py` wires it to the `plugin.capability_used` event on the
  global bus — fail-open like existing telemetry. The "auditable" promise
  becomes real.
- **Per-run budgets**: manifest `limits: {fetch_calls, chat_calls,
  fetch_bytes}` enforced per capabilities instance. A runaway plugin dies
  with `PermissionDenied`-family clarity, not a hung process.

## 6. MCP — capability declaration & negotiation

- https://github.com/alexrdclement/mcp-kotlin-sdk
- https://github.com/gannanasr/meridian-port-authority
- https://github.com/api-evangelist/mcp

**How the best does it.** Servers declare capabilities up front (`tools`,
`resources`, `prompts`, `logging`, `completions`); clients declare
theirs (`sampling`, `roots`, `elicitation`). SDKs **enforce at runtime**:
the sender checks the receiver declared the capability before calling;
tools carry typed JSON Schema input with `required` fields and
`additionalProperties: false`; `list_changed` notifications keep both
sides in sync. Servers advertise *only implemented* capabilities.

**What our classes lack.** Capability *granting* exists
(`PluginCapabilities`), but there is no negotiation/introspection: a
plugin can't ask "what can this host do for me", and the host can't ask
"what do you offer" without running code (fixed by the VS Code
`contributes` gold above). Tool schemas aren't validated. A plugin also
has **no way to talk back to the user** — no notify surface.

**Gold to merge:** `contributes.tools` entries carry JSON `schema`
(plus a tiny dependency-free validator reused from `config_schema`
work); `PluginCapabilities.notify()` behind a new `notify.send`
permission, wired to a `plugin.notify` bus event so owner surfaces
(Telegram etc.) can deliver it — the missing "talk back to the user"
channel. `registry.health()` gives the host a no-side-effects
"can this plugin load, are its entry points resolvable, deps met"
check — the runtime `assertCapabilityForMethod` equivalent.

## 7. Cross-cutting: what the current code gets wrong

- **Real bug (wiring.py):** `_make_chatter`'s inner `_chatter` calls
  `brain_for(self.context)` — `self` is undefined in that closure, so
  *every* `llm.chat` capability call raises `NameError`. The llm.chat
  permission is dead on arrival. Fix: use the closure's `context` and
  actually call through the lazily built chain.
- **Registry gaps:** no dependency resolution, no engine-compat check, no
  `upgrade` path (installing 1.1.0 over 1.0.0 is an `AlreadyInstalled`
  dance), no `search`, no health check, no events for load/run.
- **Style gap:** `nm plugin list` prints flat `[on ] name version` lines.
  Nothing in the module helps a CLI render god-tier output. Add an
  in-module `render.py` with output themes (`unicode` default, `plain`,
  `compact`, `markdown`) for plugin cards and tables — the CLI adopts it
  later without this module changing shape.

## 8. Feature list (spec floor — everything below ships)

**manifest.py**
- New fields: `display_name`, `requires_devon`, `dependencies`,
  `requirements`, `config_schema` + `default_config` (validated at load),
  `contributes` {commands, tools}, `hooks` {name: entry|{entry,priority}},
  `lifecycle` {on_install,on_enable,on_disable,on_uninstall,on_upgrade},
  `limits` {entry_timeout_s,fetch_bytes,fetch_calls,chat_calls},
  `tags`, `license`, `homepage`, `icon`, `purge_data_on_remove`, `extra`.
- Version-spec parser + `satisfies(version, spec)` (`>=`,`<=`,`>`,`<`,
  `==`,`~=`,`!=`, comma-AND). Unknown top-level keys → `extra`, not
  rejection (forward-compat).

**errors.py** — `PluginTimeout`, `DependencyError`, `IncompatibleEngine`.

**loader.py**
- `PluginHookBus`: `attach(loaded, caps)`, `detach(name)`,
  `emit(hook, *a, **k)` (priority order, error-isolated, returns
  `HookResults(results, errors)`), `emit_first` (pluggy firstresult),
  attach-time impl validation → `LoadError`.
- `call_entry(loaded, key, caps, timeout=…)` → `PluginTimeout` on expiry.
- `PluginCapabilities`: audit sink (every capability method logs),
  per-run rate limits from manifest, new `notify()` (`notify.send`),
  new `config()` (schema-validated plugin settings backed by kv).
- `PluginConfig` class: schema-validated get/set/reset with defaults.

**registry.py**
- Install: engine-compat check (`IncompatibleEngine`), dependency
  resolution (`DependencyError` naming the missing), `on_install`
  lifecycle (fail-fast), `requirements` warning event.
- `upgrade(source)`: newer-version install + `on_upgrade`, kv carries
  over (per-name namespacing).
- `enable`/`disable`/`remove`: lifecycle hooks, fail-open with events.
- `remove`: `purge_data_on_remove` → clears `plugin_kv` rows.
- `search(query)`, `health(name)` (load check, no side effects).
- New events: `plugin.upgraded`, `plugin.lifecycle_failed`,
  `plugin.capability_used` (via wiring), `plugin.hook_error`.

**wiring.py**
- Fix the `self.context` NameError (llm.chat works for real).
- Wire `notify` → `plugin.notify` bus event; wire audit sink; pass
  manifest limits into capabilities; add `purge_plugin_data(db, name)`.

**render.py (new, in-module)**
- `render_plugin_card(plugin, theme)`, `render_plugin_table(plugins,
  theme)`, themes: `unicode` (default, rich box-drawing + status glyphs),
  `plain`, `compact`, `markdown`. Contribution summaries, permission
  chips, health line.

**Tests** — `tests/test_plugins_sweep.py`: manifest new-field
validation, version-spec satisfaction, hook bus ordering/isolation/
firstresult, call_entry timeout, audit capture, rate limits, notify,
config schema validation, dependency/engine checks, lifecycle hooks,
upgrade, search, health, purge-on-remove, render themes.
