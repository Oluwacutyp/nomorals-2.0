"""God-tier terminal rendering for the plugin system.

The plugin package owns how plugins *look* when presented — cards,
tables, and health reports — so every surface (``nm plugin list``,
chat, docs) renders the same way. Four themes:

* ``unicode`` (default) — box-drawing, status glyphs, permission chips.
* ``plain`` — 7-bit ASCII, for dumb terminals and logs.
* ``compact`` — one line per plugin, for dense overviews.
* ``markdown`` — for chat messages and generated docs.

All renderers take :class:`InstalledPlugin` records (or the plain dicts
from :meth:`PluginRegistry.health`) and return strings; they never touch
the database or import plugin code.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "THEMES",
    "render_plugin_card",
    "render_plugin_table",
    "render_health",
]

THEMES = ("unicode", "plain", "compact", "markdown")


def _theme(name: str) -> str:
    if name not in THEMES:
        raise ValueError(f"unknown render theme {name!r}; want {THEMES}")
    return name


class _Glyphs:
    """Per-theme symbol sets."""

    def __init__(self, theme: str) -> None:
        if theme == "plain":
            self.on = "[on] "
            self.off = "[off]"
            self.ok = "ok"
            self.bad = "FAIL"
            self.warn = "!"
            self.bullet = "- "
            self.sep = " | "
            self.rule = "-" * 56
        elif theme == "markdown":
            self.on = "**enabled**"
            self.off = "*disabled*"
            self.ok = "✅"
            self.bad = "❌"
            self.warn = "⚠️"
            self.bullet = "- "
            self.sep = " · "
            self.rule = "---"
        else:  # unicode + compact share glyphs
            self.on = "●"
            self.off = "○"
            self.ok = "✓"
            self.bad = "✗"
            self.warn = "!"
            self.bullet = "· "
            self.sep = " · "
            self.rule = "─" * 56


def _contrib_bits(plugin: Any) -> list[str]:
    summary = plugin.manifest.contribution_summary()
    bits: list[str] = []
    if summary["commands"]:
        bits.append(f"{len(summary['commands'])} cmd")
    if summary["tools"]:
        bits.append(f"{len(summary['tools'])} tool")
    if summary["hooks"]:
        bits.append(f"{len(summary['hooks'])} hook")
    return bits


def _perm_chips(manifest: Any, g: _Glyphs) -> str:
    perms = sorted(manifest.permissions)
    if not perms:
        return "no permissions"
    return g.sep.join(perms)


def render_plugin_card(plugin: Any, theme: str = "unicode") -> str:
    """A rich multi-line card for one installed plugin."""
    theme = _theme(theme)
    g = _Glyphs(theme)
    m = plugin.manifest
    lines: list[str] = []
    status = g.on if plugin.enabled else g.off

    if theme == "markdown":
        lines.append(f"## {m.display_name} `{plugin.version}` {status}")
        if m.description:
            lines.append(f"*{m.description}*")
        meta = []
        if m.author:
            meta.append(f"by {m.author}")
        if m.license:
            meta.append(m.license)
        if m.homepage:
            meta.append(f"[homepage]({m.homepage})")
        if m.tags:
            meta.append("tags: " + ", ".join(m.tags))
        if meta:
            lines.append(g.sep.join(meta))
        lines.append(f"**Permissions:** {_perm_chips(m, g)}")
        bits = _contrib_bits(plugin)
        if bits:
            detail = []
            s = m.contribution_summary()
            if s["commands"]:
                detail.append("commands: " + ", ".join(s["commands"]))
            if s["tools"]:
                detail.append("tools: " + ", ".join(s["tools"]))
            if s["hooks"]:
                detail.append("hooks: " + ", ".join(s["hooks"]))
            lines.append(f"**Adds:** {g.sep.join(detail)}")
        if m.lifecycle:
            lines.append("**Lifecycle:** " + ", ".join(sorted(m.lifecycle)))
        deps = [f"{d.name}{d.spec}" for d in m.dependencies]
        if deps:
            lines.append("**Depends on:** " + ", ".join(deps))
        return "\n\n".join(lines)

    if theme == "compact":
        bits = _contrib_bits(plugin)
        extra = f"  [{g.sep.join(bits)}]" if bits else ""
        desc = f" — {m.description[:48]}" if m.description else ""
        return (f"{status} {plugin.name} {plugin.version}{desc}{extra}")

    # unicode + plain: ruled card
    lines.append(g.rule)
    lines.append(f"{status} {plugin.name}  {plugin.version}")
    if m.display_name and m.display_name != plugin.name:
        lines.append(f"  {m.display_name}")
    if m.description:
        lines.append(f"  {m.description}")
    meta = []
    if m.author:
        meta.append(f"by {m.author}")
    if m.license:
        meta.append(m.license)
    if m.homepage:
        meta.append(m.homepage)
    if m.tags:
        meta.append("tags: " + ", ".join(m.tags))
    if meta:
        lines.append(f"  {g.sep.join(meta)}")
    lines.append(f"  {g.bullet}permissions: {_perm_chips(m, g)}")
    s = m.contribution_summary()
    adds: list[str] = []
    if s["commands"]:
        adds.append("commands: " + ", ".join(s["commands"]))
    if s["tools"]:
        adds.append("tools: " + ", ".join(s["tools"]))
    if s["hooks"]:
        adds.append("hooks: " + ", ".join(s["hooks"]))
    if adds:
        lines.append(f"  {g.bullet}adds: {g.sep.join(adds)}")
    if m.lifecycle:
        lines.append(f"  {g.bullet}lifecycle: "
                     + ", ".join(sorted(m.lifecycle)))
    deps = [f"{d.name}{d.spec}" for d in m.dependencies]
    if deps:
        lines.append(f"  {g.bullet}depends: " + ", ".join(deps))
    if m.requires_devon:
        lines.append(f"  {g.bullet}requires Devon {m.requires_devon}")
    lines.append(g.rule)
    return "\n".join(lines)


def render_plugin_table(plugins: list[Any], theme: str = "unicode") -> str:
    """Aligned multi-plugin overview table."""
    theme = _theme(theme)
    g = _Glyphs(theme)
    if not plugins:
        return "no plugins installed."

    if theme == "markdown":
        lines = ["|  | Name | Version | Adds | Tags |",
                 "|---|---|---|---|---|"]
        for p in plugins:
            status = "✅" if p.enabled else "⬜"
            bits = ", ".join(_contrib_bits(p)) or "—"
            tags = ", ".join(p.manifest.tags) or "—"
            lines.append(f"| {status} | `{p.name}` | {p.version} | "
                         f"{bits} | {tags} |")
        return "\n".join(lines)

    if theme == "compact":
        return "\n".join(render_plugin_card(p, "compact") for p in plugins)

    rows = []
    for p in plugins:
        status = g.on if p.enabled else g.off
        bits = g.sep.join(_contrib_bits(p)) or "—"
        tags = ", ".join(p.manifest.tags) or "—"
        rows.append((status, p.name, p.version,
                     p.manifest.display_name[:28], bits, tags))
    widths = [max(len(r[i]) for r in rows) for i in range(6)]
    out: list[str] = []
    for status, name, version, display, bits, tags in rows:
        out.append(
            f"{status}  {name:<{widths[1]}}  {version:<{widths[2]}}  "
            f"{display:<{widths[3]}}  {bits:<{widths[4]}}  {tags}")
    return "\n".join(out)


def render_health(report: dict[str, Any], theme: str = "unicode") -> str:
    """Render a :meth:`PluginRegistry.health` report."""
    theme = _theme(theme)
    g = _Glyphs(theme)
    lines: list[str] = []
    name, version = report["name"], report["version"]
    status = g.on if report.get("enabled") else g.off

    if theme == "markdown":
        lines.append(f"## Health: `{name}` {version} {status}")
    else:
        lines.append(g.rule)
        lines.append(f"{status} health: {name} {version}")

    def _line(ok: bool | None, text: str) -> None:
        mark = g.ok if ok else (g.bad if ok is False else g.warn)
        if theme == "markdown":
            lines.append(f"- {mark} {text}")
        else:
            lines.append(f"  {mark} {text}")

    if report.get("load_error"):
        _line(False, f"load: {report['load_error']}")
    elif report.get("loadable"):
        _line(True, "loads cleanly")
    for key, info in report.get("entry_points", {}).items():
        if info["ok"]:
            _line(True, f"entry `{key}` → {info['target']}")
        else:
            _line(False, f"entry `{key}` broken: {info.get('error')}")
    for key, info in report.get("lifecycle", {}).items():
        if info["ok"]:
            _line(True, f"lifecycle `{key}` → {info['target']}")
        else:
            _line(False, f"lifecycle `{key}` broken: {info.get('error')}")
    for hook, err in report.get("hook_errors", {}).items():
        _line(False, f"hook `{hook}` broken: {err}")
    for dep in report.get("dependencies", []):
        have = ", ".join(dep["installed"]) or "not installed"
        _line(dep["satisfied"],
              f"dep {dep['name']}{dep['spec']} (have: {have})")
    eng = report.get("engine", {})
    _line(eng.get("ok"), f"engine: requires {eng.get('requires')}, "
                         f"running {eng.get('running')}")
    if theme != "markdown":
        lines.append(g.rule)
    return "\n".join(lines)
