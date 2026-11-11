"""Devon's web presence (group build §12/§14 close-out): her own
page, the per-group rules pages, and the two WordPress sites.

The web is ANOTHER RUNG — it replaces nothing.  The same content
lives in chat (/help, /rules); the pages exist so a member can
open the full rules in a browser and so Devon has a front door.

Three surfaces, one generator:

- ``GET /devon`` — her page: who she is, what she does, how to
  add her.  The capability catalog is GENERATED from the live
  command registry (LIST_GROUPS × LIST_ONELINERS) at render
  time — a command Devon gains appears, one she loses
  disappears; nothing is hand-copied, so the page cannot go
  stale against the registry it describes.
- ``GET /rules`` + ``GET /rules/<slug>`` — one page per CLAIMED
  group (§33), rendered from the same settings ``rules_text``
  publishes in chat.  Slugs are stable and carry NO internal
  numeric ids: title slug + a short hash of the group key, or
  ``group-<hash>`` when no title is known.  Unknown slug → the
  resolver returns None and the server answers 404.  Rules are
  member-facing by design, so these paths are public like the
  API's liveness probe; they contain rules and titles only.
- WordPress — when a site is configured (web settings:
  url + user + application password), the same bundles publish
  to it through the WP REST API (find-or-create by slug).  Two
  sites: one carries her page, one carries the rules.  When a
  site is NOT configured, publishing reports exactly that —
  the local pages still generate and serve; nothing pretends
  a publish happened.

The in-group rules card links here: the runtime appends the
rules-page URL (a native URL button on Telegram's bot surface,
a bare link where buttons don't exist) whenever a public base
URL is configured.
"""
from __future__ import annotations

import base64
import hashlib
import html
import json
import re
import urllib.request
from pathlib import Path
from typing import Any, Callable

#: The stamp every generated page carries (her front door signs
#: its work the way her group surfaces do).
PAGE_FOOTER = "Powered by peacethefirst"


def _esc(value: Any) -> str:
    return html.escape(str(value if value is not None else ""))


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(text or "").lower()).strip("-")
    return slug[:48].strip("-")


def _key_hash(group_key: str, n: int = 6) -> str:
    return hashlib.sha256(
        str(group_key).encode("utf-8")).hexdigest()[:n]


def group_slug(group_key: str, title: str = "") -> str:
    """The public slug for one group's rules page.  Stable for
    a given key (and title), unique per key via the hash
    suffix, and free of the platform's internal numeric ids —
    a Telegram chat id never appears in a URL."""
    base = slugify(title)
    if base:
        return f"{base}-{_key_hash(group_key)}"
    return f"group-{_key_hash(group_key, 10)}"


def rules_url(base_url: str, slug: str) -> str:
    return f"{str(base_url or '').rstrip('/')}/rules/{slug}"


# ── the Devon page ──────────────────────────────────────────────────

def devon_page_html(*, bot_username: str = "",
                    groups: list[tuple[str, list[str]]] | None = None,
                    oneliners: dict[str, str] | None = None) -> str:
    """Her page, generated from the live command registry.

    ``groups``/``oneliners`` default to the registry itself
    (LIST_GROUPS × LIST_ONELINERS, read at call time); tests
    inject a subset to prove the page tracks the registry —
    a command present renders, a removed one is gone."""
    if groups is None or oneliners is None:
        from .chat.control import LIST_GROUPS, LIST_ONELINERS

        groups = groups if groups is not None else LIST_GROUPS
        oneliners = oneliners if oneliners is not None \
            else LIST_ONELINERS
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,"
        "initial-scale=1'>"
        "<title>Devon — a personal AI OS</title></head><body>",
        "<h1>🥷🏼 Devon</h1>",
        "<p>Devon is a personal AI operating system that lives "
        "in her owner's chats: she runs groups and channels, "
        "plays and referees games, keeps a real group economy, "
        "watches for scams, translates the news, and does the "
        "work end to end — not a chatbot waiting for commands, "
        "a system that acts.</p>",
    ]
    if bot_username:
        user = _esc(str(bot_username).lstrip("@"))
        parts.append(
            f"<p>Add her on Telegram: "
            f"<a href='https://t.me/{user}'>@{user}</a> — or "
            "invite her to your group and make her an admin; "
            "she claims the room and her systems switch on.</p>")
    else:
        parts.append(
            "<p>Invite her to your Telegram group and make her "
            "an admin — she claims the room and her systems "
            "switch on. Her owner pairs her other accounts "
            "(WhatsApp, more Telegram) from her DM.</p>")
    parts.append("<h2>What she can do</h2>")
    parts.append(
        "<p>Generated from her live command registry — this "
        "list is exactly as current as the build serving it.</p>")
    for title, kinds in groups:
        rows = []
        for kind in kinds:
            one = str(oneliners.get(kind) or "").strip()
            if not one:
                continue
            rows.append(
                f"<li><code>/{_esc(kind)}</code> — {_esc(one)}</li>")
        if rows:
            parts.append(f"<h3>{_esc(title)}</h3><ul>"
                         + "".join(rows) + "</ul>")
    parts.append(
        "<h2>Group rules pages</h2>"
        "<p>Every group Devon runs publishes its rules as a "
        "page: <a href='/rules'>/rules</a> lists them.</p>")
    parts.append(f"<p><i>{_esc(PAGE_FOOTER)}</i></p></body></html>")
    return "".join(parts)


# ── the rules pages ─────────────────────────────────────────────────

def claimed_group_keys(surface: Any) -> list[str]:
    """Groups whose rules get a page: the CLAIMED rooms (§33 —
    designated by the owner or run by her bot).  A foreign room
    she merely sits in publishes nothing.

    The enumeration is a consistent cut: it walks a snapshot
    of the store's keys taken as the pass begins, so a room
    registered mid-publish (her live surfaces write the same
    store) never breaks the pass and never half-joins it —
    it publishes whole on the next pass instead."""
    return sorted(k for k in list(surface.store.data)
                  if surface.system_output_allowed(k))


def rules_slug_map(surface: Any, *,
                   titles: dict[str, str] | None = None
                   ) -> dict[str, str]:
    """slug → group_key for every claimed group.  Titles come
    from the caller's knowledge (the chats ledger); without one
    the group still gets a stable hash slug."""
    titles = titles or {}
    out: dict[str, str] = {}
    for key in claimed_group_keys(surface):
        out[group_slug(key, titles.get(key, ""))] = key
    return out


def _rules_of(surface: Any, group_key: str) -> tuple[list[str], dict]:
    settings = surface.store.settings(group_key)
    return list(settings.get("rules") or []), settings


def rules_page_html(surface: Any, group_key: str, *,
                    title: str = "") -> str:
    """One group's rules as a real page — the same rules the
    enforcer reads, rendered for a browser."""
    rules, settings = _rules_of(surface, group_key)
    name = title or "Group rules"
    mod = settings.get("moderation") or {}
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,"
        "initial-scale=1'>"
        f"<title>{_esc(name)} — rules</title></head><body>",
        f"<h1>📜 {_esc(name)}</h1>",
    ]
    if rules:
        parts.append("<ol>" + "".join(
            f"<li>{_esc(rule)}</li>" for rule in rules) + "</ol>")
    else:
        parts.append(
            "<p>No written rules yet — be decent, no spam, "
            "no flooding.</p>")
    parts.append(
        f"<p>Moderation: flood limit "
        f"{_esc(mod.get('flood_max', ''))} messages per "
        f"{_esc(int(mod.get('flood_window_s') or 0))}s · repeats "
        f"×{_esc(mod.get('repeat_threshold', ''))} are removed.</p>")
    if mod.get("link_allowlist"):
        parts.append(
            "<p>Links: only allowlisted domains are permitted.</p>")
    parts.append(
        "<p>These are the live rules — the same settings the "
        "group's moderation enforces, rendered as they stand "
        "now.</p>")
    parts.append(f"<p><i>{_esc(PAGE_FOOTER)}</i></p></body></html>")
    return "".join(parts)


def rules_index_html(surface: Any, *,
                     titles: dict[str, str] | None = None) -> str:
    slugs = rules_slug_map(surface, titles=titles)
    titles = titles or {}
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,"
        "initial-scale=1'>"
        "<title>Devon — group rules</title></head><body>",
        "<h1>📜 Group rules</h1>",
    ]
    if not slugs:
        parts.append(
            "<p>No group has published rules pages yet.</p>")
    else:
        parts.append("<ul>" + "".join(
            f"<li><a href='/rules/{_esc(slug)}'>"
            f"{_esc(titles.get(key) or 'Group rules')}</a></li>"
            for slug, key in sorted(slugs.items())) + "</ul>")
    parts.append(f"<p><i>{_esc(PAGE_FOOTER)}</i></p></body></html>")
    return "".join(parts)


# ── static bundle ───────────────────────────────────────────────────

def write_bundle(out_dir: str | Path, surface: Any, *,
                 bot_username: str = "",
                 titles: dict[str, str] | None = None
                 ) -> list[Path]:
    """Materialise the whole presence as static files (for any
    static host, or as the WordPress publish payload source):
    index.html (her page), rules/index.html, rules/<slug>.html.
    Returns the files written."""
    out = Path(out_dir)
    (out / "rules").mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    index = out / "index.html"
    index.write_text(devon_page_html(bot_username=bot_username),
                     encoding="utf-8")
    written.append(index)
    rindex = out / "rules" / "index.html"
    rindex.write_text(rules_index_html(surface, titles=titles),
                      encoding="utf-8")
    written.append(rindex)
    titles = titles or {}
    for slug, key in rules_slug_map(surface, titles=titles).items():
        page = out / "rules" / f"{slug}.html"
        page.write_text(
            rules_page_html(surface, key,
                            title=titles.get(key, "")),
            encoding="utf-8")
        written.append(page)
    return written


# ── WordPress ───────────────────────────────────────────────────────

#: transport(method, url, headers, body_dict) -> (status, json)
WPTransport = Callable[[str, str, dict, dict | None], tuple[int, Any]]


def _urllib_transport(method: str, url: str, headers: dict,
                      body: dict | None) -> tuple[int, Any]:
    data = None
    req_headers = dict(headers)
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        req_headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=req_headers,
                                 method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8", "replace")
            return resp.status, (json.loads(raw) if raw else {})
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
        try:
            return exc.code, json.loads(raw) if raw else {}
        except ValueError:
            return exc.code, {"error": raw[:200]}


class WordPressClient:
    """One WordPress site over its REST API, authenticated with
    an application password (Basic auth — the WP application-
    passwords shape).  Pages publish find-or-create by slug, so
    re-publishing updates in place instead of piling up copies.
    """

    def __init__(self, base_url: str, user: str, app_password: str,
                 *, transport: WPTransport | None = None) -> None:
        self.base_url = str(base_url or "").rstrip("/")
        self.user = str(user or "")
        self.app_password = str(app_password or "")
        self._transport = transport or _urllib_transport

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.user
                    and self.app_password)

    def _headers(self) -> dict[str, str]:
        token = base64.b64encode(
            f"{self.user}:{self.app_password}".encode("utf-8")
        ).decode("ascii")
        return {"Authorization": f"Basic {token}"}

    def _api(self, method: str, path: str,
             body: dict | None = None) -> tuple[int, Any]:
        return self._transport(method, f"{self.base_url}{path}",
                               self._headers(), body)

    def find_page(self, slug: str) -> dict[str, Any] | None:
        status, data = self._api(
            "GET", f"/wp-json/wp/v2/pages?slug={slug}&status=any")
        if status == 200 and isinstance(data, list) and data:
            return data[0]
        return None

    def publish_page(self, title: str, slug: str,
                     content_html: str) -> dict[str, Any]:
        """Create or update the page at ``slug``; returns the
        API's page object (id/link included) or raises with the
        API's own error."""
        existing = self.find_page(slug)
        payload = {"title": title, "slug": slug,
                   "content": content_html, "status": "publish"}
        if existing:
            status, data = self._api(
                "POST",
                f"/wp-json/wp/v2/pages/{existing['id']}", payload)
        else:
            status, data = self._api("POST", "/wp-json/wp/v2/pages",
                                     payload)
        if status not in (200, 201) or not isinstance(data, dict):
            message = ""
            if isinstance(data, dict):
                message = str(data.get("message")
                              or data.get("error") or "")
            raise RuntimeError(
                f"WordPress answered {status}"
                + (f": {message}" if message else ""))
        return data


def wordpress_clients(web: Any) -> dict[str, WordPressClient]:
    """The two sites from web settings: ``page`` (her page) and
    ``rules`` (the group-rules site)."""
    return {
        "page": WordPressClient(
            getattr(web, "wp_page_url", ""),
            getattr(web, "wp_page_user", ""),
            getattr(web, "wp_page_password", "")),
        "rules": WordPressClient(
            getattr(web, "wp_rules_url", ""),
            getattr(web, "wp_rules_user", ""),
            getattr(web, "wp_rules_password", "")),
    }


def publish_to_wordpress(web: Any, surface: Any, *,
                         bot_username: str = "",
                         titles: dict[str, str] | None = None
                         ) -> dict[str, Any]:
    """Publish both bundles to their WordPress sites.  The
    report is per site and HONEST: an unconfigured site reports
    "not configured" (its pages still serve locally), a failed
    publish reports the API's error, and only a real 200/201
    counts as published."""
    titles = titles or {}
    report: dict[str, Any] = {}
    clients = wordpress_clients(web)
    for name, client in clients.items():
        if not client.configured:
            report[name] = {
                "ok": False, "configured": False,
                "error": ("wordpress site not configured (needs "
                          "url + user + application password in "
                          "the web settings) — the bundle is "
                          "generated and served locally instead")}
            continue
        try:
            pages = []
            if name == "page":
                got = client.publish_page(
                    "Devon — a personal AI OS", "devon",
                    devon_page_html(bot_username=bot_username))
                pages.append({"slug": "devon",
                              "link": got.get("link", "")})
            else:
                got = client.publish_page(
                    "Group rules", "rules",
                    rules_index_html(surface, titles=titles))
                pages.append({"slug": "rules",
                              "link": got.get("link", "")})
                for slug, key in rules_slug_map(
                        surface, titles=titles).items():
                    got = client.publish_page(
                        titles.get(key) or "Group rules", slug,
                        rules_page_html(
                            surface, key,
                            title=titles.get(key, "")))
                    pages.append({"slug": slug,
                                  "link": got.get("link", "")})
            report[name] = {"ok": True, "configured": True,
                            "pages": pages}
        except Exception as exc:  # noqa: BLE001 - the site's own
            report[name] = {"ok": False, "configured": True,  # words
                            "error": str(exc)[:300]}
    return report


# ── serving (API context) ───────────────────────────────────────────

def _surface_for_context(context: Any) -> Any:
    from .chat.group_surface import default_surface

    settings = getattr(context, "settings", None)
    ws = getattr(settings, "workspace_dir", ".")
    return default_surface(ws)


def _titles_for_context(context: Any,
                        keys: list[str]) -> dict[str, str]:
    """Group titles from the gateway chats ledger, best-effort:
    a missing/odd DB simply means hash slugs + generic names."""
    out: dict[str, str] = {}
    db = getattr(context, "db", None)
    if db is None or not keys:
        return out
    try:
        rows = db.query(
            "SELECT key, title FROM chats WHERE key IN (%s)"
            % ",".join("?" for _ in keys), list(keys))
        for row in rows or []:
            key = row["key"] if isinstance(row, dict) else row[0]
            title = row["title"] if isinstance(row, dict) else row[1]
            if title:
                out[str(key)] = str(title)
    except Exception:  # noqa: BLE001 - titles are decoration;
        return out                 # slugs stay stable regardless
    return out


def render_site_path(context: Any, path: str) -> str | None:
    """Resolve one public site path to its HTML, or None (the
    server answers 404).  Paths: /devon, /rules, /rules/<slug>."""
    settings = getattr(context, "settings", None)
    web = getattr(settings, "web", None)
    if path == "/devon":
        return devon_page_html(
            bot_username=getattr(web, "bot_username", "") or "")
    surface = _surface_for_context(context)
    if path == "/rules":
        titles = _titles_for_context(
            context, claimed_group_keys(surface))
        return rules_index_html(surface, titles=titles)
    if path.startswith("/rules/"):
        slug = path[len("/rules/"):].strip("/")
        titles = _titles_for_context(
            context, claimed_group_keys(surface))
        key = rules_slug_map(surface, titles=titles).get(slug)
        if key is None:
            return None
        return rules_page_html(surface, key,
                               title=titles.get(key, ""))
    return None
