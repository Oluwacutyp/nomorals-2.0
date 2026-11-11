"""P4c — Devon's web presence: /devon, /rules (+per-group
pages), the static bundle, and the two WordPress sites.

Truths under test: the Devon page is GENERATED from the live
command registry (a command present renders; one removed is
gone — no hand-copied catalog), a claimed group's page renders
its REAL rules (escaped), an unclaimed room publishes nothing,
an unknown slug 404s, slugs never carry the platform's numeric
ids, WordPress publishing speaks the real REST shape against a
fake server (Basic auth from the application password,
find-or-create by slug), and an unconfigured site reports
"not configured" honestly while the local pages still serve.
"""
from __future__ import annotations

import base64
import http.client
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

from nomorals.social import site
from nomorals.social.chat.group_surface import default_surface

BOT_KEY = "telegram-bot:-1001234567"     # her bot runs here
USER_KEY = "telegram:555001"             # designated userbot room
FOREIGN_KEY = "telegram:999002"          # she merely sits here


def _surface(tmp_path):
    surface = default_surface(tmp_path)
    surface.update_settings(
        BOT_KEY, rules=["No <spam> & scams", "Be kind — always"])
    surface.update_settings(USER_KEY, rules=["Rule one: trade fair"])
    surface.set_designated(USER_KEY, True, by="99")
    surface.update_settings(FOREIGN_KEY, rules=["Foreign rules"])
    return surface


# ── slugs + the rules pages ─────────────────────────────────────────

def test_slugs_never_carry_numeric_ids(tmp_path):
    surface = _surface(tmp_path)
    slugs = site.rules_slug_map(surface)
    assert len(slugs) == 2  # claimed rooms only
    for slug in slugs:
        assert "1001234567" not in slug
        assert "555001" not in slug
    titled = site.group_slug(BOT_KEY, "Sentinel Signals!")
    assert titled.startswith("sentinel-signals-")
    assert site.group_slug(BOT_KEY, "Sentinel Signals!") == titled


def test_rules_page_renders_real_rules_escaped(tmp_path):
    surface = _surface(tmp_path)
    page = site.rules_page_html(surface, BOT_KEY,
                                title="Sentinel Signals")
    assert "Sentinel Signals" in page
    assert "No &lt;spam&gt; &amp; scams" in page  # escaped
    assert "<spam>" not in page                    # never raw
    assert "Be kind — always" in page
    assert "Moderation:" in page


def test_unclaimed_group_publishes_nothing(tmp_path):
    surface = _surface(tmp_path)
    slugs = site.rules_slug_map(surface)
    assert FOREIGN_KEY not in slugs.values()
    index = site.rules_index_html(surface)
    assert "Foreign rules" not in index


def test_bundle_writer_materialises_pages(tmp_path):
    surface = _surface(tmp_path)
    out = tmp_path / "bundle"
    written = site.write_bundle(out, surface,
                                bot_username="devonbot")
    names = sorted(str(p.relative_to(out)) for p in written)
    assert names[0] == "index.html"
    assert "rules/index.html" in names
    assert len(names) == 4  # index + rules index + 2 group pages
    index_html = (out / "index.html").read_text(encoding="utf-8")
    assert "@devonbot" in index_html


# ── the Devon page tracks the live registry ─────────────────────────

def test_devon_page_generated_from_registry():
    page = site.devon_page_html(
        groups=[("test group", ["zzalpha", "zzbeta"])],
        oneliners={"zzalpha": "does alpha things",
                   "zzbeta": "does beta things"})
    assert "/zzalpha" in page and "does alpha things" in page
    shrunk = site.devon_page_html(
        groups=[("test group", ["zzalpha"])],
        oneliners={"zzalpha": "does alpha things"})
    assert "/zzbeta" not in shrunk  # a removed command disappears


def test_devon_page_default_reads_the_live_registry():
    page = site.devon_page_html()
    # the P4c commands themselves appear — proof the page is
    # generated from the registry, not a hand-written list
    assert "/massreact" in page and "/pollvote" in page
    assert "/chfollow" in page and "/pair" in page
    assert "Generated from her live command registry" in page


# ── serving through the real API server ─────────────────────────────

def _api_context(tmp_path):
    from nomorals.storage.db import Database

    return SimpleNamespace(
        db=Database(":memory:"),
        settings=SimpleNamespace(
            api=SimpleNamespace(token="", max_body_mb=0),
            backup_dir=str(tmp_path), workspace_dir=str(tmp_path),
            web=SimpleNamespace(
                base_url="", bot_username="devonbot",
                wp_page_url="", wp_page_user="",
                wp_page_password="", wp_rules_url="",
                wp_rules_user="", wp_rules_password=""),
            chat=SimpleNamespace(telegram_api_id="",
                                 telegram_api_hash="")),
        tools=None, bus=None, memory=None)


def test_site_paths_render_and_404(tmp_path):
    surface = _surface(tmp_path)
    surface.store.save()
    context = _api_context(tmp_path)
    page = site.render_site_path(context, "/devon")
    assert page and "Devon" in page and "@devonbot" in page
    index = site.render_site_path(context, "/rules")
    assert index and "Group rules" in index
    slugs = site.rules_slug_map(
        default_surface(tmp_path))
    assert len(slugs) == 2
    for slug in slugs:
        got = site.render_site_path(context, f"/rules/{slug}")
        assert got and ("spam" in got or "trade fair" in got)
    assert site.render_site_path(context, "/rules/nope") is None
    assert site.render_site_path(context, "/other") is None


def test_site_paths_served_publicly_over_http(tmp_path):
    from nomorals.api.server import APIServer, _make_handler

    surface = _surface(tmp_path)
    surface.store.save()
    context = _api_context(tmp_path)
    api = APIServer(context, token="owner-token", principals={},
                    max_body_bytes=64 * 1024)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(api))
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever,
                              daemon=True)
    thread.start()
    try:
        def get(path):
            conn = http.client.HTTPConnection("127.0.0.1", port,
                                              timeout=10)
            conn.request("GET", path)  # no token at all
            resp = conn.getresponse()
            body = resp.read().decode("utf-8", "replace")
            conn.close()
            return resp.status, resp.getheader("Content-Type"), body

        status, ctype, body = get("/devon")
        assert status == 200 and "text/html" in ctype
        assert "Devon" in body
        slugs = site.rules_slug_map(default_surface(tmp_path))
        slug = sorted(slugs)[0]
        status, _, body = get(f"/rules/{slug}")
        assert status == 200 and "rules" in body.lower()
        status, _, _ = get("/rules/definitely-not-a-group")
        assert status == 404
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def test_site_publish_route_is_owner_gated(tmp_path):
    from nomorals.api.server import APIServer, Principal, _make_handler
    from nomorals.core.policy import Capability, CapabilitySet

    surface = _surface(tmp_path)
    surface.store.save()
    context = _api_context(tmp_path)
    api = APIServer(
        context, token="owner-token",
        principals={"ro-token": Principal(
            "ro", CapabilitySet.of(Capability.DB_READ))},
        max_body_bytes=64 * 1024)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(api))
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever,
                              daemon=True)
    thread.start()
    try:
        def post(path, token=""):
            conn = http.client.HTTPConnection("127.0.0.1", port,
                                              timeout=10)
            headers = {"Authorization": f"Bearer {token}"} \
                if token else {}
            conn.request("POST", path, body=b"{}", headers=headers)
            resp = conn.getresponse()
            body = resp.read().decode("utf-8", "replace")
            conn.close()
            return resp.status, body

        status, _ = post("/site/publish")
        assert status == 401            # no token, no publish
        status, _ = post("/site/publish", "ro-token")
        assert status == 403            # a reader is not the owner
        status, body = post("/site/publish", "owner-token")
        assert status == 200
        report = json.loads(body)["sites"]
        # nothing configured in this fixture → honest report,
        # delivered over the owner's own API
        assert report["page"]["configured"] is False
        assert "not configured" in report["page"]["error"]
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


# ── WordPress: the fake REST server ─────────────────────────────────

class _FakeWP:
    """A minimal WordPress REST shape: list pages by slug,
    create, update — with the auth header recorded."""

    def __init__(self):
        self.pages: dict[int, dict] = {}
        self.next_id = 11
        self.auths: list[str] = []
        self.posts: list[tuple[str, dict]] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # silence
                pass

            def _reply(self, status, obj):
                raw = json.dumps(obj).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                outer.auths.append(
                    self.headers.get("Authorization", ""))
                from urllib.parse import parse_qs, urlparse
                q = parse_qs(urlparse(self.path).query)
                slug = (q.get("slug") or [""])[0]
                found = [p for p in outer.pages.values()
                         if p["slug"] == slug]
                self._reply(200, found)

            def do_POST(self):
                outer.auths.append(
                    self.headers.get("Authorization", ""))
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(
                    self.rfile.read(length).decode() or "{}")
                outer.posts.append((self.path, body))
                if self.path == "/wp-json/wp/v2/pages":
                    pid = outer.next_id
                    outer.next_id += 1
                    page = {"id": pid, **body,
                            "link": f"https://wp.example/{body['slug']}/"}
                    outer.pages[pid] = page
                    self._reply(201, page)
                    return
                pid = int(self.path.rsplit("/", 1)[1])
                outer.pages[pid].update(body)
                self._reply(200, outer.pages[pid])

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(
            target=self.httpd.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}"


def test_wordpress_publish_create_then_update():
    with _FakeWP() as wp:
        client = site.WordPressClient(wp.url, "editor",
                                      "app-pass-1234")
        assert client.configured is True
        page = client.publish_page("Devon", "devon", "<p>v1</p>")
        assert page["id"] == 11
        assert page["link"].endswith("/devon/")
        page2 = client.publish_page("Devon", "devon", "<p>v2</p>")
        assert page2["id"] == 11  # updated in place, not duplicated
        assert len(wp.pages) == 1
        assert wp.pages[11]["content"] == "<p>v2</p>"
        expected = "Basic " + base64.b64encode(
            b"editor:app-pass-1234").decode()
        assert wp.auths and all(a == expected for a in wp.auths)
        create = wp.posts[0][1]
        assert create["slug"] == "devon"
        assert create["status"] == "publish"
        assert create["title"] == "Devon"


def test_publish_both_sites_with_fake_transport(tmp_path):
    surface = _surface(tmp_path)
    calls: list[tuple] = []

    def transport(method, url, headers, body):
        calls.append((method, url, headers, body))
        if method == "GET":
            return 200, []
        return 201, {"id": 5, "link": "https://x/p/"}

    web = SimpleNamespace(
        wp_page_url="https://page.example", wp_page_user="u1",
        wp_page_password="p1", wp_rules_url="https://rules.example",
        wp_rules_user="u2", wp_rules_password="p2")
    import nomorals.social.site as site_mod
    real_client = site_mod.WordPressClient

    def client(base, user, pw):
        return real_client(base, user, pw, transport=transport)

    original = site_mod.wordpress_clients
    site_mod.wordpress_clients = lambda w: {
        "page": client(w.wp_page_url, w.wp_page_user,
                       w.wp_page_password),
        "rules": client(w.wp_rules_url, w.wp_rules_user,
                        w.wp_rules_password)}
    try:
        report = site_mod.publish_to_wordpress(web, surface)
    finally:
        site_mod.wordpress_clients = original
    assert report["page"]["ok"] is True
    assert report["rules"]["ok"] is True
    # rules site: index + one page per claimed group
    assert len(report["rules"]["pages"]) == 3
    auths = {c[2]["Authorization"] for c in calls if c[2]}
    assert "Basic " + base64.b64encode(b"u1:p1").decode() in auths
    assert "Basic " + base64.b64encode(b"u2:p2").decode() in auths


def test_wordpress_unconfigured_reports_honestly(tmp_path):
    surface = _surface(tmp_path)
    web = SimpleNamespace(
        wp_page_url="", wp_page_user="", wp_page_password="",
        wp_rules_url="", wp_rules_user="", wp_rules_password="")
    report = site.publish_to_wordpress(web, surface)
    for name in ("page", "rules"):
        assert report[name]["ok"] is False
        assert report[name]["configured"] is False
        assert "not configured" in report[name]["error"]
        assert "served locally" in report[name]["error"]
    # and the local pages genuinely exist regardless
    assert site.rules_index_html(surface)


# ── the rules card link machinery ───────────────────────────────────

def test_actionable_url_button_skips_callback_budget():
    from nomorals.social.chat.tgbot_buttons import actionable

    url = ("https://devon.example/rules/"
           "a-quite-long-group-title-slug-abcdef0123")
    assert len(url) > 47  # beyond the callback_data budget
    msg = actionable("rules text", [[("📜 Rules page", url)]])
    rows = msg.buttons
    assert rows[0][0] == ("📜 Rules page", url)


def _bare_rules_runtime(tmp_path, *, base_url):
    from nomorals.agents.partner.runtime_group import (
        RuntimeGroupMixin)

    surface = _surface(tmp_path)
    rt = RuntimeGroupMixin.__new__(RuntimeGroupMixin)
    rt._group_surface = lambda: surface  # type: ignore[attr-defined]
    rt._group_key_or_none = (  # type: ignore[attr-defined]
        lambda message: message.chat.key)
    rt.gateway = None
    rt.context = SimpleNamespace(settings=SimpleNamespace(
        web=SimpleNamespace(base_url=base_url)))
    return rt


def _rules_msg(platform, key):
    return SimpleNamespace(
        chat=SimpleNamespace(platform=platform, key=key))


def test_rules_card_gains_its_page_link(tmp_path):
    rt = _bare_rules_runtime(
        tmp_path, base_url="https://devon.example/")
    # bot surface: a native URL button rides the card
    out = rt._control_rules(_rules_msg("telegram-bot", BOT_KEY))
    from nomorals.social.chat.tgbot_buttons import ActionableMessage
    assert isinstance(out, ActionableMessage)
    assert "No &lt;spam&gt; &amp; scams" in out.text
    label, url = out.buttons[0][0]
    assert label == "📜 Rules page"
    assert url.startswith("https://devon.example/rules/group-")
    assert "1001234567" not in url  # no numeric id in the link
    # userbot surface (inline keyboards dropped there): bare link
    out = rt._control_rules(_rules_msg("telegram", BOT_KEY))
    assert isinstance(out, str)
    assert "📜 full rules page: https://devon.example/rules/" in out
    # an unclaimed room publishes no page → no link
    out = rt._control_rules(_rules_msg("telegram", FOREIGN_KEY))
    assert isinstance(out, str) and "rules page" not in out


def test_rules_card_unchanged_without_a_base_url(tmp_path):
    rt = _bare_rules_runtime(tmp_path, base_url="")
    out = rt._control_rules(_rules_msg("telegram-bot", BOT_KEY))
    assert isinstance(out, str)
    assert "No &lt;spam&gt; &amp; scams" in out
    assert "http" not in out


def test_bot_adapter_renders_url_buttons():
    from nomorals.social.chat.base import ChatRef, ChatKind
    from nomorals.social.chat.telegram import TelegramBotAdapter

    adapter = TelegramBotAdapter(token="x", session=object())
    captured: dict = {}

    def fake_api(method, **params):
        captured["method"] = method
        captured.update(params)
        return {"message_id": 42}

    adapter._api = fake_api  # type: ignore[method-assign]
    chat = ChatRef(platform="telegram-bot", chat_id="-1001",
                   kind=ChatKind.GROUP)
    url = "https://devon.example/rules/group-abcdef0123"
    result = adapter.send(chat, "📜 rules",
                          buttons=[[("📜 Rules page", url)]])
    assert result.ok
    raw_markup = captured["reply_markup"]
    markup = json.loads(raw_markup) if isinstance(raw_markup, str) \
        else raw_markup
    button = markup["inline_keyboard"][0][0]
    assert button == {"text": "📜 Rules page", "url": url}
    # a callback payload still signs + stays callback_data
    adapter.send(chat, "📜 rules", buttons=[[("Back", "nav:back")]])
    raw_markup = captured["reply_markup"]
    markup = json.loads(raw_markup) if isinstance(raw_markup, str) \
        else raw_markup
    button = markup["inline_keyboard"][0][0]
    assert "callback_data" in button and "url" not in button


# ── the publish enumeration is a consistent cut ─────────────────

def test_claim_gate_never_materialises_rooms(tmp_path):
    """The §33 gate is a pure read: enumerating the claimed
    rooms — and publishing the whole bundle — must leave the
    store's key set exactly as it stood.  No phantom Telegram
    family siblings may be recorded for a room she has only
    ever seen under one surface's key (that write, landing
    mid-iteration, is what killed every publish with
    'dictionary changed size during iteration')."""
    surface = _surface(tmp_path)
    before = set(surface.store.data)
    assert site.claimed_group_keys(surface)  # claimed rooms exist
    assert set(surface.store.data) == before
    site.write_bundle(tmp_path / "bundle", surface,
                      bot_username="devonbot")
    assert set(surface.store.data) == before


def test_room_registered_mid_publish_joins_the_next_pass(tmp_path):
    """A room registered while a publish pass is enumerating
    the store (her live surfaces write the same store) neither
    breaks the pass nor half-joins it: the pass answers from
    the keys it began with, and the new room publishes whole
    on the next pass."""
    surface = _surface(tmp_path)
    new_key = "whatsapp:777001"
    real_gate = surface.system_output_allowed
    registered = []

    def gate(key, **kwargs):
        if not registered:
            registered.append(key)
            surface.update_settings(new_key, rules=["Late rules"])
            surface.set_designated(new_key, True, by="99")
        return real_gate(key, **kwargs)

    surface.system_output_allowed = gate
    try:
        slugs = site.rules_slug_map(surface)
    finally:
        surface.system_output_allowed = real_gate
    assert registered  # the registration really fired mid-pass
    assert new_key not in slugs.values()
    # the very next pass carries the new room whole
    assert new_key in site.rules_slug_map(surface).values()
