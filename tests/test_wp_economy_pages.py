"""WP economy-guide + standings pages (§12): the group economy
and the season standings as real pages, generated LIVE from the
same state the in-chat surfaces read.

Truths under test: the economy guide renders the group's REAL
treasury state (treasury, circulating, holders, policy — the same
numbers /treasury shows), the live faucet rates and the real shop
catalog; the standings page renders the §14 boards in rank order
(the same rows /top and /boards show) plus the season's top
earners; an unclaimed room publishes nothing; unknown slugs 404;
the static bundle materialises the new pages; the WordPress
publish carries them on the community site; /top and /treasury
carry the page link (URL button on the bot, bare link elsewhere).
"""
from __future__ import annotations

import time
from types import SimpleNamespace

from nomorals.agents.partner.runtime_group import RuntimeGroupMixin
from nomorals.community.group_economy import (
    DAILY_BASE,
    EARN_CAPS,
    EARN_TABLE,
    STATUS_SHOP,
    current_season,
)
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


def _fund(surface, member, amount, *, now=None):
    """Seed real sparks the way the economy wiring tests do:
    verify, then a ledger award (earn-capped at 200/day, so
    callers keep each member's award at or under the cap).
    The member record is written straight into the store — the
    message path would also pay a 5-spark streak, which would
    push a full-cap award over the daily limit."""
    now = now if now is not None else time.time()
    surface.store.group(BOT_KEY)["members"][member] = {
        "name": member.title(), "username": member}
    surface.economy.verify_member(BOT_KEY, member)
    entry = surface.economy.award(BOT_KEY, member, amount, "seed",
                                  now=now)
    assert entry is not None, f"award of {amount} blocked for {member}"
    return entry


def _health(surface):
    return surface.economy_health(BOT_KEY)


# ── the economy guide ───────────────────────────────────────────────

def test_economy_page_renders_live_treasury_state(tmp_path):
    surface = _surface(tmp_path)
    _fund(surface, "u1", 150)
    surface.pay(BOT_KEY, "u1", "u2", 100)
    health = _health(surface)
    assert health["treasury"] == 2  # the 2% transfer tax
    page = site.economy_page_html(surface, BOT_KEY,
                                  title="Sentinel Signals")
    assert "Sentinel Signals" in page
    assert "sparks" in page
    assert str(health["treasury"]) in page
    assert str(health["circulating"]) in page
    assert str(health["holders"]) in page
    assert "Sparks are game points only" in page
    assert "Updated" in page  # the §12 updated-at stamp
    assert "Powered by peacethefirst" in page


def test_economy_page_shows_live_faucet_rates_and_shop(tmp_path):
    surface = _surface(tmp_path)
    page = site.economy_page_html(surface, BOT_KEY)
    # the daily faucet, from the constants the code actually pays
    assert str(DAILY_BASE) in page
    assert "Daily" in page
    # the earn table, rendered as human labels, not internal ids
    assert "Helpful answer" in page
    assert str(EARN_TABLE["helpful_answer"]) in page
    assert "Invite" in page
    # the earn caps, from the same constants the organ enforces
    assert str(EARN_CAPS["daily"]) in page
    assert str(EARN_CAPS["weekly"]) in page
    # the status shop, with real costs
    assert "Shout-out" in page
    assert str(STATUS_SHOP["shout_out"]["cost"]) in page
    # the monetary policy line, honest when nothing is hot
    assert "normal" in page


def test_economy_page_names_throttled_policy(tmp_path):
    surface = _surface(tmp_path)
    now = time.time()
    # 6 members × the 200/day cap = 1,200 issued, 0 sunk:
    # issuance runs hot, so the policy throttles (activity pays
    # half) — the page must say so, like /treasury does.
    for i in range(6):
        _fund(surface, f"m{i}", 200, now=now)
    health = _health(surface)
    assert health["policy"]["throttled"] is True
    page = site.economy_page_html(surface, BOT_KEY)
    assert "throttled" in page
    assert "half" in page


def test_economy_page_escapes_group_content(tmp_path):
    surface = _surface(tmp_path)
    surface.update_settings(BOT_KEY, rules=["<script>alert(1)</script>"])
    page = site.economy_page_html(surface, BOT_KEY,
                                  title="<b>Bold Group</b>")
    assert "<script>" not in page
    assert "<b>Bold Group</b>" not in page
    assert "&lt;b&gt;Bold Group&lt;/b&gt;" in page


# ── the standings page ──────────────────────────────────────────────

def test_standings_page_renders_boards_in_rank_order(tmp_path):
    surface = _surface(tmp_path)
    _fund(surface, "ada", 150)
    _fund(surface, "ben", 100)
    _fund(surface, "cy", 50)
    page = site.standings_page_html(surface, BOT_KEY,
                                    title="Sentinel Signals")
    assert "Sentinel Signals" in page
    assert current_season() in page
    # richest first, in the same order /top shows
    assert page.index("Ada") < page.index("Ben") < page.index("Cy")
    assert "🥇" in page and "🥈" in page and "🥉" in page
    assert "Powered by peacethefirst" in page


def test_standings_page_names_season_top_earners(tmp_path):
    surface = _surface(tmp_path)
    _fund(surface, "ada", 150)
    _fund(surface, "ben", 100)
    page = site.standings_page_html(surface, BOT_KEY)
    # the season's top earners ride their own section, from the
    # same season_top the rollover crowns champions from
    assert "top earners" in page
    assert "Ada" in page


def test_standings_page_empty_group_has_no_fake_rows(tmp_path):
    surface = _surface(tmp_path)
    page = site.standings_page_html(surface, BOT_KEY)
    assert "no sparks held yet" in page
    assert "🥇" not in page


# ── indexes + the unclaimed-room rule ───────────────────────────────

def test_unclaimed_group_publishes_no_economy_or_standings(tmp_path):
    surface = _surface(tmp_path)
    assert FOREIGN_KEY not in site.economy_slug_map(surface).values()
    assert FOREIGN_KEY not in site.standings_slug_map(surface).values()
    assert "Foreign rules" not in site.economy_index_html(surface)
    assert "Foreign rules" not in site.standings_index_html(surface)


def test_economy_and_standings_share_the_rules_slug(tmp_path):
    """One slug per group, shared by all its pages — the economy
    and standings pages live at the same slug the rules page uses."""
    surface = _surface(tmp_path)
    rules = site.rules_slug_map(surface)
    assert site.economy_slug_map(surface) == rules
    assert site.standings_slug_map(surface) == rules
    slug = site.group_slug(BOT_KEY, "Sentinel Signals!")
    assert slug.startswith("sentinel-signals-")
    assert "1001234567" not in slug


# ── serving + the static bundle ─────────────────────────────────────

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


def test_site_paths_render_economy_and_standings_and_404(tmp_path):
    surface = _surface(tmp_path)
    surface.store.save()
    context = _api_context(tmp_path)
    assert site.render_site_path(context, "/economy")
    assert site.render_site_path(context, "/standings")
    slugs = site.economy_slug_map(default_surface(tmp_path))
    assert len(slugs) == 2
    for slug in slugs:
        got = site.render_site_path(context, f"/economy/{slug}")
        assert got and "sparks" in got
        got = site.render_site_path(context, f"/standings/{slug}")
        assert got and "standings" in got.lower()
    assert site.render_site_path(context, "/economy/nope") is None
    assert site.render_site_path(context, "/standings/nope") is None


def test_bundle_writer_materialises_economy_and_standings(tmp_path):
    surface = _surface(tmp_path)
    out = tmp_path / "bundle"
    written = site.write_bundle(out, surface,
                                bot_username="devonbot")
    names = sorted(str(p.relative_to(out)) for p in written)
    # index + rules index + 2 rules pages (before) + economy index
    # + 2 economy pages + standings index + 2 standings pages
    assert len(names) == 10
    assert "economy/index.html" in names
    assert "standings/index.html" in names
    assert sum(1 for n in names
               if n.startswith("economy/") and n != "economy/index.html") == 2
    assert sum(1 for n in names
               if n.startswith("standings/")
               and n != "standings/index.html") == 2


# ── WordPress: the community site carries the new pages ─────────────

def test_site_paths_served_publicly_over_http(tmp_path):
    """The new paths ride the same public branch as /rules: no
    token, HTML back, unknown slugs 404."""
    import http.client
    import threading
    from http.server import ThreadingHTTPServer

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

        status, ctype, body = get("/economy")
        assert status == 200 and "text/html" in ctype
        assert "economies" in body.lower()
        status, _, body = get("/standings")
        assert status == 200
        assert "standings" in body.lower()
        slugs = site.economy_slug_map(default_surface(tmp_path))
        slug = sorted(slugs)[0]
        status, _, body = get(f"/economy/{slug}")
        assert status == 200 and "sparks" in body
        status, _, body = get(f"/standings/{slug}")
        assert status == 200 and "standings" in body.lower()
        status, _, _ = get("/economy/definitely-not-a-group")
        assert status == 404
        status, _, _ = get("/standings/definitely-not-a-group")
        assert status == 404
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def test_publish_includes_economy_and_standings_pages(tmp_path):
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
    real_client = site.WordPressClient

    def client(base, user, pw):
        return real_client(base, user, pw, transport=transport)

    original = site.wordpress_clients
    site.wordpress_clients = lambda w: {
        "page": client(w.wp_page_url, w.wp_page_user,
                       w.wp_page_password),
        "rules": client(w.wp_rules_url, w.wp_rules_user,
                        w.wp_rules_password)}
    try:
        report = site.publish_to_wordpress(web, surface)
    finally:
        site.wordpress_clients = original
    assert report["page"]["ok"] is True
    assert report["rules"]["ok"] is True
    # rules site: rules index + 2 rules pages + economy index + 2
    # economy pages + standings index + 2 standings pages
    assert len(report["rules"]["pages"]) == 9
    posted = [c[3]["slug"] for c in calls if c[0] == "POST"]
    assert "economy" in posted and "standings" in posted
    slugs = site.rules_slug_map(surface)
    for slug in slugs:
        # one page per family: the WP publish is find-or-create
        # BY SLUG, so the economy/standings pages carry a
        # family suffix — the bare slug would overwrite the
        # group's rules page with the economy guide.
        assert posted.count(slug) == 1
        assert posted.count(f"{slug}-economy") == 1
        assert posted.count(f"{slug}-standings") == 1


# ── the in-chat cards link to their pages ───────────────────────────

def _bare_runtime(tmp_path, *, base_url):
    surface = _surface(tmp_path)
    rt = RuntimeGroupMixin.__new__(RuntimeGroupMixin)
    rt._group_surface = lambda: surface  # type: ignore[attr-defined]
    rt._group_key_or_none = (  # type: ignore[attr-defined]
        lambda message: message.chat.key)
    rt.gateway = None
    rt.context = SimpleNamespace(settings=SimpleNamespace(
        web=SimpleNamespace(base_url=base_url)))
    return rt


def _msg(platform, key):
    return SimpleNamespace(
        chat=SimpleNamespace(platform=platform, key=key))


def test_top_card_gains_standings_page_link(tmp_path):
    rt = _bare_runtime(tmp_path,
                       base_url="https://devon.example/")
    _fund(rt._group_surface(), "ada", 150)
    from nomorals.social.chat.tgbot_buttons import ActionableMessage
    out = rt._control_top(_msg("telegram-bot", BOT_KEY))
    assert isinstance(out, ActionableMessage)
    label, url = out.buttons[0][0]
    assert label == "🏆 Standings page"
    assert url.startswith("https://devon.example/standings/group-")
    assert "1001234567" not in url
    # userbot surface (inline keyboards dropped there): bare link
    out = rt._control_top(_msg("telegram", BOT_KEY))
    assert isinstance(out, str)
    assert "🏆 full standings page: https://devon.example/standings/" in out


def test_treasury_card_gains_economy_guide_link(tmp_path):
    rt = _bare_runtime(tmp_path,
                       base_url="https://devon.example/")
    _fund(rt._group_surface(), "ada", 150)
    from nomorals.social.chat.tgbot_buttons import ActionableMessage
    out = rt._control_treasury(_msg("telegram-bot", BOT_KEY))
    assert isinstance(out, ActionableMessage)
    label, url = out.buttons[0][0]
    assert label == "🪙 Economy guide"
    assert url.startswith("https://devon.example/economy/group-")
    assert "1001234567" not in url
    out = rt._control_treasury(_msg("telegram", BOT_KEY))
    assert isinstance(out, str)
    assert "🪙 full economy guide: https://devon.example/economy/" in out


def test_cards_unchanged_without_a_base_url(tmp_path):
    rt = _bare_runtime(tmp_path, base_url="")
    _fund(rt._group_surface(), "ada", 150)
    out = rt._control_top(_msg("telegram-bot", BOT_KEY))
    assert isinstance(out, str)
    assert "http" not in out
    out = rt._control_treasury(_msg("telegram-bot", BOT_KEY))
    assert isinstance(out, str)
    assert "http" not in out


def test_page_links_skipped_for_unclaimed_rooms(tmp_path):
    rt = _bare_runtime(tmp_path,
                       base_url="https://devon.example/")
    out = rt._control_top(_msg("telegram-bot", FOREIGN_KEY))
    assert isinstance(out, str)
    assert "aren't running" in out or "economy is off" in out \
        or "http" not in out
