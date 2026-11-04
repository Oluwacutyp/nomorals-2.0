"""Webshot (§18/§20) + the owner system (§19/§20).

Pins:
- the shot grammar carries the whole variant space on ONE
  command (full/viewport, device profiles, dark, idle, clip,
  pdf, annotate) and reply-targets a link.
- the ladder is honest: a fake local renderer yields real PNG
  bytes at the profile's size (full-page taller than the
  viewport); provider seeds answer only the options they can
  honor; every rung down is an explicit failure naming each
  rung's reason — never a fake image.
- URL discipline: private/loopback targets refused for group
  callers, data:text/html allowed, non-http(s) refused.
- the journal records successes and failures; a corrupt store
  raises after quarantining.
- page-watch: baseline silently, alert only on a real change
  (pixel-diff above threshold), silence when unchanged.
- the watermark (§19): resolve_media_mark reads the ONE
  presentation home — modes off/text/media/both, the master
  stamp switch, presented classes only, group text beats the
  owner's global mark; the drawn stamp changes the
  bottom-right pixels of a real image and nothing else.
- the owner record: bind/sudo/deputy roles (deputy is
  group-scoped), WhatsApp digit identity, edits refused for
  non-owners and refused from groups, the held audit, sealed
  rebind (wrong passphrase refused; no seal = unavailable),
  corrupt record raises.
- impersonation: the owner's handle worn by a stranger is
  flagged; the owner wearing it is not.
- the surface: owner/sudo roles land on the member ledger;
  sudo is exempt from the automatic spine while a plain
  member is not.
- the handlers: /owner refuses every verb to a non-operator
  and refuses edits in a group even to the operator; the
  card + status compose from seeded state; /webshot in an
  unclaimed room is §33-refused before any capture, in DM it
  shoots and sends; the watch driver posts a change alert.
"""
from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from nomorals.agents.partner.runtime_group import RuntimeGroupMixin
from nomorals.social.chat import webshot
from nomorals.social.chat.base import ChatKind
from nomorals.social.chat.group_surface import (
    GroupStore, GroupSurface)
from nomorals.social.chat.owner_system import (
    OwnerAuthorityError, OwnerRegistry, OwnerStoreError,
    apply_media_watermark, impersonation_check, status_card,
    verification_card)
from nomorals.social.chat.typography import (
    DEFAULT_STAMP_TEXT, resolve_media_mark)
from nomorals.social.chat.webshot import (
    CaptureResult, ShotOptions, WebshotBook, WebshotStoreError,
    WebshotUnavailable)

KEY = "telegram-bot:-1001"
WA_KEY = "whatsapp:120363@g.us"
NOW = 1_800_000_000.0


def _png(width: int, height: int, color=(255, 255, 255)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buf, "PNG")
    return buf.getvalue()


def _renderer(png: bytes, title: str = "Example Domain"):
    """A fake local rung honoring the variant space the way
    the real one does: profile-sized viewport shots, a taller
    full page, %PDF bytes for the print rung."""

    def render(url: str, opts: ShotOptions) -> CaptureResult:
        if opts.pdf:
            return CaptureResult(data=b"%PDF-1.4 fake print",
                                 title=title, final_url=url,
                                 pdf=True)
        profile = webshot.DEVICE_PROFILES[opts.device]
        w = profile["width"] * profile["scale"]
        h = profile["height"] * profile["scale"]
        if opts.full_page:
            h = h * 3
        if opts.clip:
            w, h = 300, 200
        return CaptureResult(data=_png(w, h), title=title,
                             final_url=url)

    return render


def _down(url: str, opts: ShotOptions) -> CaptureResult:
    raise RuntimeError("chromium exploded")


# ── grammar + URL discipline ─────────────────────────────────

def test_grammar_carries_the_variant_space():
    url, opts = webshot.parse_shot_tail(
        "full phone dark idle annotate https://example.com/x")
    assert url == "https://example.com/x"
    assert opts.full_page and opts.device == "phone"
    assert opts.scheme == "dark" and opts.wait == "idle"
    assert opts.annotate
    url, opts = webshot.parse_shot_tail(
        "clip .price https://example.com")
    assert opts.clip == ".price"
    url, opts = webshot.parse_shot_tail(
        "", reply_url="https://replied.example/page")
    assert url == "https://replied.example/page"
    url, _ = webshot.parse_shot_tail("example.com/bare")
    assert url == "https://example.com/bare"
    with pytest.raises(webshot.ShotUsageError):
        webshot.parse_shot_tail("sideways https://example.com")
    with pytest.raises(webshot.ShotUsageError):
        webshot.parse_shot_tail("full")


def test_url_discipline():
    assert webshot.validate_url(
        "data:text/html,<h1>hi</h1>").startswith("data:")
    with pytest.raises(webshot.ShotUsageError):
        webshot.validate_url("ftp://example.com/x")
    with pytest.raises(webshot.ShotUsageError):
        webshot.validate_url("http://127.0.0.1:8080/admin")
    assert webshot.validate_url(
        "http://127.0.0.1:8080/admin", allow_private=True)
    with pytest.raises(webshot.ShotUsageError):
        webshot.validate_url("data:image/png;base64,AAAA")
    assert webshot.extract_url(
        "look at https://a.example/b, then reply") \
        == "https://a.example/b"


# ── the ladder ───────────────────────────────────────────────

def test_capture_local_rung_real_png_and_full_page_taller():
    shot = webshot.capture("data:text/html,<h1>x</h1>",
                           ShotOptions(),
                           local_renderer=_renderer(_png(4, 4)))
    assert shot.rung == "playwright"
    assert (shot.width, shot.height) == (1440, 900)
    assert shot.data[:8] == b"\x89PNG\r\n\x1a\n"
    full = webshot.capture("data:text/html,<h1>x</h1>",
                           ShotOptions(full_page=True),
                           local_renderer=_renderer(_png(4, 4)))
    assert full.height > shot.height
    pdf = webshot.capture("data:text/html,<h1>x</h1>",
                          ShotOptions(pdf=True),
                          local_renderer=_renderer(_png(4, 4)))
    assert pdf.pdf and pdf.data.startswith(b"%PDF")


def test_capture_falls_to_a_provider_seed():
    def fetcher(url, **kw):
        if "microlink" in url:
            return 0, b"", "connection refused"
        if "thum.io" in url:
            return 200, _png(1440, 900), ""
        return 0, b"", "unknown"

    shot = webshot.capture("https://example.com",
                           ShotOptions(), local_renderer=_down,
                           fetcher=fetcher)
    assert shot.rung == "thumio"
    assert (shot.width, shot.height) == (1440, 900)


def test_capture_all_rungs_down_is_an_honest_failure():
    def fetcher(url, **kw):
        return 0, b"", "offline"

    with pytest.raises(WebshotUnavailable) as caught:
        webshot.capture("https://example.com", ShotOptions(),
                        local_renderer=_down, fetcher=fetcher)
    reasons = " ".join(caught.value.reasons)
    assert "chromium exploded" in reasons
    assert "microlink" in reasons and "thumio" in reasons


def test_providers_never_silently_dishonor_options():
    called = []

    def fetcher(url, **kw):
        called.append(url)
        return 200, _png(100, 100), ""

    with pytest.raises(WebshotUnavailable) as caught:
        webshot.capture("https://example.com",
                        ShotOptions(scheme="dark"),
                        local_renderer=_down, fetcher=fetcher)
    assert called == []  # no provider was even attempted
    assert "cannot honor" in " ".join(caught.value.reasons)


# ── the book: journal + watches ──────────────────────────────

def test_journal_records_and_corruption_raises(tmp_path):
    book = WebshotBook(tmp_path).load()
    book.record(url="https://a.example", ok=True, rung="x",
                size=10, scope="dm", requester="42")
    book.record(url="https://b.example", ok=False,
                error="boom", scope="dm", requester="42")
    rows = book.journal()
    assert [r["ok"] for r in rows] == [True, False]
    assert rows[1]["error"] == "boom"
    (tmp_path / "webshots.json").write_text("{broken",
                                            encoding="utf-8")
    with pytest.raises(WebshotStoreError):
        WebshotBook(tmp_path).load()
    assert list(tmp_path.glob("webshots.corrupt-*"))


def test_watch_baselines_then_alerts_only_on_change(tmp_path):
    book = WebshotBook(tmp_path).load()
    book.add_watch(url="https://example.com",
                   options=ShotOptions(), scope="dm:1",
                   requester="42", interval_s=900, ts=NOW)
    state = {"png": _png(1440, 900, (255, 255, 255))}

    def renderer(url, opts):
        return CaptureResult(data=state["png"], title="T",
                             final_url=url)

    alerts = book.poll_watches(now=NOW + 1,
                               local_renderer=renderer)
    assert alerts == []                      # baseline, silent
    alerts = book.poll_watches(now=NOW + 100,
                               local_renderer=renderer)
    assert alerts == []                      # not due yet
    state["png"] = _png(1440, 900, (0, 0, 0))
    alerts = book.poll_watches(now=NOW + 1000,
                               local_renderer=renderer)
    assert len(alerts) == 1
    assert alerts[0]["ratio"] > webshot.CHANGE_THRESHOLD
    alerts = book.poll_watches(now=NOW + 2000,
                               local_renderer=renderer)
    assert alerts == []                      # unchanged again
    assert webshot.change_ratio(state["png"], state["png"]) \
        == 0.0


# ── the watermark (§19) ──────────────────────────────────────

def test_media_mark_resolution_reads_the_one_home():
    mark = resolve_media_mark(None, "announcement")
    assert mark.mode == "text" and not mark.image
    assert mark.text == DEFAULT_STAMP_TEXT
    settings = {"presentation": {"watermark": "media"}}
    mark = resolve_media_mark(settings, "announcement")
    assert mark.image and mark.text == ""
    assert mark.image_text == DEFAULT_STAMP_TEXT
    settings = {"presentation": {"watermark": "both",
                                 "stamp_text": "— Ada's house"}}
    mark = resolve_media_mark(settings, "leaderboard")
    assert mark.image and mark.text == "— Ada's house"
    assert mark.image_text == "— Ada's house"
    settings = {"presentation": {"watermark": "off"}}
    assert resolve_media_mark(settings,
                              "announcement").mode == "off"
    settings = {"presentation": {"watermark": "both",
                                 "stamp": False}}
    mark = resolve_media_mark(settings, "announcement")
    assert not mark.image and mark.text == ""   # master switch
    mark = resolve_media_mark(
        {"presentation": {"watermark": "both"}}, "chat")
    assert not mark.image and mark.text == ""   # never chat
    mark = resolve_media_mark(
        {"presentation": {"watermark": "media"}}, "game",
        mark_text="Powered by the owner")
    assert mark.image_text == "Powered by the owner"
    mark = resolve_media_mark(
        {"presentation": {"watermark": "sideways"}},
        "announcement")
    assert mark.mode == "text"                  # data, not law


def test_watermark_stamp_changes_only_the_mark_region():
    raw = _png(800, 600, (250, 250, 250))
    stamped, applied = apply_media_watermark(raw, "Powered by X")
    assert applied and stamped != raw
    with Image.open(io.BytesIO(raw)) as before, \
            Image.open(io.BytesIO(stamped)) as after:
        assert after.size == before.size
        assert before.getpixel((10, 10)) \
            == after.getpixel((10, 10))          # untouched
        region_before = before.crop((500, 500, 800, 600))
        region_after = after.crop((500, 500, 800, 600))
        assert list(region_before.getdata()) \
            != list(region_after.getdata())      # the pill
    same, applied = apply_media_watermark(raw, "")
    assert not applied and same == raw
    same, applied = apply_media_watermark(b"not an image", "X")
    assert not applied and same == b"not an image"


# ── the owner record ─────────────────────────────────────────

def _registry(tmp_path) -> OwnerRegistry:
    return OwnerRegistry(tmp_path / "owner.json").load()


def test_registry_roles_and_the_ladder(tmp_path):
    reg = _registry(tmp_path)
    assert reg.bind("owner", via_group=False,
                    platform="telegram-bot", user_id="42")
    assert reg.role_of("telegram", "42") == "owner"
    assert reg.set_sudo("owner", via_group=False,
                        platform="telegram", user_id="7",
                        on=True)
    assert reg.role_of("telegram-bot", "7") == "sudo"
    assert reg.set_deputy("owner", via_group=False,
                          group_key=KEY, user_id="8", on=True)
    assert reg.role_of("telegram", "8", group_key=KEY) \
        == "deputy"
    assert reg.role_of("telegram", "8",
                       group_key="telegram-bot:-2002") == ""
    assert reg.unbind("owner", via_group=False,
                      platform="telegram", user_id="42")
    assert reg.role_of("telegram", "42") == ""
    actions = [row["action"] for row in reg.audit()]
    assert "bind" in actions and "sudo_add" in actions
    assert "deputy_add" in actions and "unbind" in actions


def test_registry_whatsapp_identity_is_the_digits(tmp_path):
    reg = _registry(tmp_path)
    reg.bind("owner", via_group=False, platform="whatsapp",
             user_id="2348012345678")
    assert reg.is_owner("whatsapp",
                        "2348012345678@s.whatsapp.net")
    assert not reg.is_owner("whatsapp", "2349999999999")


def test_registry_edits_require_the_owner_outside_groups(
        tmp_path):
    reg = _registry(tmp_path)
    edits = [
        lambda role, via: reg.bind(
            role, via_group=via, platform="telegram",
            user_id="1"),
        lambda role, via: reg.set_sudo(
            role, via_group=via, platform="telegram",
            user_id="1", on=True),
        lambda role, via: reg.set_deputy(
            role, via_group=via, group_key=KEY, user_id="1",
            on=True),
        lambda role, via: reg.set_mark_text(
            role, via_group=via, text="mark"),
    ]
    for edit in edits:
        with pytest.raises(OwnerAuthorityError):
            edit("sudo", False)      # sudo is not ownership
        with pytest.raises(OwnerAuthorityError):
            edit("", False)          # a stranger
        with pytest.raises(OwnerAuthorityError):
            edit("owner", True)      # never from a group
    assert reg.audit() == []         # refused edits never land


def test_registry_seed_fills_from_existing_bindings(tmp_path):
    reg = _registry(tmp_path)
    account = SimpleNamespace(platform="whatsapp",
                              identity="2348012345678",
                              remote_id="")
    changed = reg.seed(telegram_ids={"4242"}, accounts=[account],
                       handles=["oluwacutyp"])
    assert changed
    assert reg.is_owner("telegram", "4242")
    assert reg.is_owner("whatsapp", "2348012345678")
    assert reg.handles() == ["oluwacutyp"]


def test_registry_corruption_raises_and_quarantines(tmp_path):
    (tmp_path / "owner.json").write_text("{broken",
                                         encoding="utf-8")
    with pytest.raises(OwnerStoreError):
        OwnerRegistry(tmp_path / "owner.json").load()
    assert list(tmp_path.glob("owner.corrupt-*"))


def test_sealed_rebind(tmp_path, monkeypatch):
    import nomorals.core.owner as core_owner

    reg = _registry(tmp_path)
    with pytest.raises(OwnerAuthorityError):
        reg.rebind_with_seal(
            platform="telegram", new_id="99",
            identity_claim="oluwacutyp",
            passphrase="whatever-long-enough")
    seal = core_owner.make_seal("the owner passphrase, long")
    monkeypatch.setattr(core_owner, "PASSPHRASE_SEAL", seal)
    with pytest.raises(OwnerAuthorityError):
        reg.rebind_with_seal(
            platform="telegram", new_id="99",
            identity_claim="oluwacutyp",
            passphrase="the wrong passphrase, long")
    assert reg.rebind_with_seal(
        platform="telegram", new_id="99",
        identity_claim="oluwacutyp",
        passphrase="the owner passphrase, long")
    assert reg.identities("telegram") == ["99"]
    assert reg.audit()[-1]["action"] == "rebind"
    assert "passphrase" not in json.dumps(reg.audit())


def test_impersonation_sensing():
    handles = ["peacethefirt1"]
    assert impersonation_check(
        "someone", "peacethefirt1", owner_handles=handles)
    assert impersonation_check(
        "peacethefirt1", "other", owner_handles=handles)
    assert impersonation_check(
        "Devon", "notdevon", owner_handles=handles)
    assert impersonation_check(
        "Peace The Firt", "peace_fan", owner_handles=handles) \
        == ""
    reg = OwnerRegistry(Path("/nonexistent/owner.json"))
    reg.data = {"identities": {"telegram": ["42"]},
                "handles": handles, "sudo": {}, "deputies": {},
                "mark_text": "", "audit": []}
    assert reg.owner_elsewhere(
        "telegram", "42", "peacethefirt1", "") == ""
    assert reg.owner_elsewhere(
        "telegram", "77", "peacethefirt1", "")


# ── the surface: marks, exemption, flags ─────────────────────

def _surface(tmp_path, registry=None) -> GroupSurface:
    surface = GroupSurface(
        GroupStore(tmp_path / "groups.json").load(),
        owner_registry=registry)
    surface.update_settings(KEY, moderation={
        "flood_max": 6, "flood_window_s": 10.0,
        "repeat_threshold": 3, "link_allowlist": [],
        "word_filters": ["badword"], "filter_action": "delete"})
    return surface


def test_surface_marks_roles_and_exempts_the_ladder(tmp_path):
    reg = _registry(tmp_path)
    reg.bind("owner", via_group=False, platform="telegram",
             user_id="42")
    reg.set_sudo("owner", via_group=False, platform="telegram",
                 user_id="7", on=True)
    surface = _surface(tmp_path, reg)
    out = surface.handle_message(
        KEY, "7", name="Sudo", text="badword post",
        message_id="s1", ts=NOW)
    assert out["actions"] == []               # sudo exempt
    member = surface.store.group(KEY)["members"]["7"]
    assert member["role"] == "sudo"
    out = surface.handle_message(
        KEY, "8", name="Plain", text="badword post",
        message_id="p1", ts=NOW)
    assert [a["kind"] for a in out["actions"]] == ["delete"]
    surface.handle_message(KEY, "42", name="Boss", text="hi",
                           message_id="o1", ts=NOW)
    member = surface.store.group(KEY)["members"]["42"]
    assert member["role"] == "owner"


def test_surface_flags_a_worn_face_once(tmp_path):
    reg = _registry(tmp_path)
    reg.bind("owner", via_group=False, platform="telegram",
             user_id="42")
    reg.seed(handles=["peacethefirt1"])
    surface = _surface(tmp_path, reg)
    for i in range(2):
        surface.handle_message(
            KEY, "99", username="peacethefirt1",
            name="Totally Real", text="hello",
            message_id=f"i{i}", ts=NOW + i)
    member = surface.store.group(KEY)["members"]["99"]
    assert member["impersonation_flag"]
    flags = [a for a in surface.store.group(KEY)["audit"]
             if a.get("kind") == "impersonation_flag"]
    assert len(flags) == 1                     # once, not spam
    surface.handle_message(
        KEY, "42", username="peacethefirt1", name="Boss",
        text="hello", message_id="o9", ts=NOW + 5)
    member = surface.store.group(KEY)["members"]["42"]
    assert "impersonation_flag" not in member  # the owner himself


# ── the cards ────────────────────────────────────────────────

def test_cards_compose_from_seeded_state(tmp_path):
    reg = _registry(tmp_path)
    reg.bind("owner", via_group=False, platform="telegram",
             user_id="42")
    reg.set_sudo("owner", via_group=False, platform="telegram",
                 user_id="7", on=True)
    card = verification_card(reg, seal_configured=False,
                             account_lines=["whatsapp 234… — "
                                            "linked"],
                             sources=["seeded from config"])
    assert "42" in card and "sudo: telegram: 7" in card
    assert "not baked" in card
    text = status_card(uptime_s=90061, version="0.1.0",
                       accounts={"registered": 2,
                                 "connected": 1, "dropped": 1},
                       queues={"channel posts queued": 3},
                       last_briefing_ts=NOW, now=NOW + 3600)
    assert "1d 1h 1m" in text and "0.1.0" in text
    assert "commit not stamped" in text
    assert "2 registered · 1 connected · 1 dropped" in text
    assert "last briefing" in text and "1h 0m" in text
    text = status_card(uptime_s=None, version="0.1.0")
    assert "uptime: unknown" in text
    assert "last briefing: none on record" in text


# ── the runtime handlers (bare mixin rig) ────────────────────

class _Gateway:
    def __init__(self):
        self.files = []

    def send_file(self, platform, chat_key, path, caption=""):
        self.files.append((platform, chat_key, path, caption))
        return SimpleNamespace(ok=True)


def _rig(tmp_path, *, operator_ids=("42",)):
    rig = RuntimeGroupMixin()
    rig.context = SimpleNamespace(
        settings=SimpleNamespace(workspace_dir=str(tmp_path)),
        db=None)
    rig.gateway = _Gateway()
    rig._owner_sender_ids = set(operator_ids)
    rig._is_operator = lambda message: str(
        message.sender_id) in set(operator_ids)
    return rig


def _msg(*, chat_key="telegram:555", platform="telegram",
         kind=ChatKind.DM, sender_id="42",
         username="oluwacutyp", text="", meta=None):
    return SimpleNamespace(
        chat=SimpleNamespace(platform=platform, key=chat_key,
                             chat_id=chat_key.split(":")[-1],
                             kind=kind, title=""),
        sender_id=sender_id, sender="Sender",
        sender_username=username, text=text,
        meta=meta or {}, message_id="m1", reply_to="")


def test_owner_handler_refuses_non_operators_every_verb(
        tmp_path):
    rig = _rig(tmp_path)
    stranger = _msg(sender_id="99", username="mallory")
    for tail in ("", "status", "audit", "bind telegram 1",
                 "sudo add telegram 1", "deputy add g 1",
                 "mark hello", "rebind telegram 1 some pass"):
        out = rig._control_owner(tail, stranger)
        assert "only to the owner" in out


def test_owner_handler_edits_are_dm_only(tmp_path):
    rig = _rig(tmp_path)
    group_msg = _msg(chat_key=KEY, platform="telegram-bot",
                     kind=ChatKind.GROUP)
    out = rig._control_owner("bind telegram 4242", group_msg)
    assert "never editable from a group" in out
    registry = rig._owner_registry()
    assert registry.identities("telegram") == ["42"]  # seed only
    out = rig._control_owner("bind telegram 4242", _msg())
    assert "bound" in out
    assert "4242" in registry.identities("telegram")
    card = rig._control_owner("", _msg())
    assert "4242" in card and "Owner — verification" in card
    status = rig._control_owner("status", _msg())
    assert "Owner — status" in status
    assert "uptime: unknown" in status  # bare rig: no snapshot


def test_webshot_handler_group_gate_and_dm_shot(
        tmp_path, monkeypatch):
    rig = _rig(tmp_path)
    calls = []

    def fake_capture(url, opts, **kw):
        calls.append(url)
        return CaptureResult(data=_png(1440, 900), title="Ex",
                             final_url=url, rung="playwright",
                             width=1440, height=900)

    monkeypatch.setattr(webshot, "capture", fake_capture)
    # URL discipline is pinned at organ level with literal
    # IPs; this sandbox's DNS wildcard-resolves every name
    # into private space, so the handler rig stubs the
    # resolver verdict out — gating/delivery is what's here.
    monkeypatch.setattr(webshot, "_host_is_private",
                        lambda host: False)
    foreign = _msg(chat_key="telegram:-1001",
                   platform="telegram", kind=ChatKind.GROUP,
                   sender_id="8", username="member")
    out = rig._control_webshot("https://example.com", foreign)
    assert "isn't claimed" in out
    assert calls == []                      # gated pre-capture
    dm = _msg(sender_id="8", username="member")
    # .invalid never resolves, so URL discipline cannot call
    # it private on any DNS — the capture rung is faked anyway.
    out = rig._control_webshot("full https://shots.invalid/page",
                               dm)
    assert "via playwright" in out
    assert calls == ["https://shots.invalid/page"]
    assert len(rig.gateway.files) == 1
    platform, chat_key, path, caption = rig.gateway.files[0]
    assert caption.startswith("🌐 Ex")
    assert Path(path).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    book = rig._webshot_book()
    assert book.journal()[-1]["ok"] is True
    assert book.journal()[-1]["rung"] == "playwright"


def test_webshot_handler_reply_targets_the_replied_link(
        tmp_path, monkeypatch):
    rig = _rig(tmp_path)
    seen = {}

    def fake_capture(url, opts, **kw):
        seen["url"] = url
        return CaptureResult(data=_png(10, 10), title="T",
                             final_url=url, rung="playwright",
                             width=10, height=10)

    monkeypatch.setattr(webshot, "capture", fake_capture)
    msg = _msg(meta={"reply_to_text":
                     "check https://news.example/story out"})
    out = rig._control_webshot("phone", msg)
    assert seen["url"] == "https://news.example/story"
    assert "phone" in out


def test_watch_driver_posts_a_change_alert(tmp_path,
                                           monkeypatch):
    rig = _rig(tmp_path)
    book = rig._webshot_book()
    dm = _msg()
    out = rig._control_webshot("watch https://example.com",
                               dm)
    assert "watching" in out
    state = {"png": _png(1440, 900, (255, 255, 255))}

    def fake_capture(url, opts, **kw):
        return CaptureResult(data=state["png"], title="Ex",
                             final_url=url, rung="playwright",
                             width=1440, height=900)

    monkeypatch.setattr(webshot, "capture", fake_capture)
    watch = book.watches()[0]
    watch["next_due"] = 0
    rig._webshot_watch_last = 0.0
    rig._drive_webshot_watches()
    assert rig.gateway.files == []          # baseline: silent
    state["png"] = _png(1440, 900, (0, 0, 0))
    watch["next_due"] = 0
    book.save()
    rig._webshot_watch_last = 0.0
    rig._drive_webshot_watches()
    assert len(rig.gateway.files) == 1
    assert "page changed" in rig.gateway.files[0][3]
