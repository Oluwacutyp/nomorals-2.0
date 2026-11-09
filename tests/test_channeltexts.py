"""/channeltexts — pull a channel's stored texts as a FILE
(owner ask 2026-11-08 01:31 + file delivery 01:34).

Pins, by layer:

* the vault reader/index over the channel kinds: WhatsApp
  sweep rows (text + media, captions kept — G4), Telegram
  archive rows, missed rows, the owner's own posts (full text
  kept — G5's journal half), oldest → newest;
* the Telegram archive writer at the sighting funnel: gated
  on status_saving, deduped across restarts, failure-isolated,
  and the 2026-11-02 posture intact (the post reaches no
  brain, no memory, no reply — the archive is a store the
  owner pulls from, nothing reads it back into the brain);
* the export builder: txt/md/csv shapes, window + latest-cap,
  coverage honesty in the header (stored span next to the
  exported window; the Telegram archive's start named);
* the command end-to-end over the wave-F rig: resolution
  across both platforms, ambiguity answered never guessed,
  empty results carrying the store's real coverage and the
  saving toggle's state in words, --pull backfilling from
  the userbot history verb, the file delivered through
  gateway.send_file and the export journaled;
* the published ring keeps the full text beside its label
  (G5); the sweep asks the adapter for its full 50-post
  window (G2).
"""

from __future__ import annotations

import os
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

from nomorals.social.chat import channeltexts as ct  # noqa: E402
from nomorals.social.chat.base import ChatKind, ChatMessage, ChatRef  # noqa: E402
from nomorals.social.pairing import AccountRecord, PairingRegistry  # noqa: E402
from nomorals.social.personal import PersonalSurfaces, PersonalVault  # noqa: E402

WA_ACC = "whatsapp:2348012345678"
TG_ACC = "telegram:777000"
NL = "120363000111222333@newsletter"      # a WhatsApp channel
TG_CHAN = "-100222333444"                 # a Telegram channel

T0 = 1762000000.0                          # a fixed past instant
DAY = 86400.0


def make_personal(tmp_path, *, wa_saving=True, tg_saving=True):
    registry = PairingRegistry(tmp_path / "paired.json")
    vault = PersonalVault(tmp_path / "personal")
    personal = PersonalSurfaces(registry, vault)
    wa = registry.upsert(AccountRecord(
        id=WA_ACC, platform="whatsapp", identity="2348012345678",
        status="linked", created_at=1.0))
    tg = registry.upsert(AccountRecord(
        id=TG_ACC, platform="telegram", identity="777000",
        status="linked", created_at=1.0))
    registry.set_toggles(wa.id, status_saving=wa_saving)
    registry.set_toggles(tg.id, status_saving=tg_saving)
    return personal, registry, vault, wa, tg


def seed_wa(personal, tmp_path):
    """Two text posts, one captioned media post and one own
    post on the WhatsApp channel, through the real observe
    path with fetcher-shaped inputs."""
    media_file = tmp_path / "chart.png"
    media_file.write_bytes(b"\x89PNG\r\n")
    personal.observe_channel_post(
        "whatsapp", channel_id=NL, channel_name="Gold Signals",
        text="gold update one", source_id="s1", ts=T0)
    personal.observe_channel_post(
        "whatsapp", channel_id=NL, channel_name="Gold Signals",
        text="gold update two", source_id="s2", ts=T0 + DAY)
    personal.observe_channel_post(
        "whatsapp", channel_id=NL, channel_name="Gold Signals",
        text="chart caption here",
        media=[{"path": str(media_file), "kind": "photo"}],
        source_id="s3", ts=T0 + 2 * DAY)
    personal.note_channel_posted(
        "whatsapp", account_id=WA_ACC, channel_id=NL,
        channel_name="Gold Signals",
        text="my own announcement " + "x" * 250,
        source_id="own1", ts=T0 + 3 * DAY)


def seed_tg(personal):
    personal.observe_channel_text(
        "telegram", channel_id=TG_CHAN, channel_name="Gold Signals",
        text="tg post one", source_id="11", ts=T0 + 3600)
    personal.observe_channel_text(
        "telegram", channel_id=TG_CHAN, channel_name="Gold Signals",
        text="", media_kind="photo", source_id="12",
        ts=T0 + DAY + 3600)


# ── the archive writer (store level) ─────────────────────────


def test_tg_archive_writes_row_when_saving_on(tmp_path):
    personal, _r, vault, _wa, tg = make_personal(tmp_path)
    out = personal.observe_channel_text(
        "telegram", channel_id=TG_CHAN, channel_name="Gold Signals",
        text="hello channel", source_id="42", ts=T0)
    assert len(out) == 1
    row = out[0]
    assert row["kind"] == "tg_channel_post"
    assert row["platform"] == "telegram"
    assert row["account"] == tg.id
    assert row["counterpart_id"] == TG_CHAN
    assert row["poster_name"] == "Gold Signals"
    assert row["text"] == "hello channel"
    assert row["media_kind"] == "text"
    assert vault.journal(tg.id, "tg_channel_post") == [row]


def test_tg_archive_gated_on_status_saving(tmp_path):
    personal, _r, vault, _wa, tg = make_personal(
        tmp_path, tg_saving=False)
    out = personal.observe_channel_text(
        "telegram", channel_id=TG_CHAN, channel_name="Gold Signals",
        text="hello channel", source_id="42", ts=T0)
    assert out == []
    assert vault.journal(tg.id, "tg_channel_post") == []


def test_tg_archive_dedupes_across_restarts(tmp_path):
    personal, registry, vault, _wa, _tg = make_personal(tmp_path)
    personal.observe_channel_text(
        "telegram", channel_id=TG_CHAN, channel_name="Gold Signals",
        text="once only", source_id="42", ts=T0)
    # a fresh surface over the same vault: the seen set seeds
    # from the journal, so a redelivery writes nothing
    again = PersonalSurfaces(registry, vault)
    out = again.observe_channel_text(
        "telegram", channel_id=TG_CHAN, channel_name="Gold Signals",
        text="once only", source_id="42", ts=T0)
    assert out == []
    assert len(vault.journal(kind="tg_channel_post")) == 1


def test_tg_archive_media_only_post_keeps_a_row(tmp_path):
    personal, _r, _v, _wa, _tg = make_personal(tmp_path)
    out = personal.observe_channel_text(
        "telegram", channel_id=TG_CHAN, channel_name="Gold Signals",
        text="", media_kind="video", source_id="7", ts=T0)
    assert len(out) == 1
    assert out[0]["media_kind"] == "video"
    # a post with neither text nor media is not a post at all
    assert personal.observe_channel_text(
        "telegram", channel_id=TG_CHAN, channel_name="Gold Signals",
        text="", source_id="8", ts=T0) == []


def test_wa_media_post_keeps_its_caption(tmp_path):
    personal, _r, vault, wa, _tg = make_personal(tmp_path)
    media_file = tmp_path / "clip.mp4"
    media_file.write_bytes(b"\x00\x00")
    personal.observe_channel_post(
        "whatsapp", channel_id=NL, channel_name="Gold Signals",
        text="the caption is text too",
        media=[{"path": str(media_file), "kind": "video"}],
        source_id="m1", ts=T0)
    rows = [r for r in vault.journal(wa.id, "channel_post")
            if r.get("source_id") == "m1"]
    assert len(rows) == 1
    assert rows[0]["text"] == "the caption is text too"
    assert rows[0]["media_kind"] == "video"


def test_own_post_journal_keeps_full_text(tmp_path):
    personal, _r, vault, wa, _tg = make_personal(tmp_path)
    long_text = "announcement " + "y" * 400
    personal.note_channel_posted(
        "whatsapp", account_id=wa.id, channel_id=NL,
        channel_name="Gold Signals", text=long_text,
        source_id="o1", ts=T0)
    rows = vault.journal(wa.id, "channel_posted")
    assert rows[0]["text"] == long_text


def test_toggle_activity_counts_channel_saves(tmp_path):
    personal, _r, _v, _wa, tg = make_personal(tmp_path)
    assert personal.toggle_activity(tg.id).get(
        "status_saving") is None
    personal.observe_channel_text(
        "telegram", channel_id=TG_CHAN, channel_name="Gold Signals",
        text="proof of life", source_id="1", ts=T0)
    assert personal.toggle_activity(tg.id)["status_saving"] == T0


# ── the vault reader / index ────────────────────────────────


def test_channel_index_and_reader(tmp_path):
    personal, _r, vault, _wa, _tg = make_personal(tmp_path)
    seed_wa(personal, tmp_path)
    seed_tg(personal)
    index = {(s["platform"], s["channel_id"]): s
             for s in vault.channel_index()}
    wa_slot = index[("whatsapp", NL)]
    assert wa_slot["channel_name"] == "Gold Signals"
    assert wa_slot["text"] == 2 and wa_slot["media"] == 1
    assert wa_slot["own"] == 1 and wa_slot["missed"] == 0
    assert wa_slot["first_ts"] == T0
    assert wa_slot["last_ts"] == T0 + 3 * DAY
    tg_slot = index[("telegram", TG_CHAN)]
    assert tg_slot["text"] == 1 and tg_slot["media"] == 1

    rows = vault.channel_texts(platform="whatsapp",
                               channel_id=NL)
    assert [r["source_id"] for r in rows] == \
        ["s1", "s2", "s3", "own1"]          # oldest → newest
    texts = [r.get("text") for r in rows]
    assert texts[2] == "chart caption here"
    assert texts[3].startswith("my own announcement")


def test_reader_matches_own_post_by_name(tmp_path):
    """An own Telegram post journaled under its @name ref is
    still the channel's row once the channel is resolved."""
    personal, _r, vault, _wa, tg = make_personal(tmp_path)
    seed_tg(personal)
    personal.note_channel_posted(
        "telegram", account_id=tg.id, channel_id="@goldsignals",
        channel_name="Gold Signals", text="posted from the queue",
        source_id="99", ts=T0 + 2 * DAY)
    rows = vault.channel_texts(platform="telegram",
                               channel_id=TG_CHAN,
                               channel_name="Gold Signals")
    assert [r["source_id"] for r in rows] == ["11", "12", "99"]
    assert rows[-1]["text"] == "posted from the queue"


# ── the export builder (pure) ───────────────────────────────


def _rows(tmp_path):
    personal, _r, vault, _wa, _tg = make_personal(tmp_path)
    seed_wa(personal, tmp_path)
    return vault.channel_texts(platform="whatsapp", channel_id=NL)


def test_parse_tail_forms():
    query, opts, err = ct.parse_tail("Gold Signals --days 7")
    assert (query, err) == ("Gold Signals", "")
    assert opts["days"] == 7 and opts["since"] > 0
    query, opts, err = ct.parse_tail("--format md Gold Signals")
    assert query == "Gold Signals" and opts["format"] == "md"
    query, opts, err = ct.parse_tail("Gold Signals --pull")
    assert opts["pull"] == ct.DEFAULT_PULL
    query, opts, err = ct.parse_tail("Gold Signals --pull 25")
    assert opts["pull"] == 25
    _q, _o, err = ct.parse_tail("Gold Signals --format pdf")
    assert "txt, md or csv" in err
    _q, _o, err = ct.parse_tail("Gold Signals --bogus 1")
    assert "unknown option" in err
    _q, _o, err = ct.parse_tail("Gold Signals --days nope")
    assert "--days" in err


def test_render_txt_shape_and_coverage(tmp_path):
    rows = _rows(tmp_path)
    stored = ct.coverage(rows)
    body = ct.render_export(
        rows, channel_name="Gold Signals", platform="whatsapp",
        channel_id=NL, fmt="txt", stored=stored,
        opts={"since": 0.0, "until": 0.0, "limit": 1000},
        exported_ts=T0 + 4 * DAY)
    assert "Channel texts — Gold Signals" in body
    assert "Stored coverage: 4 stored post(s)" in body
    assert "gold update one" in body
    assert "chart caption here" in body
    assert "[photo — saved file:" in body
    assert "You (posted via Devon)" in body
    assert "my own announcement" in body


def test_render_md_and_csv(tmp_path):
    rows = _rows(tmp_path)
    stored = ct.coverage(rows)
    opts = {"since": 0.0, "until": 0.0, "limit": 1000}
    md = ct.render_export(
        rows, channel_name="Gold Signals", platform="whatsapp",
        channel_id=NL, fmt="md", stored=stored, opts=opts,
        exported_ts=T0)
    assert md.startswith("# Channel texts — Gold Signals")
    assert "## " in md and "gold update two" in md
    csv_body = ct.render_export(
        rows, channel_name="Gold Signals", platform="whatsapp",
        channel_id=NL, fmt="csv", stored=stored, opts=opts,
        exported_ts=T0)
    lines = csv_body.strip().splitlines()
    assert lines[0].startswith("ts,datetime,sender,kind")
    assert len(lines) == 5
    assert any("chart caption here" in ln for ln in lines)


def test_render_tg_header_names_archive_start(tmp_path):
    personal, _r, vault, _wa, _tg = make_personal(tmp_path)
    seed_tg(personal)
    rows = vault.channel_texts(platform="telegram",
                               channel_id=TG_CHAN)
    body = ct.render_export(
        rows, channel_name="Gold Signals", platform="telegram",
        channel_id=TG_CHAN, fmt="txt", stored=ct.coverage(rows),
        opts={"since": 0.0, "until": 0.0, "limit": 1000},
        exported_ts=T0 + 9 * DAY)
    assert "was not stored" in body
    assert "before this archive" in body
    assert "tg post one" in body
    assert "[photo — the post's media itself is not in the" \
        " archive]" in body


def test_filter_window_and_latest_cap(tmp_path):
    rows = _rows(tmp_path)
    windowed = ct.filter_rows(
        rows, {"since": T0 + DAY, "until": 0.0, "limit": 1000})
    assert [r["source_id"] for r in windowed] == \
        ["s2", "s3", "own1"]
    capped = ct.filter_rows(
        rows, {"since": 0.0, "until": 0.0, "limit": 2})
    assert [r["source_id"] for r in capped] == ["s3", "own1"]


# ── the sighting funnel (full runtime, posture rig shape) ───


def _posture_rig(tmp_path):
    from nomorals.agents.context import build_context
    from nomorals.agents.partner_runtime import PartnerRuntime
    from nomorals.core.config import load_settings

    settings = load_settings(overrides={
        "home": str(tmp_path),
        "partner.platforms": "local",
        "chat.local_enabled": "true",
    })
    context = build_context(settings, with_executor=False,
                            with_tools=False)

    class _Router:
        def chat(self, *a, **k):
            raise AssertionError("the brain must not be called "
                                 "for a channel post")

    context.router = _Router()
    sends: list = []

    class _Gateway:
        adapters: dict = {}
        dry_run = False

        def status(self):
            return {"local": {"running_in_session": True}}

        def set_rate_limit(self, *a, **k):
            return None

        def typing(self, *a, **k):
            return True

        def send(self, platform, chat, text, **kw):
            sends.append(text)
            return SimpleNamespace(ok=True)

    runtime = PartnerRuntime(context, gateway=_Gateway())
    return runtime, sends


def _tg_channel_msg(text="big announcement everyone", mid="42"):
    return ChatMessage(
        chat=ChatRef(platform="telegram", chat_id=TG_CHAN,
                     kind=ChatKind.CHANNEL, title="Gold Signals"),
        incoming=True, text=text, sender="channel",
        sender_id="", message_id=mid, ts=T0)


def test_funnel_archives_tg_post_and_posture_holds(tmp_path):
    runtime, sends = _posture_rig(tmp_path)
    svc = runtime._pairing_personal()
    registry, personal = svc["registry"], svc["personal"]
    rec = registry.upsert(AccountRecord(
        id=TG_ACC, platform="telegram", identity="777000",
        status="linked", created_at=1.0))
    registry.set_toggles(rec.id, status_saving=True)
    runtime._process_inner(_tg_channel_msg())
    assert sends == []                      # silence, as ever
    rows = personal.vault.journal(kind="tg_channel_post")
    assert len(rows) == 1
    assert rows[0]["text"] == "big announcement everyone"
    assert rows[0]["counterpart_id"] == TG_CHAN
    assert rows[0]["poster_name"] == "Gold Signals"
    assert rows[0]["account"] == TG_ACC
    # and the post fed nothing else in the vault
    assert {r["kind"] for r in personal.vault.journal()} == \
        {"tg_channel_post"}


def test_funnel_archives_nothing_when_saving_off(tmp_path):
    runtime, sends = _posture_rig(tmp_path)
    svc = runtime._pairing_personal()
    svc["registry"].upsert(AccountRecord(
        id=TG_ACC, platform="telegram", identity="777000",
        status="linked", created_at=1.0))
    # status_saving stays off (the default)
    runtime._process_inner(_tg_channel_msg())
    assert sends == []
    assert svc["personal"].vault.journal(
        kind="tg_channel_post") == []


def test_funnel_archive_failure_never_breaks_the_funnel(tmp_path):
    runtime, sends = _posture_rig(tmp_path)
    svc = runtime._pairing_personal()
    rec = svc["registry"].upsert(AccountRecord(
        id=TG_ACC, platform="telegram", identity="777000",
        status="linked", created_at=1.0))
    svc["registry"].set_toggles(rec.id, status_saving=True)

    def _boom(*a, **k):
        raise RuntimeError("vault on fire")

    svc["personal"].observe_channel_text = _boom
    runtime._process_inner(_tg_channel_msg())   # must not raise
    assert sends == []


# ── the command (wave-F rig shape) ──────────────────────────


class _FakeWA:
    def __init__(self, channels=()):
        self._channels = list(channels)

    def chats(self, limit=500):
        return list(self._channels)


class _FakeTG:
    def __init__(self, history_rows=()):
        self._history = list(history_rows)

    def history(self, chat, limit=20):
        return list(self._history)[:limit]


def _runtime(tmp_path, *, wa=None, tg=None, saving=True):
    from nomorals.agents.partner.runtime_chatops import RuntimeChatOpsMixin
    from nomorals.agents.partner.runtime_provision import RuntimeProvisionMixin

    registry = PairingRegistry(tmp_path / "paired.json")
    vault = PersonalVault(tmp_path / "personal")
    personal = PersonalSurfaces(registry, vault)
    wa_rec = registry.upsert(AccountRecord(
        id=WA_ACC, platform="whatsapp", identity="2348012345678",
        status="linked", created_at=1.0))
    tg_rec = registry.upsert(AccountRecord(
        id=TG_ACC, platform="telegram", identity="777000",
        status="linked", created_at=1.0))
    registry.set_toggles(wa_rec.id, status_saving=saving)
    registry.set_toggles(tg_rec.id, status_saving=saving)
    files: list = []

    def fake_send_file(platform, chat, path, caption="",
                       name=""):
        with open(path, encoding="utf-8") as fh:
            body = fh.read()
        files.append({"platform": platform, "path": path,
                      "caption": caption, "name": name,
                      "body": body})
        return SimpleNamespace(ok=True, error="")

    class _RT(RuntimeChatOpsMixin, RuntimeProvisionMixin):
        pass

    rt = object.__new__(_RT)
    rt.context = SimpleNamespace(
        db=None,
        settings=SimpleNamespace(workspace_dir=str(tmp_path)))
    rt.gateway = SimpleNamespace(
        adapters={"whatsapp": wa if wa is not None else _FakeWA(),
                  "telegram": tg if tg is not None else _FakeTG()},
        send_file=fake_send_file)
    rt._pair_personal_svc = {"registry": registry,
                             "personal": personal}
    return rt, personal, vault, files


DM = ChatRef(platform="telegram", chat_id="777",
             kind=ChatKind.DM, title="Owner", peer="777")


def _cmd(rt, tail):
    msg = ChatMessage(chat=DM, incoming=True, text=tail,
                      sender="Owner", sender_id="777",
                      message_id="c1")
    return rt._control_chatops("channeltexts", tail, msg)


def test_command_exports_wa_channel_as_file(tmp_path):
    rt, personal, vault, files = _runtime(tmp_path)
    seed_wa(personal, tmp_path)
    out = _cmd(rt, "Gold Signals")
    assert len(files) == 1
    assert files[0]["name"] == "gold-signals-texts.txt"
    body = files[0]["body"]
    assert "gold update one" in body
    assert "gold update two" in body
    assert "chart caption here" in body
    assert "my own announcement" in body
    assert "4 post(s) in gold-signals-texts.txt" in out
    assert "Stored coverage: 4 stored post(s)" in out
    # the export itself is journaled
    kinds = [r["kind"] for r in vault.journal()]
    assert "channel_texts_exported" in kinds


def test_command_ambiguity_is_answered_never_guessed(tmp_path):
    rt, personal, _v, files = _runtime(tmp_path)
    seed_wa(personal, tmp_path)
    seed_tg(personal)
    out = _cmd(rt, "Gold Signals")
    assert "more than one channel answers" in out
    assert "WhatsApp" in out and "Telegram" in out
    assert files == []
    out = _cmd(rt, "tg:Gold Signals")
    assert len(files) == 1
    assert "tg post one" in files[0]["body"]
    out = _cmd(rt, "wa:Gold Signals")
    assert len(files) == 2
    assert "gold update one" in files[1]["body"]


def test_command_resolves_jid_directly(tmp_path):
    rt, personal, _v, files = _runtime(tmp_path)
    seed_wa(personal, tmp_path)
    _cmd(rt, NL)
    assert len(files) == 1
    assert "gold update two" in files[0]["body"]


def test_command_unknown_channel_lists_stored(tmp_path):
    rt, personal, _v, files = _runtime(tmp_path)
    seed_wa(personal, tmp_path)
    out = _cmd(rt, "No Such Place")
    assert "no channel here answers" in out
    assert "Gold Signals" in out
    assert files == []


def test_command_empty_window_reports_coverage(tmp_path):
    rt, personal, _v, files = _runtime(tmp_path)
    seed_wa(personal, tmp_path)
    out = _cmd(rt, "Gold Signals --days 0 --since 2030-01-01")
    assert "nothing in that window" in out
    assert "4 stored post(s)" in out
    assert files == []


def test_command_saving_off_names_the_toggle(tmp_path):
    rt, _p, _v, files = _runtime(tmp_path, saving=False)
    out = _cmd(rt, NL)
    assert "status_saving is off" in out
    assert WA_ACC in out
    assert files == []
    out = _cmd(rt, TG_CHAN)
    assert "status_saving is off" in out
    assert TG_ACC in out


def test_command_limit_notes_truncation(tmp_path):
    rt, personal, _v, files = _runtime(tmp_path)
    seed_wa(personal, tmp_path)
    out = _cmd(rt, "Gold Signals --limit 2")
    body = files[0]["body"]
    assert "gold update one" not in body
    assert "chart caption here" in body
    assert "latest 2 of 4" in out


def test_command_formats(tmp_path):
    rt, personal, _v, files = _runtime(tmp_path)
    seed_wa(personal, tmp_path)
    _cmd(rt, "Gold Signals --format md")
    assert files[0]["name"].endswith(".md")
    assert files[0]["body"].startswith("# Channel texts")
    _cmd(rt, "Gold Signals --format csv")
    assert files[1]["name"].endswith(".csv")
    assert files[1]["body"].startswith("ts,datetime")


def test_command_pull_backfills_from_history(tmp_path):
    history = [
        ChatMessage(
            chat=ChatRef(platform="telegram", chat_id=TG_CHAN,
                         kind=ChatKind.CHANNEL,
                         title="Gold Signals"),
            incoming=True, text="an older post", sender="channel",
            message_id="5", ts=T0 - DAY),
        ChatMessage(
            chat=ChatRef(platform="telegram", chat_id=TG_CHAN,
                         kind=ChatKind.CHANNEL,
                         title="Gold Signals"),
            incoming=True, text="a newer old post",
            sender="channel", message_id="6", ts=T0 - 3600),
    ]
    rt, personal, vault, files = _runtime(
        tmp_path, tg=_FakeTG(history))
    out = _cmd(rt, f"{TG_CHAN} --pull")
    assert "Pulled 2 more post(s)" in out
    body = files[0]["body"]
    assert "an older post" in body and "a newer old post" in body
    assert len(vault.journal(kind="tg_channel_post")) == 2
    # a second pull writes nothing twice (dedupe by source id)
    out = _cmd(rt, f"{TG_CHAN} --pull")
    assert "Pulled 0 more post(s)" in out
    assert len(vault.journal(kind="tg_channel_post")) == 2


def test_command_pull_on_whatsapp_names_the_sweep(tmp_path):
    rt, personal, _v, files = _runtime(tmp_path)
    seed_wa(personal, tmp_path)
    out = _cmd(rt, "Gold Signals --pull")
    assert len(files) == 1
    assert "saving sweep" in out


# ── G5: the published ring keeps the full text ──────────────


def test_published_ring_keeps_full_text_beside_label(tmp_path):
    from nomorals.social.chat.group_surface import GroupStore, GroupSurface

    surface = GroupSurface(
        GroupStore(tmp_path / "g.json").load())
    key = "telegram-bot:-100222"
    surface.note_channel_seen(key, title="Gold Signals")
    surface.note_bot_admin(key, True)
    assert surface.claim_state(key)["claimed"]
    long_text = "full announcement body " + "z" * 200
    surface.channel_enqueue(key, {
        "kind": "once", "text": long_text,
        "due": time.time() - 5000, "source": "dm"})
    sent = []

    def _send(text, entry):
        sent.append(text)
        return True, "m1"

    posts = surface.channel_cycle(key, now=time.time(),
                                  send=_send, force=True)
    assert len(posts) == 1
    record = surface.channel_posts(key)[0]
    assert record["text"] == sent[0]
    assert long_text in record["text"]
    assert record["label"] == long_text[:60]


# ── G2: the sweep asks for the adapter's full window ────────


def test_channel_fetcher_requests_full_depth(tmp_path):
    from nomorals.agents.partner.runtime_provision import RuntimeProvisionMixin

    seen = {}

    class _Adapter:
        def fetch_channel_posts(self, channels=None, *, count=10):
            seen["count"] = count
            return []

    class _RT(RuntimeProvisionMixin):
        pass

    rt = object.__new__(_RT)
    rt.gateway = SimpleNamespace(adapters={"whatsapp": _Adapter()})
    # the fetcher closure lives inside _pairing_personal; drive
    # it through a built bundle instead of re-implementing it
    rt.context = SimpleNamespace(
        settings=SimpleNamespace(
            workspace_dir=str(tmp_path),
            chat=SimpleNamespace(telegram_api_id="",
                                 telegram_api_hash="")))
    svc = rt._pairing_personal()
    out = svc["personal"].channel_fetcher(
        SimpleNamespace(id=WA_ACC))
    assert out == []
    assert seen["count"] == 50


# ── catalog (the wave-F parity suites pin the rest) ─────────


def test_catalog_registration():
    from nomorals.social.chat import control
    from nomorals.social.chat.chatops import CHATOPS_COMMANDS

    assert "channeltexts" in CHATOPS_COMMANDS
    assert "channeltexts" in control.CONTROL_COMMANDS
    assert "channeltexts" in control.COMMAND_DETAILS
    assert "channeltexts" in control.LIST_ONELINERS
    assert "channeltexts" not in control.PUBLIC_COMMANDS
