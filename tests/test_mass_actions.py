"""P4c §14 — the mass tools: massreact / pollvote / chfollow.

Discipline under test (the owner's warn-first rule, built in):
dry-run is default for a tool's first run and touches NOTHING
(fake actors assert zero platform calls), a live run refuses
until the tool is ARMED (and the refusal is journaled with its
account list), pacing has a floor the owner can raise but never
lower, one reaction per account with a hard count cap, stop-on-
error marks the rest skipped, vote options resolve by number
(no fetch) and by text (fetched once), chfollow is idempotent,
and every run lands in the journal complete.  WhatsApp poll
voting is the named gap — refusal names it, never fakes it.
"""
from __future__ import annotations

import json
import socket
import threading
from types import SimpleNamespace

import pytest

from nomorals.social.mass_actions import (
    TG_CAPABILITIES, WA_CAPABILITIES, WA_VOTE_GAP, WARN_TEXT,
    AlreadyFollowing, MassAccount, MassArms, MassJournal,
    MassRefusal, MassRunner, MassTarget, TelegramSessionActor,
    WhatsAppBridgeActor, bridge_request, parse_target,
    render_run, resolve_option)
from nomorals.social.pairing import AccountRecord


# ── fakes ───────────────────────────────────────────────────────────

class FakeClock:
    def __init__(self) -> None:
        self.t = 1000.0
        self.sleeps: list[float] = []

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.t += seconds


class FakeActor:
    """A platform actor with scripted behaviour + a call log."""

    def __init__(self, *, fail: str = "", options=None,
                 already: bool = False) -> None:
        self.calls: list[tuple] = []
        self._fail = fail
        self._options = options or ["Yes", "No", "Maybe"]
        self._already = already

    def _maybe_fail(self) -> None:
        if self._fail:
            raise RuntimeError(self._fail)

    def react(self, target, emoji):
        self.calls.append(("react", target.peer, target.message_id,
                           emoji))
        self._maybe_fail()
        return {"ok": True}

    def poll_options(self, target):
        self.calls.append(("poll_options",))
        return list(self._options)

    def vote(self, target, option_index):
        self.calls.append(("vote", option_index))
        self._maybe_fail()
        return {"ok": True}

    def follow(self, target):
        self.calls.append(("follow", target.peer))
        if self._already:
            raise AlreadyFollowing("already participant")
        self._maybe_fail()
        return {"ok": True}


def make_pool(n: int, *, platform: str = "telegram",
              caps=None, actor_kwargs=None) -> list[MassAccount]:
    caps = caps if caps is not None else (
        TG_CAPABILITIES if platform == "telegram"
        else WA_CAPABILITIES)
    out = []
    for i in range(n):
        rec = AccountRecord(
            id=f"{platform}:acc{i}", platform=platform,
            identity=f"acc{i}", status="linked", created_at=float(i))
        actor = FakeActor(**(actor_kwargs or {}))
        out.append(MassAccount(record=rec, capabilities=caps,
                               factory=lambda a=actor: a))
        out[-1].fake = actor  # type: ignore[attr-defined]
    return out


def make_svc(tmp_path, *, floor: float = 2.0):
    arms = MassArms(tmp_path / "arms.json")
    journal = MassJournal(tmp_path / "journal.jsonl")
    clock = FakeClock()
    runner = MassRunner(arms, journal, clock=clock.now,
                        sleep=clock.sleep, jitter_s=0.0,
                        pacing_floor=floor)
    return arms, journal, runner, clock


TG_POST = MassTarget(platform="telegram", raw="https://t.me/chan/9",
                     peer="chan", message_id=9)
TG_POLL = MassTarget(platform="telegram",
                     raw="https://t.me/chan/10", peer="chan",
                     message_id=10)
TG_CHAN = MassTarget(platform="telegram", raw="https://t.me/chan",
                     peer="chan")
WA_POST = MassTarget(platform="whatsapp",
                     raw="https://whatsapp.com/channel/CODE/77",
                     peer="CODE", server_id="77")


# ── link parsing ────────────────────────────────────────────────────

def test_parse_telegram_links():
    t, ride = parse_target("https://t.me/somechannel/482")
    assert (t.platform, t.peer, t.message_id, ride) == (
        "telegram", "somechannel", 482, 0)
    t, _ = parse_target("https://t.me/s/somechannel/482")
    assert (t.peer, t.message_id) == ("somechannel", 482)
    t, _ = parse_target("https://t.me/c/1234567/89")
    assert (t.peer_id, t.message_id) == (-1001234567, 89)
    t, _ = parse_target("https://t.me/c/1234567/4/89")
    assert t.message_id == 89  # topic form: last segment wins
    t, _ = parse_target("https://t.me/somechannel")
    assert t.peer == "somechannel" and t.message_id == 0


def test_parse_whatsapp_channel_link_and_riding_count():
    t, ride = parse_target(
        "https://whatsapp.com/channel/0029VaXyZ/175/12")
    assert (t.platform, t.peer, t.server_id, ride) == (
        "whatsapp", "0029VaXyZ", "175", 12)
    t, ride = parse_target("https://whatsapp.com/channel/0029VaXyZ")
    assert t.server_id == "" and ride == 0
    t, _ = parse_target("120363040123@newsletter")
    assert t.peer == "120363040123@newsletter"


def test_parse_refuses_unknown_links():
    for bad in ("https://example.com/x", "hello", "",
                "https://t.me/"):
        with pytest.raises(MassRefusal):
            parse_target(bad)


def test_resolve_option_by_number_text_and_refusals():
    opts = ["Yes", "No", "Maybe later"]
    assert resolve_option(opts, "2") == 1
    assert resolve_option(opts, "no") == 1       # exact, folded
    assert resolve_option(opts, "maybe") == 2    # unique substring
    with pytest.raises(MassRefusal):
        resolve_option(opts, "9")                # out of range
    with pytest.raises(MassRefusal):
        resolve_option(["Yes we can", "Yes sir"], "yes")  # ambiguous
    with pytest.raises(MassRefusal):
        resolve_option(opts, "banana")


# ── dry-run: zero platform calls, exact plan ────────────────────────

def test_dry_run_touches_nothing_and_plans_exactly(tmp_path):
    arms, journal, runner, clock = make_svc(tmp_path)
    pool = make_pool(3)
    run = runner.run("massreact", TG_POST, accounts=pool,
                     emoji="🔥", dry_run=True)
    assert run.dry_run is True
    assert all(h.fake.calls == [] for h in pool)  # zero calls
    assert [e["account"] for e in run.entries] == [
        "telegram:acc0", "telegram:acc1", "telegram:acc2"]
    # schedule: first at once, then one pacing gap each — the
    # arms store's own 20s floor governs the plan (a runner-
    # level override can never plan tighter than the floor)
    offsets = [e["offset_s"] for e in run.entries]
    assert offsets == [0.0, 20.0, 40.0]
    assert all(e["detail"].startswith("would react")
               for e in run.entries)
    assert clock.sleeps == []  # a dry run never actually waits
    text = render_run(run)
    assert "DRY RUN" in text and "🔥" in text
    # and it journaled as a plan, kind=run with dry flag
    rows = journal.tail(5)
    assert rows and rows[-1]["dry_run"] is True
    assert rows[-1]["tool"] == "massreact"


def test_first_run_defaults_to_dry_until_first_live(tmp_path):
    # The default stays dry until the owner has taken the tool
    # live once explicitly; after that first live run, plain
    # runs go live on their own.
    arms, journal, runner, _ = make_svc(tmp_path)
    arms.arm("massreact")
    pool = make_pool(2)
    first = runner.run("massreact", TG_POST, accounts=pool,
                       emoji="🔥")
    assert first.dry_run is True
    assert all(h.fake.calls == [] for h in pool)
    second = runner.run("massreact", TG_POST, accounts=pool,
                        emoji="🔥")
    assert second.dry_run is True  # still no live run on record
    third = runner.run("massreact", TG_POST, accounts=pool,
                       emoji="🔥", dry_run=False)
    assert third.dry_run is False
    assert all(len(h.fake.calls) == 1 for h in pool)
    assert arms.state("massreact")["live_runs"] == 1
    fourth = runner.run("massreact", TG_POST, accounts=pool,
                        emoji="🔥")
    assert fourth.dry_run is False  # the default has flipped


# ── the arm gate ────────────────────────────────────────────────────

def test_unarmed_live_run_refuses_and_journals(tmp_path):
    arms, journal, runner, _ = make_svc(tmp_path)
    pool = make_pool(2)
    with pytest.raises(MassRefusal) as err:
        runner.run("massreact", TG_POST, accounts=pool, emoji="🔥",
                   dry_run=False)
    assert "not armed" in str(err.value)
    assert "flag" in str(err.value)  # the plain risk warning
    assert all(h.fake.calls == [] for h in pool)
    rows = journal.tail(5, tool="massreact")
    assert rows and rows[-1]["kind"] == "run_refused"
    assert rows[-1]["note"]
    assert [a["account"] for a in rows[-1]["accounts"]] == [
        "telegram:acc0", "telegram:acc1"]  # full account list


def test_arm_warn_text_is_the_plain_warning():
    assert "may flag, restrict, or ban" in WARN_TEXT
    assert "never the bot" in WARN_TEXT


def test_disarm_revokes(tmp_path):
    arms, journal, runner, _ = make_svc(tmp_path)
    arms.arm("chfollow")
    arms.disarm("chfollow")
    pool = make_pool(1)
    with pytest.raises(MassRefusal):
        runner.run("chfollow", TG_CHAN, accounts=pool, dry_run=False)


# ── pacing floor ────────────────────────────────────────────────────

def test_pacing_floor_cannot_be_lowered(tmp_path):
    arms = MassArms(tmp_path / "arms.json")
    with pytest.raises(MassRefusal):
        arms.set_pacing("massreact", 1.0)  # below the 20s floor
    assert arms.set_pacing("massreact", 45.0) == 45.0
    assert arms.pacing("massreact") == 45.0


def test_executed_gaps_respect_the_floor(tmp_path):
    arms, journal, runner, clock = make_svc(tmp_path, floor=3.0)
    arms.arm("massreact")
    runner.run("massreact", TG_POST, accounts=make_pool(1),
               emoji="🔥")  # first run: dry
    pool = make_pool(3)
    run = runner.run("massreact", TG_POST, accounts=pool,
                     emoji="🔥", dry_run=False)
    assert run.dry_run is False
    ats = [e["at"] for e in run.entries]
    assert all(b - a >= 3.0 for a, b in zip(ats, ats[1:]))
    assert clock.sleeps and all(s >= 3.0 for s in clock.sleeps)


# ── one per account + count cap + stop-on-error ─────────────────────

def test_one_reaction_per_account_and_count_cap(tmp_path):
    arms, journal, runner, _ = make_svc(tmp_path)
    arms.arm("massreact")
    pool = make_pool(5)
    run = runner.run("massreact", TG_POST, accounts=pool,
                     emoji="❤️", count=2, dry_run=False)
    assert len(run.entries) == 2
    assert len({e["account"] for e in run.entries}) == 2
    assert sum(len(h.fake.calls) for h in pool) == 2
    # count beyond the pool clamps to the pool
    run = runner.run("massreact", TG_POST, accounts=pool,
                     emoji="❤️", count=99, dry_run=False)
    assert len(run.entries) == 5


def test_stop_on_error_marks_the_rest_skipped(tmp_path):
    arms, journal, runner, _ = make_svc(tmp_path)
    arms.arm("massreact")
    pool = make_pool(3)
    pool[1].fake._fail = "message not found"
    run = runner.run("massreact", TG_POST, accounts=pool,
                     emoji="🔥", dry_run=False)
    assert [e["ok"] for e in run.entries] == [True, False, False]
    assert "message not found" in run.entries[1]["error"]
    assert "skipped" in run.entries[2]["detail"]
    assert pool[2].fake.calls == []  # never attempted


# ── voting ──────────────────────────────────────────────────────────

def test_vote_by_number_never_fetches_options(tmp_path):
    arms, journal, runner, _ = make_svc(tmp_path)
    arms.arm("pollvote")
    pool = make_pool(2)
    runner.run("pollvote", TG_POLL, accounts=pool,
               option_spec="2", dry_run=False)
    for h in pool:
        assert h.fake.calls == [("vote", 1)]  # no poll_options call


def test_vote_by_text_fetches_options_once(tmp_path):
    arms, journal, runner, _ = make_svc(tmp_path)
    arms.arm("pollvote")
    pool = make_pool(3)
    runner.run("pollvote", TG_POLL, accounts=pool,
               option_spec="maybe", dry_run=False)
    fetches = sum(1 for h in pool for c in h.fake.calls
                  if c[0] == "poll_options")
    assert fetches == 1  # resolved once, cast by all
    assert all(("vote", 2) in h.fake.calls for h in pool)


def test_whatsapp_pollvote_refusal_names_the_gap(tmp_path):
    arms, journal, runner, _ = make_svc(tmp_path)
    arms.arm("pollvote")
    pool = make_pool(2, platform="whatsapp")
    with pytest.raises(MassRefusal) as err:
        runner.run("pollvote", WA_POST, accounts=pool,
                   option_spec="1", dry_run=False)
    assert "no Baileys verb" in str(err.value)
    assert WA_VOTE_GAP in str(err.value)


# ── chfollow idempotence ────────────────────────────────────────────

def test_chfollow_already_following_counts_as_done(tmp_path):
    arms, journal, runner, _ = make_svc(tmp_path)
    arms.arm("chfollow")
    pool = make_pool(2)
    for h in pool:
        h.fake._already = True
    run = runner.run("chfollow", TG_CHAN, accounts=pool,
                     dry_run=False)
    assert all(e["ok"] for e in run.entries)
    assert all("already" in e["detail"] for e in run.entries)
    # second run: same outcome, no error — idempotent
    run2 = runner.run("chfollow", TG_CHAN, accounts=pool,
                      dry_run=False)
    assert all(e["ok"] for e in run2.entries)


# ── journal completeness ────────────────────────────────────────────

def test_journal_record_is_complete(tmp_path):
    arms, journal, runner, _ = make_svc(tmp_path)
    arms.arm("massreact")
    pool = make_pool(2)
    runner.run("massreact", TG_POST, accounts=pool, emoji="🔥",
               dry_run=False)
    rows = journal.tail(5, tool="massreact")
    rec = rows[-1]
    assert rec["kind"] == "run" and rec["dry_run"] is False
    assert rec["target"] == TG_POST.raw
    assert rec["started_at"] and rec["finished_at"]
    assert len(rec["accounts"]) == 2
    for entry in rec["accounts"]:
        assert entry["account"].startswith("telegram:")
        assert entry["at"] > 0 and entry["ok"] is True


def test_arms_store_corruption_quarantines(tmp_path):
    path = tmp_path / "arms.json"
    path.write_text("{broken", encoding="utf-8")
    from nomorals.social.mass_actions import MassStoreError
    with pytest.raises(MassStoreError):
        MassArms(path)
    assert list(tmp_path.glob("arms.corrupt-*.json"))


# ── session actor (fake Telethon-shaped client) ─────────────────────

class FakeMassClient:
    """The async client shape TelegramSessionActor drives (the
    _TelethonMassClient interface): connect/disconnect, verbs
    taking the MassTarget, polls as {text, option-bytes}."""

    def __init__(self, *, answers=("Yes", "No")) -> None:
        self.calls: list[tuple] = []
        self._answers = [{"text": t, "option": t.encode()}
                         for t in answers]

    async def connect(self):
        self.calls.append(("connect",))

    async def disconnect(self):
        self.calls.append(("disconnect",))

    async def react(self, target, emoji):
        self.calls.append(("react", target.peer,
                           target.message_id, emoji))

    async def get_poll(self, target):
        self.calls.append(("get_poll", target.peer,
                           target.message_id))
        return [dict(a) for a in self._answers]

    async def vote(self, target, option_bytes):
        self.calls.append(("vote", option_bytes))

    async def follow(self, target):
        self.calls.append(("follow", target.peer))


def test_session_actor_maps_verbs(tmp_path):
    client = FakeMassClient()
    rec = AccountRecord(id="telegram:acc", platform="telegram",
                        identity="acc", status="linked",
                        session_path=str(tmp_path / "sess"))
    actor = TelegramSessionActor(rec, api_id="1", api_hash="h",
                                 client_factory=lambda path: client)
    assert actor.react(TG_POST, "🔥")["ok"] is True
    assert actor.poll_options(TG_POLL) == ["Yes", "No"]
    assert actor.vote(TG_POLL, 1)["ok"] is True
    assert actor.follow(TG_CHAN)["ok"] is True
    kinds = [c[0] for c in client.calls]
    assert kinds == ["connect", "react", "disconnect",
                     "connect", "get_poll", "disconnect",
                     "connect", "get_poll", "vote", "disconnect",
                     "connect", "follow", "disconnect"]
    # the vote cast the poll's OWN option bytes for index 1
    assert ("vote", b"No") in client.calls


def test_session_actor_refuses_without_credentials(tmp_path):
    rec = AccountRecord(id="telegram:acc", platform="telegram",
                        identity="acc", status="linked",
                        session_path=str(tmp_path / "sess"))
    actor = TelegramSessionActor(rec, api_id="", api_hash="")
    with pytest.raises(MassRefusal):
        actor.react(TG_POST, "🔥")


# ── WhatsApp bridge actor + the real socket path ────────────────────

def test_bridge_actor_call_shapes():
    seen: list[dict] = []

    def fake_request(payload):
        seen.append(payload)
        if payload["cmd"] == "channel_resolve":
            return {"ok": True, "chat": "120@newsletter",
                    "name": "Chan"}
        return {"ok": True}

    rec = AccountRecord(id="whatsapp:acc", platform="whatsapp",
                        identity="acc", status="linked", port=9101)
    actor = WhatsAppBridgeActor(rec, request=fake_request)
    assert actor.follow(WA_POST)["ok"] is True
    assert actor.react(WA_POST, "🔥")["ok"] is True
    cmds = [p["cmd"] for p in seen]
    # the invite resolves once; the JID is cached for the react
    assert cmds == ["channel_resolve", "channel_follow",
                    "channel_react"]
    react_payload = seen[-1]
    assert react_payload["chat"] == "120@newsletter"
    assert react_payload["server_id"] == "77"
    assert react_payload["emoji"] == "🔥"
    out = actor.vote(WA_POST, 0)
    assert out["ok"] is False and "Baileys" in out["error"]
    assert len(seen) == 3  # the vote never touched the bridge


def test_bridge_request_over_a_real_socket():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]
    received: list[dict] = []

    def serve():
        conn, _ = server.accept()
        data = b""
        while not data.endswith(b"\n"):
            chunk = conn.recv(4096)
            if not chunk:
                break
            data += chunk
        req = json.loads(data.decode())
        received.append(req)
        conn.sendall(json.dumps(
            {"id": req["id"], "ok": True,
             "chat": "120@newsletter"}).encode() + b"\n")
        conn.close()
        server.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    out = bridge_request(port, {"cmd": "channel_resolve",
                                "invite": "0029VaXyZ"})
    thread.join(timeout=5)
    assert out["ok"] is True and out["chat"] == "120@newsletter"
    assert received[0]["cmd"] == "channel_resolve"
    assert received[0]["invite"] == "0029VaXyZ"


# ── gateway actor (primary sessions) ────────────────────────────────

class _Res:
    def __init__(self, ok=True, error=""):
        self.ok = ok
        self.error = error


class FakeGatewayTG:
    def __init__(self):
        self.calls = []

    def react(self, chat, message_id, emoji):
        self.calls.append(("react", chat, message_id, emoji))
        return _Res()

    def poll_options(self, chat, message_id):
        self.calls.append(("options", chat, message_id))
        return ["A", "B"]

    def vote_poll(self, chat, message_id, option_index):
        self.calls.append(("vote", chat, message_id, option_index))
        return _Res()

    def join_channel(self, chat):
        self.calls.append(("join", chat, None))
        return _Res()


def test_gateway_actor_uses_the_live_userbot_verbs():
    from nomorals.social.mass_actions import GatewayAdapterActor

    fake = FakeGatewayTG()
    actor = GatewayAdapterActor(fake, "telegram")
    assert actor.react(TG_POST, "🔥")["ok"] is True
    assert actor.poll_options(TG_POLL) == ["A", "B"]
    assert actor.vote(TG_POLL, 1)["ok"] is True
    assert actor.follow(TG_CHAN)["ok"] is True
    # the adapter receives a ChatRef for the channel (@-form for
    # a public username) and the message id as a string
    assert [(c[0], c[1].chat_id, c[2]) for c in fake.calls] == [
        ("react", "@chan", "9"), ("options", "@chan", "10"),
        ("vote", "@chan", "10"), ("join", "@chan", None)]
    assert fake.calls[0][3] == "🔥"
    assert fake.calls[2][3] == 1


# ── the DM handler grammar ──────────────────────────────────────────

def _bare_mass_runtime(tmp_path, pool):
    from nomorals.agents.partner.runtime_provision import (
        RuntimeProvisionMixin)
    from nomorals.social.mass_actions import (MassArms, MassJournal,
                                              MassRunner)

    rt = RuntimeProvisionMixin.__new__(RuntimeProvisionMixin)
    clock = FakeClock()
    arms = MassArms(tmp_path / "arms.json")
    journal = MassJournal(tmp_path / "journal.jsonl")
    runner = MassRunner(arms, journal, clock=clock.now,
                        sleep=clock.sleep, jitter_s=0.0,
                        pacing_floor=0.5)
    svc = {"arms": arms, "journal": journal, "runner": runner,
           "registry": SimpleNamespace(all=lambda: [])}
    rt._mass_service = lambda: svc  # type: ignore[attr-defined]
    rt._mass_handles = lambda platform: pool  # type: ignore[attr-defined]
    return rt, arms, journal


def test_handler_arm_status_and_runs(tmp_path):
    pool = make_pool(2)
    rt, arms, journal = _bare_mass_runtime(tmp_path, pool)
    out = rt._control_mass("massreact", "arm")
    assert "may flag, restrict, or ban" in out  # warning at arm
    assert "ARMED" in out
    assert arms.state("massreact")["armed"] is True
    out = rt._control_mass("massreact", "status")
    assert "ARMED" in out and "pool:" in out
    out = rt._control_mass("massreact", "disarm")
    assert "disarmed" in out
    assert arms.state("massreact")["armed"] is False


def test_handler_run_grammar_dry_then_live(tmp_path):
    pool = make_pool(2)
    rt, arms, journal = _bare_mass_runtime(tmp_path, pool)
    rt._control_mass("massreact", "arm")
    out = rt._control_mass(
        "massreact", "https://t.me/chan/9 | 🔥")
    assert "DRY RUN" in out  # first run defaults dry
    assert all(h.fake.calls == [] for h in pool)
    out = rt._control_mass(
        "massreact", "https://t.me/chan/9 | 🔥 | live")
    assert "DRY RUN" not in out
    assert all(len(h.fake.calls) == 1 for h in pool)
    out = rt._control_mass("massreact", "runs")
    assert "recent /massreact runs" in out


def test_handler_unarmed_live_refusal_carries_the_warning(tmp_path):
    pool = make_pool(1)
    rt, arms, journal = _bare_mass_runtime(tmp_path, pool)
    out = rt._control_mass(
        "massreact", "https://t.me/chan/9 | 🔥 | live")
    assert out.startswith("⚠️")
    assert "may flag, restrict, or ban" in out


def test_handler_pacing_and_bad_link(tmp_path):
    pool = make_pool(1)
    rt, arms, journal = _bare_mass_runtime(tmp_path, pool)
    out = rt._control_mass("massreact", "pacing 1")
    assert out.startswith("⚠️")  # floor refusal (real 20s floor)
    out = rt._control_mass("massreact", "pacing 30")
    assert "30s" in out
    out = rt._control_mass("massreact", "https://example.com/x | 🔥")
    assert out.startswith("⚠️")
    out = rt._control_mass("massreact", "https://t.me/chan/9")
    assert "needs its emoji" in out
