"""Oral-tradition ingest wiring: the pipeline module existed, but no
production caller, surface, or retrieval reached it. These tests pin
the wiring: keeper seam, durable drafts across processes, CLI loop,
chat loop, and ask retrieval.

Failing-first: on the pristine base this file fails at collection
(ImportError: keeper has no `oral`, no `parse_oral_flags`).
"""
from __future__ import annotations

import io
import json
import os
import sys
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nomorals.cmdline.commands.wisdom import _cmd_wisdom
from nomorals.cmdline.parser import _parser
from nomorals.wisdom import WisdomKeeper
from nomorals.wisdom.oral import (
    MissingTranscriberError,
    OralError,
    OralIngest,
    PermissionRecord,
    parse_oral_flags,
)
from nomorals.wisdom.traditions import RestrictedCorpusError

TRANSCRIPT = "Ada: The river remembers every footstep.\nElder: And the drum answers.\n"


def _args(argv):
    return _parser().parse_args(argv)


def _run_cli(argv, ctx):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = _cmd_wisdom(_args(argv), ctx)
    return rc, out.getvalue(), err.getvalue()


def _ctx(tmp):
    settings = SimpleNamespace(workspace_dir=str(tmp))
    return SimpleNamespace(db=None, extras={}, settings=settings)


def test_keeper_exposes_oral(tmp_path):
    keeper = WisdomKeeper(_ctx(tmp_path))
    oral = keeper.oral
    assert isinstance(oral, OralIngest)
    assert str(oral.store_dir).endswith("wisdom/oral")
    assert keeper.oral is oral  # lazy, memoized


def test_parse_oral_flags():
    flags, pos = parse_oral_flags(
        ["d1", "--lang", "Yoruba", "--pack", "nigeria", "--reject"])
    assert pos == ["d1"]
    assert flags["lang"] == "Yoruba"
    assert flags["pack"] == "nigeria"
    assert flags["reject"] is True
    with pytest.raises(ValueError):
        parse_oral_flags(["d1", "--bogus", "x"])


def test_durable_draft_across_instances(tmp_path):
    store = tmp_path / "oral"
    a = OralIngest(store)
    a.submit_draft("d1", transcript=TRANSCRIPT, language="Yoruba",
                   tradition_pack_id="nigeria")
    # A FRESH instance (a later one-shot process) sees the draft.
    b = OralIngest(store)
    draft = b.get("d1")
    assert draft.status == "draft"
    assert len(draft.turns) == 2
    b.verify("d1", owner="owner", approve=True)
    c = OralIngest(store)
    assert c.get("d1").status == "verified"
    out = c.ingest("d1")
    assert out["ok"] is True
    assert out["words"] == 9
    assert (store / "d1.json").is_file()
    assert not (store / "d1.draft.json").is_file()


def test_restricted_and_unverified_refuse(tmp_path):
    oral = OralIngest(tmp_path / "oral")
    with pytest.raises(RestrictedCorpusError):
        oral.submit_draft("d-blocked", transcript="Elder: secret.",
                          restriction="initiated")
    perm = PermissionRecord(granted_by="community elder",
                            scope="one recording")
    oral.submit_draft("d2", transcript="Ada: Hi.", language="Yoruba",
                      tradition_pack_id="nigeria",
                      restriction="community-permission",
                      permission=perm)
    with pytest.raises(OralError):
        oral.ingest("d2")  # not verified yet
    oral.verify("d2", owner="owner", approve=False)
    with pytest.raises(OralError):
        oral.ingest("d2")  # rejected, never ingestible
    with pytest.raises(MissingTranscriberError):
        oral.transcribe("audio.mp3")


def test_search_and_list_ingested(tmp_path):
    oral = OralIngest(tmp_path / "oral")
    oral.submit_draft("d1", transcript=TRANSCRIPT, language="Yoruba",
                      tradition_pack_id="nigeria", contributor="Ada")
    oral.verify("d1", owner="owner", approve=True)
    oral.ingest("d1")
    hits = oral.search("river remembers")
    assert len(hits) == 1
    assert hits[0]["draft_id"] == "d1"
    assert hits[0]["speaker"] == "Ada"
    assert oral.search("nothing matches here") == []
    listed = oral.list_ingested()
    assert len(listed) == 1
    assert listed[0]["draft_id"] == "d1"
    assert listed[0]["words"] == 9
    rec = oral.read_ingested("d1")
    assert rec["provenance"]["contributor"] == "Ada"
    with pytest.raises(OralError):
        oral.read_ingested("ghost")


def test_cli_full_loop(tmp_path):
    ctx = _ctx(tmp_path)
    tpath = tmp_path / "t.txt"
    tpath.write_text(TRANSCRIPT, encoding="utf-8")
    rc, out, _ = _run_cli(
        ["wisdom", "oral", "submit", "d1", "--file", str(tpath),
         "--language", "Yoruba", "--pack", "nigeria",
         "--contributor", "Ada"], ctx)
    assert rc == 0, out
    assert "draft d1 submitted" in out
    rc, out, _ = _run_cli(["wisdom", "oral", "verify", "d1"], ctx)
    assert rc == 0, out
    assert "verified by owner" in out
    rc, out, _ = _run_cli(["wisdom", "oral", "ingest", "d1"], ctx)
    assert rc == 0, out
    assert "ingested: d1" in out
    rc, out, _ = _run_cli(["wisdom", "oral", "list"], ctx)
    assert rc == 0, out
    assert "d1: ingested" in out
    rc, out, _ = _run_cli(["wisdom", "oral", "read", "d1"], ctx)
    assert rc == 0, out
    assert "Ada: The river remembers every footstep." in out
    assert "contributor: Ada" in out
    # Retrieval: ask finds the ingested oral passage.
    rc, out, _ = _run_cli(["wisdom", "ask", "river remembers"], ctx)
    assert rc == 0, out
    assert "oral tradition" in out
    assert "The river remembers every footstep." in out
    assert "/wisdom oral read d1" in out
    # Status carries the oral count.
    rc, out, _ = _run_cli(["wisdom", "status", "--json"], ctx)
    assert rc == 0, out
    assert json.loads(out)["oral"]["ingested"] == 1


def test_cli_unknown_flag_and_missing_file(tmp_path):
    ctx = _ctx(tmp_path)
    # An unknown --flag is rejected by argparse itself (exit 2), the
    # same as every other nm command with a bad flag.
    with pytest.raises(SystemExit) as exc:
        _run_cli(["wisdom", "oral", "submit", "d1", "--bogus", "x"], ctx)
    assert exc.value.code == 2
    rc, _, err = _run_cli(
        ["wisdom", "oral", "submit", "d1", "--file",
         str(tmp_path / "nope.txt")], ctx)
    assert rc == 1
    assert "cannot read transcript" in err
    tpath = tmp_path / "t.txt"
    tpath.write_text(TRANSCRIPT, encoding="utf-8")
    rc, _, err = _run_cli(
        ["wisdom", "oral", "submit", "d2", "--file", str(tpath),
         "--restriction", "initiated"], ctx)
    assert rc == 1  # restricted, no permission record
    assert "STOP" in err


def test_chat_full_loop(tmp_path):
    from nomorals.agents.partner.runtime_wisdom import RuntimeWisdomMixin

    class FakeRuntime(RuntimeWisdomMixin):
        def __init__(self, ctx):
            self.context = ctx
            self.sent: list[str] = []

        def _ref_from_key(self, key):
            plat, _, cid = key.partition(":")
            return SimpleNamespace(platform=plat or "local",
                                   chat_id=cid, key=key)

        def _send_long_checked(self, platform, chat, text):
            self.sent.append(text)
            return ""

    ctx = _ctx(tmp_path)
    rt = FakeRuntime(ctx)
    reply = rt._control_wisdom(
        "oral submit d1 --lang Yoruba --pack nigeria\n" + TRANSCRIPT,
        chat_key="telegram:1")
    assert "draft d1 submitted" in reply
    reply = rt._control_wisdom("oral verify d1", chat_key="telegram:1")
    assert "verified by owner" in reply
    reply = rt._control_wisdom("oral ingest d1", chat_key="telegram:1")
    assert "ingested: d1" in reply
    reply = rt._control_wisdom("oral list", chat_key="telegram:1")
    assert "d1: ingested" in reply
    rt._control_wisdom("oral read d1", chat_key="telegram:1")
    assert "The river remembers every footstep." in rt.sent[-1]
    # Retrieval through the chat ask path.
    rt._control_wisdom("ask river remembers", chat_key="telegram:1")
    assert "oral tradition" in rt.sent[-1]
    assert "The river remembers every footstep." in rt.sent[-1]
    # Usage mentions the oral verbs; unknown oral verb names itself.
    assert "/wisdom oral submit" in rt._control_wisdom(
        "", chat_key="telegram:1")
    assert "unknown /wisdom oral verb" in rt._control_wisdom(
        "oral frobnicate", chat_key="telegram:1")


def test_ask_include_oral_opt_out(tmp_path):
    keeper = WisdomKeeper(_ctx(tmp_path))
    keeper.oral.submit_draft("d1", transcript=TRANSCRIPT,
                             language="Yoruba",
                             tradition_pack_id="nigeria")
    keeper.oral.verify("d1", owner="owner", approve=True)
    keeper.oral.ingest("d1")
    ans = keeper.ask("river remembers", top=5)
    assert any(p.canon_status == "oral-tradition" for p in ans.passages)
    assert "/wisdom oral read d1" in ans.synthesis
    ans2 = keeper.ask("river remembers", top=5, include_oral=False)
    assert not any(p.canon_status == "oral-tradition"
                   for p in ans2.passages)
