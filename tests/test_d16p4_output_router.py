"""Doc 16 Pass 4 — the output router facade.

Pins the §3 routing decision (signals → route → recorded reason),
end-to-end recovery of the full text from every non-chunk route,
and the preservation of LongOutput's legacy tuple contract, which
the Telegram adapter and the notifier overflow rung consume
positionally and which this pass must not disturb.
"""

from __future__ import annotations

import json

from nomorals.social.chat.output_router import (
    ROUTE_CHUNKS,
    ROUTE_DOCUMENT,
    ROUTE_READER,
    ROUTE_READER_EDITION,
    OutputRouter,
    transport_caps,
)
from nomorals.social.chat.progress_throttle import (
    LongOutput,
    split_or_document,
)

KEY = "chat:telegram:42"

PROSE = (
    "Weekly research digest\n\n"
    + ("Finding: the signal held across every session this week, "
       "with volume confirming the move and no sign of the usual "
       "exhaustion pattern near the highs. " * 12 + "\n\n")
    + ("Risk note: the invalidation sits below the prior swing "
       "low; a close through it ends the thesis and the plan "
       "reverts to observation only. " * 12 + "\n\n")
)
MEDIUM = PROSE * 2          # ~7.5k chars → 2 chunks at the 3800 limit
LONG = PROSE * 14           # ~52k chars → past the 10-chunk reader band
CODE = "def compute(x):\n    return x * 2  # exact bytes matter\n" * 220


def _router(tmp_path=None) -> OutputRouter:
    return OutputRouter(tmp_path / "longdoc" if tmp_path else None)


class TestRoutingDecisions:
    def test_short_is_chunks_and_matches_legacy_route(self):
        router = _router()
        plan = router.plan("hello", platform="telegram", reader_key=KEY)
        assert plan.decision.kind == ROUTE_CHUNKS
        assert plan.intro_text == "" and plan.file_bytes == b""
        # byte-identical to today's behaviour: one native send
        assert LongOutput().route("hello") == ("chunks", ["hello"])

    def test_medium_goes_to_reader_with_working_verbs(self):
        router = _router()
        plan = router.plan(MEDIUM, platform="telegram", reader_key=KEY)
        assert plan.decision.kind == ROUTE_READER
        assert "reader" in plan.decision.reason
        assert plan.token
        assert f"read go {plan.token} n" in plan.intro_text
        assert MEDIUM[:200] in plan.intro_text  # page 1 rides the intro
        rows = [b for row in (plan.buttons or []) for b in row]
        assert any(payload.startswith("read go ") for _, payload in rows)

    def test_long_goes_reader_plus_edition_actions(self):
        router = _router()
        plan = router.plan(LONG, platform="telegram", reader_key=KEY)
        assert plan.decision.kind == ROUTE_READER_EDITION
        assert plan.decision.chunks > 10
        assert f"/read get {plan.token} html" in plan.intro_text
        assert f"/read get {plan.token} txt" in plan.intro_text
        payloads = [p for row in (plan.buttons or []) for _, p in row]
        assert f"read get {plan.token} html" in payloads
        assert f"read get {plan.token} txt" in payloads

    def test_code_kind_is_raw_document_byte_exact(self):
        router = _router()
        plan = router.plan(CODE, platform="telegram", reader_key=KEY,
                           kind="code")
        assert plan.decision.kind == ROUTE_DOCUMENT
        assert plan.file_bytes == CODE.encode("utf-8")
        assert plan.file_name.endswith(".txt")
        assert "byte for byte" in plan.caption

    def test_explicit_inline_wins_even_for_long_content(self):
        router = _router()
        plan = router.plan(LONG, platform="telegram", reader_key=KEY,
                           intent="inline")
        assert plan.decision.kind == ROUTE_CHUNKS
        assert "explicit inline" in plan.decision.reason

    def test_whatsapp_gets_reader_without_buttons(self):
        # The books pager's own gate: no inline buttons on WhatsApp —
        # the controls are the plain-text commands in the intro.
        router = _router()
        plan = router.plan(LONG, platform="whatsapp", reader_key=KEY)
        assert plan.decision.kind == ROUTE_READER_EDITION
        assert plan.buttons is None
        assert f"read go {plan.token} n" in plan.intro_text
        assert f"/read get {plan.token} txt" in plan.intro_text

    def test_sms_degrades_to_lossless_chunks(self):
        router = _router()
        caps = transport_caps("sms")
        assert not caps.reader and not caps.files
        plan = router.plan(LONG, platform="sms", reader_key=KEY)
        assert plan.decision.kind == ROUTE_CHUNKS
        assert "reader page" in plan.decision.reason

    def test_code_on_fileless_transport_degrades_to_chunks(self):
        router = _router()
        plan = router.plan(CODE, platform="sms", reader_key=KEY,
                           kind="code")
        assert plan.decision.kind == ROUTE_CHUNKS
        assert "cannot carry files" in plan.decision.reason

    def test_no_reader_identity_degrades_to_chunks(self):
        router = _router()
        plan = router.plan(MEDIUM, platform="telegram", reader_key="")
        assert plan.decision.kind == ROUTE_CHUNKS
        assert "reader identity" in plan.decision.reason


class TestRecovery:
    def _collect_pages(self, router: OutputRouter, plan) -> str:
        svc = router.reader
        session = svc.session_for(KEY, plan.decision.document_id)
        assert session is not None
        resolver = router.resolver()
        view = svc.view(session.session_id, resolver)
        bodies = [view.body]
        for _ in range(view.page_count - 1):
            view = svc.navigate(session.session_id, resolver, "next")
            assert view.status == "ok"
            bodies.append(view.body)
        return "".join(bodies)

    def test_reader_pages_join_exact(self):
        router = _router()
        plan = router.plan(MEDIUM, platform="telegram", reader_key=KEY)
        assert self._collect_pages(router, plan) == MEDIUM

    def test_reader_pages_join_exact_for_long(self):
        router = _router()
        plan = router.plan(LONG, platform="telegram", reader_key=KEY)
        assert self._collect_pages(router, plan) == LONG

    def test_edition_contains_every_part_of_the_text(self):
        router = _router()
        plan = router.plan(LONG, platform="telegram", reader_key=KEY)
        html = router.edition_html(plan.decision.document_id)
        assert html is not None
        assert "Powered by peacethefirst" in html
        # slices taken from inside paragraphs (start / middle / end
        # of the document), where the edition renders text verbatim
        assert LONG[100:400] in html
        assert LONG[2500:2800] in html
        assert LONG[-1500:-1200] in html
        assert "Complete edition" in html

    def test_raw_download_file_is_byte_exact(self, tmp_path):
        router = _router(tmp_path)
        plan = router.plan(LONG, platform="telegram", reader_key=KEY)
        found = router.file_for(plan.decision.document_id, "txt")
        assert found is not None
        path, name = found
        assert path.read_bytes() == LONG.encode("utf-8")
        assert name.endswith(".txt")

    def test_html_file_action_renders_the_edition(self, tmp_path):
        router = _router(tmp_path)
        plan = router.plan(LONG, platform="telegram", reader_key=KEY)
        found = router.file_for(plan.decision.document_id, "html")
        assert found is not None
        path, name = found
        assert name.endswith(".html")
        assert "Powered by peacethefirst" in path.read_text(
            encoding="utf-8")

    def test_stale_token_never_moves_the_reader(self):
        router = _router()
        plan = router.plan(MEDIUM, platform="telegram", reader_key=KEY)
        svc = router.reader
        first = svc.act(plan.token, "n", router.resolver())
        assert first.status == "ok"
        stale = svc.act(plan.token, "n", router.resolver())
        assert stale.status == "stale"
        assert "has not moved" in stale.note


class TestDecisionJournal:
    def test_route_and_reason_are_journaled(self, tmp_path):
        router = _router(tmp_path)
        router.plan(MEDIUM, platform="telegram", reader_key=KEY)
        journal = tmp_path / "longdoc" / "routes.jsonl"
        rows = [json.loads(line) for line in
                journal.read_text(encoding="utf-8").splitlines()]
        assert rows[-1]["route"] == ROUTE_READER
        assert rows[-1]["reason"]
        assert rows[-1]["reader_key"] == KEY

    def test_recent_lists_only_this_reader(self, tmp_path):
        router = _router(tmp_path)
        router.plan(MEDIUM, platform="telegram", reader_key=KEY)
        mine = router.recent(KEY)
        assert len(mine) == 1 and mine[0]["route"] == ROUTE_READER
        assert router.recent("chat:telegram:other") == []


class TestLegacyContractPreserved:
    """progress_throttle.LongOutput is consumed positionally by the
    Telegram adapter (``route(text)[0]``) and the notifier overflow
    rung; its tuple shape and thresholds are untouched by the
    facade."""

    def test_route_tuple_shape(self):
        short = LongOutput().route("one message")
        assert short[0] == "chunks" and short[1] == ["one message"]
        big = LongOutput().route("x" * 60_000)
        assert big[0] == "document"
        assert big[1] == ("x" * 60_000).encode("utf-8")

    def test_split_or_document_short(self):
        assert split_or_document("short") == ("chunks", ["short"])
