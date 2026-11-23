"""Doc 16 residual — the standalone shared live reader.

Pins ``LiveReader`` (``nomorals/social/chat/live_reader.py``): one
shared reader over the shipped ``ReaderService`` engine and the
output router's store.  The live-reading behaviour — opening a
document for live reading, actuating a reading token, reopening a
routed output, delivering its file actions — was spread across the
router's ``plan()``, the ``/read`` desk and the ``/books`` desk,
each hand-rolling the same session lookup + act + render sequence
against two separately constructed reader services.  This suite
pins the one module that owns it now:

* one shared service — the live reader's service IS the router's
  service, and a session opened by a routed send is the session a
  ``/read go`` token actuates;
* live reads resolve at read time — a revision that arrives while
  a document is open is flagged, never silently adopted, and the
  ``u`` switch is explicit and honest about where the reader
  lands; a segment's production state is labelled from its
  manifest only;
* the token grammar (n/p/c/jN/u), stale/unknown handling, the
  surface gate (``text:`` answers to ``/read``, ``story:`` to
  ``/books``), and the file actions all answer through the one
  module, in the desks' own words.
"""

from __future__ import annotations

from nomorals.core.longdoc import (
    Completeness,
    SegmentSpec,
    build_manifest,
)
from nomorals.social.chat.live_reader import (
    LiveReader,
    LiveReaderFault,
    LiveReadResult,
)
from nomorals.social.chat.output_router import (
    ROUTE_CHUNKS,
    ROUTE_DOCUMENT,
    ROUTE_READER,
    ROUTE_READER_EDITION,
)
from nomorals.social.chat.reader_service import (
    content_resolver_from_map,
)

KEY = "chat:telegram:42"
BOOKS_KEY = "chat:telegram:43"

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


def _live(tmp_path=None) -> LiveReader:
    return LiveReader(tmp_path / "longdoc" if tmp_path else None)


def _manifest(doc_id: str, bodies: dict[str, str | None],
              titles: dict[str, str] | None = None,
              metadata: dict[str, dict] | None = None):
    """A manifest over ordered ``segment_id → body`` (None = absent)."""
    titles = titles or {}
    metadata = metadata or {}
    specs = [
        SegmentSpec(segment_id=sid, kind="chapter",
                    title=titles.get(sid, sid), content=body,
                    content_ref=f"{doc_id}#{sid}",
                    metadata=metadata.get(sid, {}))
        for sid, body in bodies.items()
    ]
    return build_manifest(
        document_id=doc_id, source_kind="story",
        title=titles.get("title", doc_id),
        segment_specs=specs,
        completeness=Completeness(
            status="complete", declared_total=len(specs),
            present_count=sum(1 for b in bodies.values()
                              if b is not None),
            note="test manifest"),
    )


def _resolver(bodies: dict[str, str]):
    return content_resolver_from_map(
        {f"story:slug#{sid}": body for sid, body in bodies.items()})


class TestModule:
    def test_imports_and_exports(self):
        assert LiveReader is not None
        assert LiveReadResult("ok", "page").status == "ok"
        assert issubclass(LiveReaderFault, Exception)
        import nomorals.social.chat.live_reader as mod

        assert "LiveReader" in mod.__all__
        assert "LiveReadResult" in mod.__all__

    def test_service_is_the_routers_service(self):
        live = _live()
        assert live.service is live.router.reader
        assert live.reader is live.service

    def test_for_context_shares_one_store(self, tmp_path):
        import types

        ctx = types.SimpleNamespace(settings=types.SimpleNamespace(
            workspace_dir=str(tmp_path)))
        first = LiveReader.for_context(ctx)
        plan = first.open_text(MEDIUM, reader_key=KEY,
                               platform="telegram")
        assert plan.decision.kind == ROUTE_READER
        # A fresh reader over the same store resumes the session.
        second = LiveReader.for_context(ctx)
        session = second.service.session_for(
            KEY, plan.decision.document_id)
        assert session is not None
        assert session.token == plan.token

    def test_install_epub_renderer_is_idempotent(self):
        live = _live()
        assert live.install_epub_renderer(lambda m, t: b"epub") is True
        assert live.install_epub_renderer(lambda m, t: b"other") is False
        assert live.router.epub_renderer is not None


class TestOpenText:
    def test_short_is_chunks(self):
        live = _live()
        plan = live.open_text("hello", reader_key=KEY,
                              platform="telegram")
        assert plan.decision.kind == ROUTE_CHUNKS

    def test_medium_goes_to_reader_with_working_verbs(self):
        live = _live()
        plan = live.open_text(MEDIUM, reader_key=KEY,
                              platform="telegram")
        assert plan.decision.kind == ROUTE_READER
        assert plan.token
        assert f"read go {plan.token} n" in plan.intro_text
        result = live.read(plan.token, "n")
        assert result.status == "ok"
        assert result.kind == "page"

    def test_long_goes_reader_plus_edition_actions(self):
        live = _live()
        plan = live.open_text(LONG, reader_key=KEY,
                              platform="telegram")
        assert plan.decision.kind == ROUTE_READER_EDITION
        assert f"/read get {plan.token} html" in plan.intro_text
        assert f"/read get {plan.token} pdf" in plan.intro_text
        assert f"/read get {plan.token} epub" in plan.intro_text
        assert f"/read get {plan.token} txt" in plan.intro_text

    def test_code_kind_is_raw_document_byte_exact(self):
        live = _live()
        code = "def compute(x):\n    return x * 2\n" * 220
        plan = live.open_text(code, reader_key=KEY,
                              platform="telegram", kind="code")
        assert plan.decision.kind == ROUTE_DOCUMENT
        assert plan.file_bytes == code.encode("utf-8")

    def test_explicit_inline_wins(self):
        live = _live()
        plan = live.open_text(LONG, reader_key=KEY,
                              platform="telegram", intent="inline")
        assert plan.decision.kind == ROUTE_CHUNKS

    def test_pages_join_exact_through_the_live_reader(self):
        live = _live()
        plan = live.open_text(MEDIUM, reader_key=KEY,
                              platform="telegram")
        session = live.service.session_for(
            KEY, plan.decision.document_id)
        assert session is not None
        seen = []
        view = live.service.view(session.session_id,
                                 live.router.resolver())
        while True:
            seen.append(view.body)
            if not (view.has_next_page or view.has_next_segment):
                break
            view = live.service.navigate(
                session.session_id, live.router.resolver(), "next")
        assert "".join(seen) == MEDIUM


class TestRead:
    def _opened(self, live, bodies=None):
        bodies = bodies or {
            "ch-1": "Chapter one. " * 400,
            "ch-2": "Chapter two. " * 400,
            "ch-3": "Chapter three. " * 400,
        }
        titles = {"ch-1": "Chapter one", "ch-2": "Chapter two",
                  "ch-3": "Chapter three"}
        manifest = _manifest("story:slug", bodies, titles=titles)
        session = live.service.open(
            BOOKS_KEY, manifest, command_prefix="books go")
        return session, manifest, bodies

    def test_next_prev_contents_jump(self):
        live = _live()
        session, _manifest, _bodies = self._opened(live)
        resolver = _resolver({
            "ch-1": "Chapter one. " * 400,
            "ch-2": "Chapter two. " * 400,
            "ch-3": "Chapter three. " * 400,
        })
        first = live.read(session.token, "n", surface="books",
                          story_resolver=resolver)
        assert first.status == "ok"
        assert first.kind == "page"
        contents = live.read(first.view.token, "c", surface="books",
                             story_resolver=resolver)
        assert contents.status == "ok"
        assert contents.kind == "contents"
        assert "Chapter one" in contents.text
        assert "Chapter three" in contents.text
        jumped = live.read(contents.view.token if contents.view
                           else first.view.token, "j3",
                           surface="books", story_resolver=resolver)
        assert jumped.status == "ok"
        assert jumped.view.segment_id == "ch-3"

    def test_stale_token_never_moves_the_reader(self):
        live = _live()
        session, _manifest, _bodies = self._opened(live)
        resolver = _resolver({
            "ch-1": "Chapter one. " * 400,
            "ch-2": "Chapter two. " * 400,
            "ch-3": "Chapter three. " * 400,
        })
        old_token = session.token
        moved = live.read(old_token, "n", surface="books",
                          story_resolver=resolver)
        assert moved.status == "ok"
        stale = live.read(old_token, "n", surface="books",
                          story_resolver=resolver)
        assert stale.status == "stale"
        assert "has not moved" in stale.text
        current = live.service.session_for(
            BOOKS_KEY, "story:slug")
        assert current.token == moved.view.token

    def test_unknown_token_names_the_desk(self):
        live = _live()
        result = live.read("deadbeef0000", "n", surface="read")
        assert result.status == "unknown"
        assert "/read lists what this chat has open" in result.text
        books = live.read("deadbeef0000", "n", surface="books")
        assert books.status == "unknown"
        assert "/books read <slug>" in books.text

    def test_surface_gate_keeps_documents_on_their_desk(self):
        live = _live()
        plan = live.open_text(MEDIUM, reader_key=KEY,
                              platform="telegram")
        wrong = live.read(plan.token, "n", surface="books")
        assert wrong.status == "note"
        assert "different document" in wrong.text
        session, _manifest, _bodies = self._opened(live)
        resolver = _resolver({
            "ch-1": "Chapter one. " * 400,
            "ch-2": "Chapter two. " * 400,
            "ch-3": "Chapter three. " * 400,
        })
        wrong = live.read(session.token, "n", surface="read",
                          story_resolver=resolver)
        assert wrong.status == "note"
        assert "story controls answer to /books go" in wrong.text

    def test_story_without_resolver_is_refused_honestly(self):
        live = _live()
        session, _manifest, _bodies = self._opened(live)
        result = live.read(session.token, "n", surface="books")
        assert result.status == "note"
        assert "not attached here" in result.text

    def test_live_revision_flag_and_honest_switch(self):
        live = _live()
        v1 = {"ch-1": "Chapter one v1. " * 400,
              "ch-2": "Chapter two v1. " * 400}
        live.service.open(
            BOOKS_KEY, _manifest("story:slug", v1),
            command_prefix="books go")
        v2 = {"ch-1": "Chapter one v2 — rewritten. " * 400,
              "ch-2": "Chapter two v1. " * 400}
        live.service.open(BOOKS_KEY, _manifest("story:slug", v2),
                          command_prefix="books go")
        resumed = live.service.session_for(BOOKS_KEY, "story:slug")
        assert resumed.update_available is True
        assert resumed.pinned().segments[0].content_ref.endswith(
            "ch-1")
        switched = live.read(resumed.token, "u", surface="books",
                             story_resolver=_resolver(v2))
        assert switched.status == "ok"
        assert "newer revision" in switched.view.note
        assert "v2" in switched.view.body
        assert switched.view.update_available is False

    def test_growth_elsewhere_never_moves_the_page(self):
        live = _live()
        v1 = {"ch-1": "Chapter one. " * 400}
        live.service.open(BOOKS_KEY, _manifest("story:slug", v1),
                          command_prefix="books go")
        v2 = {"ch-1": "Chapter one. " * 400,
              "ch-2": "Chapter two, new. " * 400}
        resumed = live.service.open(
            BOOKS_KEY, _manifest("story:slug", v2),
            command_prefix="books go")
        assert resumed.update_available is False
        assert resumed.segment_id == "ch-1"

    def test_draft_state_labelled_from_the_manifest_only(self):
        live = _live()
        bodies = {"ch-1": "Draft chapter, being written. " * 200,
                  "ch-2": "Finished chapter. " * 200}
        manifest = _manifest(
            "story:slug", bodies,
            metadata={"ch-1": {"status": "writing"}})
        session = live.service.open(
            BOOKS_KEY, manifest, command_prefix="books go")
        resolver = _resolver(bodies)
        view = live.service.view(session.session_id, resolver)
        assert view.segment_state == "draft — still being written"
        result = live.read(session.token, "n", surface="books",
                           story_resolver=resolver)
        assert "draft — still being written" in result.text

    def test_contents_for_renders_a_manifest(self):
        live = _live()
        bodies = {"ch-1": "Chapter one. " * 100,
                  "ch-2": None,
                  "ch-3": "Chapter three. " * 100}
        manifest = _manifest("story:slug", bodies)
        text = live.contents_for(
            manifest, open_hint="open one: /books read slug 1")
        assert "2 of 3 held" in text
        assert "not held" in text
        assert "open one: /books read slug 1" in text


class TestDesk:
    def test_open_recent_reopens_at_the_current_page(self,
                                                     tmp_path):
        live = _live(tmp_path)
        plan = live.open_text(MEDIUM, reader_key=KEY,
                              platform="telegram")
        assert plan.decision.kind == ROUTE_READER
        live.read(plan.token, "n")
        reopened = live.open_recent(KEY, 1)
        assert reopened.status == "ok"
        assert reopened.kind == "page"
        assert reopened.view.page_index == 1

    def test_open_recent_out_of_range_is_usage(self, tmp_path):
        live = _live(tmp_path)
        live.open_text(MEDIUM, reader_key=KEY, platform="telegram")
        result = live.open_recent(KEY, 99)
        assert result.status == "note"
        assert "usage: /read open <n>" in result.text

    def test_list_text_empty_and_filled(self, tmp_path):
        live = _live(tmp_path)
        assert "no routed long outputs" in live.list_text(KEY)
        live.open_text(MEDIUM, reader_key=KEY, platform="telegram",
                       title="Weekly digest")
        listed = live.list_text(KEY)
        assert "Weekly digest" in listed
        assert "/read open 1" in listed

    def test_file_for_txt_is_the_original(self, tmp_path):
        live = _live(tmp_path)
        plan = live.open_text(MEDIUM, reader_key=KEY,
                              platform="telegram")
        result = live.file_for(plan.token, "txt")
        assert result.status == "ok"
        assert result.kind == "file"
        path, name, caption = result.file
        assert path.exists()
        assert path.read_bytes().decode("utf-8") == MEDIUM
        assert "byte for byte" in caption
        assert name.endswith(".txt")

    def test_file_for_html_renders_the_edition(self, tmp_path):
        live = _live(tmp_path)
        plan = live.open_text(MEDIUM, reader_key=KEY,
                              platform="telegram")
        result = live.file_for(plan.token, "html")
        assert result.status == "ok"
        path, name, _caption = result.file
        assert name.endswith(".html")
        assert "Weekly research digest" in path.read_text(
            encoding="utf-8")

    def test_file_for_unknown_and_foreign_tokens(self, tmp_path):
        live = _live(tmp_path)
        unknown = live.file_for("deadbeef0000", "txt")
        assert unknown.status == "unknown"
        assert "file control has expired" in unknown.text
        session, _manifest, _bodies = TestRead()._opened(live)
        foreign = live.file_for(session.token, "txt")
        assert foreign.status == "note"
        assert "different document" in foreign.text

    def test_session_survives_a_fresh_live_reader(self, tmp_path):
        first = _live(tmp_path)
        plan = first.open_text(MEDIUM, reader_key=KEY,
                               platform="telegram")
        moved = first.read(plan.token, "n")
        assert moved.status == "ok"
        second = _live(tmp_path)
        session = second.service.session_for(
            KEY, plan.decision.document_id)
        assert session is not None
        assert session.token == moved.view.token
        view = second.service.view(session.session_id,
                                   second.router.resolver())
        assert view.page_index == 1
