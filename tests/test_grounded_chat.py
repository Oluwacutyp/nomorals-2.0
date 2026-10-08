"""Tests for grounded hybrid retrieval, chat-bound sessions, and /research verbs.

All offline: the hashing embedder needs no network, the vector backend is
the built-in legacy store on :memory: sqlite, and the runtime verbs run
against a stub context/router.
"""

import tempfile
import unittest
from pathlib import Path

from nomorals.agents import features
from nomorals.agents.partner.runtime_live import RuntimeLiveMixin
from nomorals.memory.embeddings import Embedder
from nomorals.memory.vector_backends import LegacyStoreBackend
from nomorals.research.grounded import GroundedSession, GroundedError
from nomorals.research.grounded_store import GroundedSessionStore
from nomorals.social.chat.base import ChatMessage, ChatRef, MediaRef
from nomorals.storage.db import Database

DOC_A = "The engineers optimized the pipeline for faster throughput."
DOC_B = "Bananas grow in tropical climates near the equator."
# morphological variants of DOC_A's content words: FTS (no stemming, exact
# terms) finds nothing, but the hashing embedder's stemmer collides
# engineer/engineers, optimizing/optimized, pipelines/pipeline.
PARAPHRASE_Q = "engineer optimizing pipelines"


def _hybrid_session():
    db = Database(":memory:")
    db.migrate()
    backend = LegacyStoreBackend(db, owner_type="grounded")
    return GroundedSession(embedder=Embedder(provider="hashing"),
                           vector_db=backend)


def _llm_citing(prompt):
    assert "[S1]" in prompt
    return "The engineers did the optimization work [S1]."


class HybridRetrievalTests(unittest.TestCase):
    def test_fts_misses_paraphrase(self):
        # prove the test wedge: FTS alone finds nothing for the paraphrase
        s = GroundedSession()
        s.add_text(DOC_A, title="doc a")
        s.add_text(DOC_B, title="doc b")
        self.assertEqual([], s.index.search(PARAPHRASE_Q, limit=10))

    def test_hybrid_catches_paraphrase(self):
        s = _hybrid_session()
        id_a = s.add_text(DOC_A, title="doc a")
        s.add_text(DOC_B, title="doc b")
        ans = s.ask(PARAPHRASE_Q, llm_fn=_llm_citing)
        self.assertFalse(ans.refused)
        self.assertIn("[1]", ans.text)
        self.assertNotIn("[S1]", ans.text)
        self.assertEqual(1, len(ans.sources))
        self.assertTrue(ans.sources[0].doc_id.startswith(id_a),
                        f"expected doc a chunk, got {ans.sources[0].doc_id}")

    def test_hybrid_fts_hit_still_cited(self):
        s = _hybrid_session()
        s.add_text(DOC_A, title="doc a")
        ans = s.ask("what did the engineers optimize?", llm_fn=_llm_citing)
        self.assertFalse(ans.refused)
        self.assertIn("[1]", ans.text)

    def test_vector_index_failure_degrades_to_fts(self):
        class BrokenPut:
            def put_many(self, items):
                raise RuntimeError("disk is gone")
            def search(self, vector, *, limit=10, min_score=-1.0):
                raise RuntimeError("disk is gone")
        s = GroundedSession(embedder=Embedder(provider="hashing"),
                            vector_db=BrokenPut())
        s.add_text(DOC_A, title="doc a")
        # ingest-time vector failure disabled the vector lane; FTS still answers
        ans = s.ask("what did the engineers optimize?", llm_fn=_llm_citing)
        self.assertFalse(ans.refused)
        self.assertIn("[1]", ans.text)

    def test_vector_search_failure_falls_back_to_fts(self):
        db = Database(":memory:")
        db.migrate()
        backend = LegacyStoreBackend(db, owner_type="grounded")
        real_search = backend.search
        calls = {"n": 0}
        def flaky_search(vector, *, limit=10, min_score=-1.0):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("transient")
            return real_search(vector, limit=limit, min_score=min_score)
        backend.search = flaky_search
        s = GroundedSession(embedder=Embedder(provider="hashing"),
                            vector_db=backend)
        s.add_text(DOC_A, title="doc a")
        ans = s.ask("what did the engineers optimize?", llm_fn=_llm_citing)
        self.assertFalse(ans.refused)
        self.assertIn("[1]", ans.text)

    def test_hybrid_no_hits_anywhere_labeled_fallback(self):
        s = _hybrid_session()
        s.add_text(DOC_A, title="doc a")
        def llm(prompt):
            if "do not cover" in prompt:
                return "here is the general picture."
            return "CANNOT_ANSWER"
        # vector lane always returns *something* when vectors exist, but the
        # model still refuses on irrelevant sources -> labeled fallback
        ans = s.ask("zxqwkj asdfzxcv qwerty unrelated", llm_fn=llm)
        self.assertFalse(ans.refused)
        self.assertIn("Not in your documents", ans.text)

    def test_fts_only_path_unchanged(self):
        # no embedder/vector_db -> identical behavior to before hybrid existed
        s = GroundedSession()
        self.assertFalse(s._vector_enabled)
        s.add_text(DOC_A, title="doc a")
        with self.assertRaises(GroundedError):
            s.ask("   ")
        ans = s.ask("engineers?", llm_fn=_llm_citing)
        self.assertIn("[1]", ans.text)


class SessionStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = GroundedSessionStore(Path(self.tmp.name) / "grounded")

    def test_bind_get_add_upload_docs(self):
        s = self.store.bind("chat1")
        self.assertIsInstance(s, GroundedSession)
        self.assertIs(s, self.store.get("chat1"))  # get-or-create identity
        doc_id = self.store.add_upload("chat1", b"hello world " * 50,
                                       "notes.txt", mime="text/plain")
        self.assertTrue(doc_id)
        docs = self.store.list_docs("chat1")
        self.assertEqual(1, len(docs))
        self.assertEqual("notes.txt", docs[0]["title"])
        # the upload lives inside the session dir
        session_dir = self.store._session_dir("chat1")
        self.assertTrue((session_dir / "notes.txt").is_file())
        self.assertTrue((session_dir / "vectors.db").is_file())
        # and the session answers from it (hybrid: vector + FTS both live)
        ans = s.ask("hello?", llm_fn=_llm_citing)
        self.assertIn("[1]", ans.text)

    def test_get_missing_returns_none(self):
        self.assertIsNone(self.store.get("nope"))

    def test_drop_removes_session_and_dir(self):
        self.store.add_upload("chat9", b"data " * 50, "f.txt")
        session_dir = self.store._session_dir("chat9")
        self.assertTrue(session_dir.is_dir())
        self.store.drop("chat9")
        self.assertIsNone(self.store.get("chat9"))
        self.assertFalse(session_dir.exists())

    def test_ttl_expiry(self):
        store = GroundedSessionStore(Path(self.tmp.name) / "g2", ttl_seconds=-1)
        store.add_upload("c", b"data " * 50, "f.txt")
        session_dir = store._session_dir("c")
        self.assertTrue(session_dir.is_dir())
        # everything is instantly expired with a negative ttl
        self.assertIsNone(store.get("c"))
        self.assertFalse(session_dir.exists())

    def test_sweep_drops_expired(self):
        store = GroundedSessionStore(Path(self.tmp.name) / "g3", ttl_seconds=-1)
        store.add_upload("c1", b"data " * 50, "f.txt")
        store.add_upload("c2", b"data " * 50, "g.txt")
        dropped = store.sweep()
        self.assertEqual(2, dropped)
        self.assertFalse(store._session_dir("c1").exists())
        self.assertFalse(store._session_dir("c2").exists())

    def test_sweep_keeps_fresh(self):
        self.store.add_upload("fresh", b"data " * 50, "f.txt")
        self.assertEqual(0, self.store.sweep())
        self.assertIsNotNone(self.store.get("fresh"))

    def test_rehydrate_after_restart(self):
        self.store.add_upload("rc", b"rehydrate me " * 50, "r.txt")
        # simulate a process restart: new store over the same root
        store2 = GroundedSessionStore(Path(self.tmp.name) / "grounded")
        s2 = store2.bind("rc")
        docs = store2.list_docs("rc")
        self.assertEqual(1, len(docs))
        self.assertEqual("r.txt", docs[0]["title"])
        ans = s2.ask("rehydrate?", llm_fn=_llm_citing)
        self.assertIn("[1]", ans.text)

    def test_empty_upload_rejected(self):
        with self.assertRaises(ValueError):
            self.store.add_upload("c", b"", "empty.txt")


class _Resp:
    def __init__(self, text):
        self.text = text


class _Router:
    def __init__(self, text="The engineers did the optimization work [S1]."):
        self._text = text

    def complete(self, prompt):
        assert "[S1]" in prompt or "do not cover" in prompt
        return _Resp(self._text)


class _Settings:
    def __init__(self, root):
        self._root = Path(root)

    def resolve(self, relative):
        return self._root / relative


class _Ctx:
    def __init__(self, root):
        self.db = None
        self.settings = _Settings(root)
        self.router = _Router()


class _Runtime(RuntimeLiveMixin):
    def __init__(self, context):
        self.context = context
        self.gateway = None


def _msg_with_doc(path, name="notes.txt"):
    return ChatMessage(
        chat=ChatRef(platform="test", chat_id="c1"),
        incoming=True,
        text="/research from",
        media=[MediaRef(path=str(path), mime="text/plain",
                        kind="document", name=name)],
    )


class ResearchVerbTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self._old = features.FEATURES["research"]
        features.FEATURES["research"] = (True, "test")
        self.addCleanup(lambda: features.FEATURES.__setitem__("research", self._old))
        self.rt = _Runtime(_Ctx(self.tmp.name))

    def _doc_path(self, name="notes.txt"):
        p = Path(self.tmp.name) / name
        p.write_text("The engineers optimized the pipeline for faster throughput. " * 10)
        return p

    def test_from_ask_docs_done_cycle(self):
        msg = _msg_with_doc(self._doc_path())
        reply = self.rt._control_research("from", chat_key="chat1", message=msg)
        self.assertIn("grounded on 1 document(s): notes.txt.", reply)
        self.assertIn("/research ask", reply)

        reply = self.rt._control_research("docs", chat_key="chat1")
        self.assertIn("grounded documents (1):", reply)
        self.assertIn("notes.txt", reply)

        reply = self.rt._control_research("ask what did the engineers optimize?",
                                          chat_key="chat1")
        self.assertIn("[1]", reply)
        self.assertIn("Sources:", reply)

        reply = self.rt._control_research("done", chat_key="chat1")
        self.assertIn("cleared", reply)

        reply = self.rt._control_research("ask anything", chat_key="chat1")
        self.assertIn("/research from", reply)

    def test_from_with_tail_path(self):
        p = self._doc_path("tail.txt")
        reply = self.rt._control_research(f"from {p}", chat_key="chat2")
        self.assertIn("grounded on 1 document(s): tail.txt.", reply)

    def test_from_skips_pathless_media(self):
        msg = ChatMessage(
            chat=ChatRef(platform="test", chat_id="c1"),
            incoming=True, text="/research from",
            media=[MediaRef(path=None, mime="text/plain",
                            kind="document", name="ghost.txt")],
        )
        reply = self.rt._control_research("from", chat_key="chat3", message=msg)
        self.assertIn("nothing to ground on", reply)

    def test_from_nothing_to_ground(self):
        reply = self.rt._control_research("from", chat_key="chat4")
        self.assertIn("nothing to ground on", reply)

    def test_ask_needs_question(self):
        msg = _msg_with_doc(self._doc_path())
        self.rt._control_research("from", chat_key="chat5", message=msg)
        reply = self.rt._control_research("ask", chat_key="chat5")
        self.assertIn("usage: /research ask", reply)

    def test_docs_empty_session(self):
        msg = _msg_with_doc(self._doc_path())
        self.rt._control_research("from", chat_key="chat6", message=msg)
        self.rt._control_research("clear", chat_key="chat6")
        reply = self.rt._control_research("docs", chat_key="chat6")
        self.assertIn("/research from", reply)

    def test_sessions_are_per_chat(self):
        msg = _msg_with_doc(self._doc_path())
        self.rt._control_research("from", chat_key="chatA", message=msg)
        reply = self.rt._control_research("docs", chat_key="chatB")
        self.assertIn("/research from", reply)

    def test_existing_verbs_untouched(self):
        # status verb still routes to the ResearchAgent path (feature on, no db
        # rows) — the grounded branch must not swallow it
        reply = self.rt._control_research("status", chat_key="chat1")
        self.assertIn("research feature", reply)


if __name__ == "__main__":
    unittest.main()
