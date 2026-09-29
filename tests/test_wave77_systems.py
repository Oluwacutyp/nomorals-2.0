"""Wave 77 — the five expansion systems.

* (b) corpus learning loop — cracked/taught words feed the wordlist
* (a) browser-acting monitor — headless render + volatile-tolerant diff
* (c) deploy TLS — real self-signed HTTPS termination for `nm apps`
* (d) training checkpoint resume — skip corrupt mid-save corpses
* (e) skill auto-pruning — decay skills that are used often, fail often

Every test is hermetic: temp homes, fake HTML (no network), fake
checkpoint trees, in-thread TLS servers on 127.0.0.1.
"""
from __future__ import annotations

import hashlib
import json
import os
import socket
import ssl
import tempfile
import threading
import time
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from nomorals.agents.context import build_context
from nomorals.core.config import Settings


def _ctx(tmp: str):
    return build_context(Settings(home=tmp), with_executor=False,
                         with_tools=True, with_router=False,
                         with_memory=False)


# ═══════════════════════════════════════════════════════════════════════════
# (b) corpus learning loop
# ═══════════════════════════════════════════════════════════════════════════

class TestCorpusLearning(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.ctx = _ctx(self._tmp.name)
        self.db = self.ctx.db

    def tearDown(self):
        self._tmp.cleanup()

    def test_learn_word_new_dedupe_and_reject(self):
        from nomorals.core.corpus import learn_word

        self.assertTrue(learn_word(self.db, "Sunshine2019!"))
        # same word, different case → dupe
        self.assertFalse(learn_word(self.db, "sunshine2019!"))
        # too short → rejected (brute force owns the short space)
        self.assertFalse(learn_word(self.db, "ab"))
        self.assertFalse(learn_word(self.db, "   "))

    def test_learn_word_core_normalization(self):
        from nomorals.core.corpus import learned_words, learn_word

        learn_word(self.db, "ZebraFish99!")
        words = learned_words(self.db)
        # the exact form AND the alphanumeric core are both learned
        self.assertIn("zebrafish99!", words)
        self.assertIn("zebrafish99", words)

    def test_learn_phrase_full_and_parts(self):
        from nomorals.core.corpus import learn_phrase

        added = learn_phrase(self.db, "Xylophone42!x", source="cracked")
        self.assertIn("xylophone42!x", added)
        self.assertIn("xylophone42", added)
        self.assertNotIn("x", added)  # single char part is not learned

    def test_forget_word(self):
        from nomorals.core.corpus import forget_word, learn_word

        learn_word(self.db, "tempword")
        self.assertTrue(forget_word(self.db, "tempword"))
        self.assertFalse(forget_word(self.db, "tempword"))

    def test_base_words_merge_and_order(self):
        from nomorals.core.corpus import BUILTIN_WORDS, base_words, learn_word

        learn_word(self.db, "brandnewword")
        words = base_words(self.db)
        # builtins keep their order at the front
        self.assertEqual(words[:len(BUILTIN_WORDS)],
                         list(BUILTIN_WORDS))
        self.assertIn("brandnewword", words)
        # no db → just the builtins
        self.assertEqual(len(base_words(None)), len(BUILTIN_WORDS))

    def test_corpus_stats_counts_learned(self):
        from nomorals.core.corpus import corpus_stats, learn_word

        before = corpus_stats(self.db)["learned_words"]
        learn_word(self.db, "anotherword")
        st = corpus_stats(self.db)
        self.assertEqual(st["learned_words"], before + 1)
        self.assertEqual(st["effective_base_words"],
                         st["base_words"] + st["learned_words"])

    def test_crack_prelearned_word_is_instant(self):
        from nomorals.core.corpus import learn_word
        from nomorals.tools.hashcrack import crack_hash

        secret = "zebrafish99"
        self.assertTrue(learn_word(self.db, secret))
        h = hashlib.md5(secret.encode()).hexdigest()
        res = crack_hash(h, mode="hybrid", max_candidates=100_000, db=self.db)
        self.assertEqual(res.found.get(h), secret)
        # phase 1 only: a few hundred candidates, never the brute force
        self.assertLess(res.tested, 5000)
        self.assertTrue(res.learned)

    def test_crack_auto_learns_plaintext_and_chains(self):
        from nomorals.core.corpus import learned_words
        from nomorals.tools.hashcrack import crack_hash

        # 'snoopy' is a corpus word; +year rule (2019) lands it in phase 2
        h = hashlib.md5(b"snoopy2019").hexdigest()
        res = crack_hash(h, mode="hybrid", max_candidates=200_000, db=self.db)
        self.assertEqual(res.found.get(h), "snoopy2019")
        # the plaintext is in the learned corpus now
        self.assertIn("snoopy2019", learned_words(self.db))
        # and the digest is in the known-hash chain → instant re-hit
        res2 = crack_hash(h, mode="hybrid", db=self.db)
        self.assertEqual(res2.backend, "known-hash")
        self.assertEqual(res2.found.get(h), "snoopy2019")
        self.assertEqual(res2.tested, 0)

    def test_engine_without_db_still_works(self):
        from nomorals.tools.hashcrack import Engine

        h = hashlib.md5(b"password").hexdigest()
        engine = Engine([h], mode="hybrid", max_candidates=50_000, quiet=True)
        res = engine.run()
        self.assertEqual(res.found.get(h), "password")


# ═══════════════════════════════════════════════════════════════════════════
# (a) browser-acting monitor
# ═══════════════════════════════════════════════════════════════════════════

def _html(price: str, nonce: str) -> str:
    return (
        "<html><head><title>S</title></head><body>"
        f"<h1>Price</h1><p>${price}</p>"
        f"<script>var nonce={nonce};</script>"
        "</body></html>"
    )


class TestPageMonitor(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.ctx = _ctx(self._tmp.name)
        self.state = {"html": _html("10", "0.1")}
        from nomorals.agents.monitor import MonitorAgent

        self.agent = MonitorAgent(self.ctx, notifier=None)
        self.url = "https://example.test/status"

    def tearDown(self):
        self._tmp.cleanup()

    def _fetch_patch(self, status: int = 200):
        from nomorals.tools.browser import BrowserSession

        return mock.patch.object(
            BrowserSession, "_fetch",
            lambda s, u: {"url": u, "status": status,
                          "text": self.state["html"]})

    def test_content_change_alerts_with_diff(self):
        with self._fetch_patch():
            self.agent.add(self.url, kind="page", interval=30)
            t0 = time.time()
            r1 = self.agent.tick(now=t0)
            self.assertEqual(r1["checked"], 1)
            self.assertFalse(r1["changed"])
            # markup churn only (script nonce) → NO alert
            self.state["html"] = _html("10", "0.999")
            r2 = self.agent.tick(now=t0 + 60)
            self.assertFalse(r2["changed"],
                             f"script churn must not alert: {r2}")
            # real content change → alert with diff
            self.state["html"] = _html("9", "0.5")
            r3 = self.agent.tick(now=t0 + 120)
            self.assertTrue(r3["changed"])
            diff = r3["changed"][0].get("diff", "")
            self.assertIn("$10", diff)
            self.assertIn("$9", diff)

    def test_volatile_regex_stripped_before_diff(self):
        from nomorals.tools.browser import BrowserSession

        def fetch(s, u):
            return {"url": u, "status": 200, "text": self.state["html"]}

        with mock.patch.object(BrowserSession, "_fetch", fetch):
            self.agent.add(self.url, kind="page", interval=30,
                           volatile=r"\d{2,}ms")
            t0 = time.time()
            self.agent.tick(now=t0)
            # a volatile latency value changes → no alert
            self.state["html"] = (
                "<html><head><title>S</title></head><body>"
                "<h1>Price</h1><p>$10</p>"
                "<span class=\"lat\">123ms</span></body></html>")
            r2 = self.agent.tick(now=t0 + 60)
            self.assertFalse(r2["changed"],
                             f"volatile value must not alert: {r2}")
            # a real change still alerts
            self.state["html"] = (
                "<html><head><title>S</title></head><body>"
                "<h1>Price</h1><p>$8</p>"
                "<span class=\"lat\">999ms</span></body></html>")
            r3 = self.agent.tick(now=t0 + 120)
            self.assertTrue(r3["changed"])

    def test_404_is_an_error_not_a_change(self):
        from nomorals.tools.browser import BrowserSession

        with mock.patch.object(
                BrowserSession, "_fetch",
                lambda s, u: {"url": u, "status": 404, "text": ""}):
            info = self.agent.add(self.url, kind="page", interval=30)
            self.assertTrue(info["id"])
            t0 = time.time()
            r = self.agent.tick(now=t0)
            self.assertEqual(len(r["errors"]), 1)
            self.assertIn("404", r["errors"][0]["error"])

    def test_page_kind_requires_http_url(self):
        with self.assertRaises(ValueError):
            self.agent.add("/workspace/somefile.txt", kind="page")

    def test_volatile_bad_regex_rejected_at_add(self):
        with self.assertRaises(Exception):
            self.agent.add(self.url, kind="page", volatile=r"[unclosed")

    def test_page_watch_fields_in_list(self):
        from nomorals.tools.browser import BrowserSession

        with mock.patch.object(BrowserSession, "_fetch",
                               lambda s, u: {"url": u, "status": 200,
                                             "text": "<p>x</p>"}):
            self.agent.add(self.url, kind="page", volatile=r"tok-\w+")
        row = self.agent.list()[0]
        self.assertEqual(row["kind"], "page")
        self.assertEqual(row["volatile"], r"tok-\w+")

    def test_page_watch_skips_auto_decode(self):
        # page watches diff rendered markdown — decoding it is a no-op,
        # and the decode feed must not run for kind=page
        from nomorals.agents import monitor as monitor_mod
        from nomorals.tools.browser import BrowserSession

        called = {"n": 0}

        def spy(self_, row, content, now):
            called["n"] += 1
            return None

        with mock.patch.object(BrowserSession, "_fetch",
                               lambda s, u: {"url": u, "status": 200,
                                             "text": "<p>x</p>"}), \
                mock.patch.object(monitor_mod.MonitorAgent, "_decode_feed",
                                  spy):
            self.agent.add(self.url, kind="page", interval=30)
            self.agent.tick(now=time.time())
        self.assertEqual(called["n"], 0)


# ═══════════════════════════════════════════════════════════════════════════
# (c) deploy TLS
# ═══════════════════════════════════════════════════════════════════════════

class TestDeployTLS(unittest.TestCase):
    def test_self_signed_cert_generation_and_reuse(self):
        from nomorals.builders_proxy import ensure_self_signed_cert

        with tempfile.TemporaryDirectory() as tmp:
            c, k = os.path.join(tmp, "t.crt"), os.path.join(tmp, "t.key")
            c2, k2 = ensure_self_signed_cert(c, k, common_name="x.test")
            self.assertEqual((c2, k2), (c, k))
            cert = open(c).read()
            key = open(k).read()
            self.assertIn("BEGIN CERTIFICATE", cert)
            self.assertIn("PRIVATE KEY", key)
            # second call reuses the existing pair
            self.assertEqual(ensure_self_signed_cert(c, k), (c, k))
            self.assertIn("BEGIN CERTIFICATE", open(c).read())

    def test_tls_proxy_serves_https_and_rejects_plain(self):
        import nomorals.builders_proxy as bp

        class H(BaseHTTPRequestHandler):
            def do_GET(self):
                body = json.dumps(
                    {"hello": "backend", "path": self.path}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        back = ThreadingHTTPServer(("127.0.0.1", 0), H)
        bport = back.server_address[1]
        threading.Thread(target=back.serve_forever, daemon=True).start()

        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        lport = s.getsockname()[1]
        s.close()
        with tempfile.TemporaryDirectory() as tmp:
            cert, key = os.path.join(tmp, "p.crt"), os.path.join(tmp, "p.key")
            threading.Thread(
                target=bp.run,
                kwargs={"listen_host": "127.0.0.1", "listen_port": lport,
                        "backend_host": "127.0.0.1", "backend_port": bport,
                        "tls": True, "cert": cert, "key": key,
                        "domain": "example.test"},
                daemon=True).start()
            deadline = time.time() + 10
            while time.time() < deadline:
                try:
                    with socket.create_connection(
                            ("127.0.0.1", lport), timeout=1):
                        break
                except OSError:
                    time.sleep(0.2)
            else:
                self.fail("TLS proxy never came up")
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            with urllib.request.urlopen(
                    f"https://127.0.0.1:{lport}/x/y", context=ctx,
                    timeout=5) as r:
                data = json.loads(r.read())
            self.assertEqual(data["hello"], "backend")
            self.assertEqual(data["path"], "/x/y")
            # plain HTTP to a TLS port must fail (real termination)
            with self.assertRaises(Exception):
                urllib.request.urlopen(
                    f"http://127.0.0.1:{lport}/", timeout=3)
        back.shutdown()


# ═══════════════════════════════════════════════════════════════════════════
# (d) training checkpoint resume
# ═══════════════════════════════════════════════════════════════════════════

def _mk_checkpoint(tmp: str, step: int, *, missing: tuple = (),
                   empty: tuple = (), no_payload: bool = False,
                   state_step=None) -> Path:
    d = Path(tmp) / f"checkpoint-{step}"
    d.mkdir(parents=True, exist_ok=True)
    ss = step if state_step is None else state_step
    if "trainer_state.json" not in missing:
        (d / "trainer_state.json").write_text(
            json.dumps({"global_step": ss}))
    if "optimizer.bin" not in missing:
        (d / "optimizer.bin").write_bytes(
            b"" if "optimizer.bin" in empty else b"x" * 100)
    if "scheduler.pt" not in missing:
        (d / "scheduler.pt").write_bytes(
            b"" if "scheduler.pt" in empty else b"y" * 10)
    if not no_payload:
        (d / "adapter_model.safetensors").write_bytes(b"z" * 50)
    return d


class TestCheckpointResume(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def test_validate_good_checkpoint(self):
        from nomorals.training.checkpoints import validate_checkpoint

        d = _mk_checkpoint(self.tmp, 500)
        ok, why = validate_checkpoint(d)
        self.assertTrue(ok, why)

    def test_validate_corrupt_variants(self):
        from nomorals.training.checkpoints import validate_checkpoint

        cases = [
            ("missing trainer_state.json",
             dict(missing=("trainer_state.json",))),
            ("empty optimizer", dict(empty=("optimizer.bin",))),
            ("step mismatch", dict(state_step=499)),
            ("no payload", dict(no_payload=True)),
        ]
        i = 0
        for label, kw in cases:
            i += 1
            d = _mk_checkpoint(self.tmp, 100 * i, **kw)
            ok, why = validate_checkpoint(d)
            self.assertFalse(ok, f"{label} should be corrupt: {why}")

    def test_pick_skips_corrupt_tails(self):
        from nomorals.training.checkpoints import pick_checkpoint

        _mk_checkpoint(self.tmp, 100)
        _mk_checkpoint(self.tmp, 500)
        _mk_checkpoint(self.tmp, 1000)
        # a kill mid-save left the newest one incomplete…
        _mk_checkpoint(self.tmp, 1500, missing=("trainer_state.json",))
        _mk_checkpoint(self.tmp, 2000, state_step=1999)
        picked = pick_checkpoint(self.tmp, max_step=6000)
        self.assertEqual(picked["step"], 1000)
        self.assertEqual(len(picked["skipped"]), 2)
        self.assertFalse(picked["complete"])

    def test_pick_complete_flag(self):
        from nomorals.training.checkpoints import pick_checkpoint

        _mk_checkpoint(self.tmp, 100)
        _mk_checkpoint(self.tmp, 1000)
        picked = pick_checkpoint(self.tmp, max_step=1000)
        self.assertEqual(picked["step"], 1000)
        self.assertTrue(picked["complete"])
        self.assertIn("DONE", picked["note"])

    def test_report_counts_and_remaining(self):
        from nomorals.training.checkpoints import report

        _mk_checkpoint(self.tmp, 100)
        _mk_checkpoint(self.tmp, 500)
        _mk_checkpoint(self.tmp, 1000)
        _mk_checkpoint(self.tmp, 1500, missing=("optimizer.bin",))
        rep = report(self.tmp, max_step=6000)
        self.assertEqual(rep["checkpoints"], 4)
        self.assertEqual(rep["valid"], 3)
        self.assertEqual(rep["corrupt"], 1)
        self.assertEqual(rep["resume_step"], 1000)
        self.assertEqual(rep["steps_remaining"], 5000)
        self.assertFalse(rep["complete"])

    def test_colab_script_embeds_validated_resume(self):
        import py_compile

        from nomorals.training.finetune import write_colab_script

        with tempfile.TemporaryDirectory() as t2:
            base = Path(t2) / "persona-mix"
            base.write_text("[]")
            out = write_colab_script(base)
            code = Path(out).read_text()
            py_compile.compile(out, doraise=True)
            self.assertIn("def _pick_checkpoint", code)
            self.assertIn("resuming from", code)
            self.assertIn("skipping", code)
            self.assertIn("trainer.train(resume_from_checkpoint=path)",
                          code)
            self.assertNotIn("last[-1]", code)  # naive resume is gone

    def test_notebook_embeds_validated_resume(self):
        from nomorals.training.finetune import write_colab_notebook

        with tempfile.TemporaryDirectory() as t2:
            out = write_colab_notebook(Path(t2) / "persona-mix")
            nb = json.loads(Path(out).read_text())
            srcs = ["".join(c["source"])
                    for c in nb["cells"] if c["cell_type"] == "code"]
            self.assertTrue(any("def _pick_checkpoint" in s for s in srcs),
                            "picker missing from notebook")
            self.assertTrue(any("SKIP_ADAPTER_SAVE" in s for s in srcs),
                            "save-guard flag missing")
            self.assertTrue(any("newest VALID" in s for s in srcs),
                            "handoff picker missing")
            self.assertFalse(any("<PICK_CHECKPOINT>" in s for s in srcs),
                             "uninjected marker")
            # the training cell (picker + resume) is pure python
            for s in srcs:
                if "def _pick_checkpoint" in s:
                    compile(s, "<nb-training-cell>", "exec")
                    break
            else:
                self.fail("training cell not found")


# ═══════════════════════════════════════════════════════════════════════════
# (e) skill auto-pruning
# ═══════════════════════════════════════════════════════════════════════════

class TestSkillPruning(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.ctx = _ctx(self._tmp.name)
        from nomorals.agents.skills import SkillLibrary

        self.lib = SkillLibrary(self.ctx.db)
        self.now = time.time()

    def tearDown(self):
        self._tmp.cleanup()

    def _skill(self, name, uses, successes, age_days=30):
        s = self.lib.save(name, kind="strategy", body="x", source="test")
        for i in range(uses):
            self.lib.record_use(s.id, success=(i < successes), task="t")
        back = self.now - age_days * 86400
        self.ctx.db.execute(
            "UPDATE skills SET created_at=?, updated_at=? WHERE id=?",
            (back, back, s.id))
        return s

    def test_prune_quarantines_dead_skill(self):
        self._skill("dead-strategy", uses=8, successes=1)  # 12.5%
        self._skill("good-strategy", uses=8, successes=7)  # 87.5%
        self._skill("young-failing", uses=2, successes=0)  # too few uses
        res = self.lib.prune(min_uses=5, max_success_rate=0.34,
                             min_age_days=7)
        self.assertEqual(res["pruned"], ["dead-strategy"])
        self.assertEqual(res["pruned_total"], 1)
        self.assertEqual(res["total"], 3)

    def test_prune_respects_age(self):
        s = self._skill("fresh-failing", uses=9, successes=0,
                        age_days=1)  # fails but is NEW
        res = self.lib.prune(min_uses=5, max_success_rate=0.34,
                             min_age_days=7)
        self.assertEqual(res["pruned"], [])

    def test_pruned_excluded_from_list_and_recall(self):
        self._skill("dead-strategy", uses=8, successes=1)
        self._skill("good-strategy", uses=8, successes=7)
        self.lib.prune()
        names = [s.name for s in self.lib.list()]
        self.assertNotIn("dead-strategy", names)
        self.assertIn("good-strategy", names)
        rec = [s.name for s, _ in self.lib.recall("strategy")]
        self.assertNotIn("dead-strategy", rec)

    def test_pruned_visible_with_flag(self):
        self._skill("dead-strategy", uses=8, successes=1)
        self.lib.prune()
        all_names = [s.name for s in self.lib.list(include_pruned=True)]
        self.assertIn("dead-strategy", all_names)
        pruned_skill = self.lib.get_by_name("dead-strategy")
        self.assertTrue(pruned_skill.pruned)

    def test_restore_lifts_prune(self):
        s = self._skill("dead-strategy", uses=8, successes=1)
        self.lib.prune()
        self.assertNotIn("dead-strategy",
                         [x.name for x in self.lib.list()])
        restored = self.lib.restore(s.id)
        self.assertFalse(restored.pruned)
        self.assertIn("dead-strategy", [x.name for x in self.lib.list()])

    def test_improve_rescues_pruned_skill(self):
        s = self._skill("dead-strategy", uses=8, successes=1)
        self.lib.prune()
        fixed = self.lib.improve(s.id, body="revised", note="rescue")
        self.assertFalse(fixed.pruned)
        self.assertEqual(fixed.version, 2)

    def test_stats_include_prune_counts(self):
        self._skill("dead-strategy", uses=8, successes=1)
        self._skill("good-strategy", uses=8, successes=7)
        self.lib.prune()
        st = self.lib.stats()
        self.assertEqual(st["total"], 2)
        self.assertEqual(st["pruned"], 1)
        self.assertEqual(st["active"], 1)

    def test_prune_idempotent(self):
        self._skill("dead-strategy", uses=8, successes=1)
        first = self.lib.prune()
        second = self.lib.prune()
        self.assertEqual(len(first["pruned"]), 1)
        self.assertEqual(len(second["pruned"]), 0)
        self.assertEqual(second["pruned_total"], 1)


# ═══════════════════════════════════════════════════════════════════════════
# migration 28 shape
# ═══════════════════════════════════════════════════════════════════════════

class TestMigration28(unittest.TestCase):
    def test_new_tables_and_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = _ctx(tmp)
            db = ctx.db
            cols = {r["name"] for r in db.query("PRAGMA table_info(skills)")}
            self.assertIn("pruned", cols)
            self.assertIn("pruned_at", cols)
            mcols = {r["name"] for r in
                     db.query("PRAGMA table_info(monitors)")}
            self.assertIn("volatile", mcols)
            db.execute(
                "INSERT INTO corpus_words (word, source, ts) "
                "VALUES ('migrated', 'test', 1.0)")
            row = db.query_one(
                "SELECT word FROM corpus_words WHERE word='migrated'")
            self.assertIsNotNone(row)


if __name__ == "__main__":
    unittest.main()
