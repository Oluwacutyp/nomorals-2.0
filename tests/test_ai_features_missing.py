"""Tests for the AI_FEATURES_MISSING build.

Every module is tested through its real code path (no mocks of the
module under test).  Judge tests use a fake ``ask`` callable that picks
by candidate CONTENT, so the judge's swapped-order pairwise calls stay
consistent — the same contract the real judge relies on.
"""

import json
import math
import sqlite3

import pytest

# ── helpers ─────────────────────────────────────────────────────────────


def _fake_judge_ask(messages, params, **kw):
    """Judge ``ask``: picks the candidate containing the GOOD marker."""
    from nomorals.llm.base import LLMResponse

    user_text = ""
    for msg in messages:
        content = getattr(msg, "content", "") or ""
        if "candidate 1" in content:
            user_text = content
    blocks = {}
    current = None
    for line in user_text.splitlines():
        if line.startswith("--- candidate"):
            try:
                current = int(line.split()[2])
            except (IndexError, ValueError):
                current = None
            blocks[current] = []
        elif current is not None:
            blocks[current].append(line)
    winner = 1
    for num, lines in blocks.items():
        # First non-empty line only: trailing instruction lines
        # ("Pick exactly ONE winner…") belong to no candidate.
        first = next((ln.strip() for ln in lines if ln.strip()), "")
        if "GOOD" in first or first == "bonjour monde":
            winner = num
    ranking = [winner] + [n for n in sorted(blocks) if n != winner]
    return LLMResponse(
        text=json.dumps({"winner": winner, "ranking": ranking,
                         "rationale": "content match", "confidence": 0.9}))


def _pref_pairs(n=15, reverse=False):
    from nomorals.training.preference import PreferencePair

    pairs = []
    for i in range(n):
        chosen = "concise summary cite source verified"
        rejected = "verbose guess no citation rambling"
        if reverse:
            chosen, rejected = rejected, chosen
        pairs.append(PreferencePair(
            prompt=f"act {i}", chosen=chosen, rejected=rejected,
            source="owner_correction", weight=2.0, action_type="message",
            created_at=1000.0 + i))
    return pairs


# ── 1. preference ───────────────────────────────────────────────────────


class TestPreference:
    def test_build_pairs_from_ledger_disapproval_weight(self):
        from nomorals.training.preference import (
            DISAPPROVAL_WEIGHT, build_preference_pairs)

        class Rec:
            def __init__(self, **kw):
                self.__dict__.update(kw)

        class Ledger:
            def recent(self, limit=500, since_days=90.0):
                return [
                    Rec(id="a1", description="sent verbose guess",
                        outcome="success", owner_feedback=-1,
                        feedback_text="use concise summary cite source",
                        action_type="message", created_at=100.0),
                    Rec(id="a2", description="sent fine note",
                        outcome="success", owner_feedback=1,
                        feedback_text="", action_type="message",
                        created_at=101.0),
                ]

            def _counterfactuals_for(self, _id):
                return []

        pairs, stats = build_preference_pairs(Ledger())
        assert len(pairs) == 1
        assert pairs[0].chosen == "use concise summary cite source"
        assert pairs[0].rejected == "sent verbose guess"
        assert pairs[0].weight == DISAPPROVAL_WEIGHT == 2.0
        assert stats["approved_skipped"] == 1  # approved: no rejected side

    def test_held_out_split_is_chronological(self):
        from nomorals.training.preference import split_held_out

        pairs = _pref_pairs(10)
        train, held = split_held_out(pairs, held_out_fraction=0.2)
        assert len(held) == 2
        assert all(h.created_at > t.created_at for h in held for t in train)

    def test_gate_passes_good_adapter(self):
        from nomorals.training.preference import (
            evaluate_on_held_out, split_held_out, train_preference_adapter)

        pairs = _pref_pairs(15)
        train, held = split_held_out(pairs)
        adapter = train_preference_adapter(train, base_model="test-base")
        report = evaluate_on_held_out(adapter, held)
        assert report["passed"] is True
        assert report["adapter_win_rate"] == 1.0

    def test_gate_rejects_bad_adapter(self):
        """The loop MUST reject its own update when held-out fails."""
        from nomorals.training.preference import (
            production_gate, split_held_out, train_preference_adapter)

        pairs = _pref_pairs(15)
        train, held = split_held_out(pairs)
        bad = train_preference_adapter(_pref_pairs(12, reverse=True))
        passed, report = production_gate(bad, held, train_pairs=train)
        assert passed is False
        assert report["reasons"]

    def test_gate_rejects_too_small_held_out(self):
        from nomorals.training.preference import (
            PreferenceAdapter, production_gate)

        passed, report = production_gate(
            PreferenceAdapter.empty(), _pref_pairs(2))
        assert passed is False
        assert any("too small" in r for r in report["reasons"])

    def test_dpo_loss_improves_with_training(self):
        from nomorals.training.preference import (
            PreferenceAdapter, pairwise_loss, train_preference_adapter)

        pairs = _pref_pairs(10)
        before = pairwise_loss(pairs, PreferenceAdapter.empty())
        after = pairwise_loss(pairs, train_preference_adapter(pairs))
        assert after < before

    def test_adapter_save_load_round_trip(self, tmp_path):
        from nomorals.training.preference import (
            PreferenceAdapter, train_preference_adapter)

        adapter = train_preference_adapter(_pref_pairs(5))
        path = adapter.save(tmp_path)
        loaded = PreferenceAdapter.load(path)
        assert loaded.token_weights == adapter.token_weights
        assert loaded.pair_count == 5
        assert loaded.lora_config()["peft_type"] == "PREFERENCE_DELTA"

    def test_brain_serves_adapter_lessons(self):
        from nomorals.llm.brain import Brain
        from nomorals.training.preference import train_preference_adapter

        brain = Brain()
        assert brain.preference_lessons() == []
        adapter = train_preference_adapter(
            _pref_pairs(3), lessons=["LESSON — message: be concise"])
        brain.set_preference_adapter(adapter)
        assert brain.preference_lessons("message") == [
            "LESSON — message: be concise"]


# ── 2. distillation ─────────────────────────────────────────────────────


class TestDistillation:
    def test_capture_kept_passed_and_empty(self):
        from nomorals.training.distillation import (
            capture_owner_kept, capture_owner_passed)

        kept = capture_owner_kept("q", "ctx", "a", teacher="cloud")
        passed = capture_owner_passed("q", "ctx", "a", teacher="cloud")
        assert kept is not None and kept.source == "kept"
        assert passed is not None and passed.weight > kept.weight
        assert capture_owner_kept("q", "ctx", "   ") is None

    def test_store_dedupes_and_exports(self, tmp_path):
        from nomorals.training.distillation import (
            DistillationStore, capture_owner_kept)

        store = DistillationStore(tmp_path / "pairs.jsonl")
        pair = capture_owner_kept("q", "", "a")
        assert store.add(pair) is True
        assert store.add(pair) is False  # exact duplicate
        assert len(store) == 1
        exported = store.export_jsonl(tmp_path / "out.jsonl")
        assert json.loads(exported.read_text().splitlines()[0])["source"] == "kept"

    def test_judge_and_capture_picks_winner(self):
        from nomorals.training.distillation import judge_and_capture

        pair, info = judge_and_capture(
            "question?",
            ["BAD vague answer", "GOOD precise answer with evidence"],
            _fake_judge_ask)
        assert pair is not None
        assert "GOOD" in pair.answer
        assert pair.source == "judge"
        assert info["ok"] is True

    def test_judge_and_capture_no_candidates(self):
        from nomorals.training.distillation import judge_and_capture

        pair, info = judge_and_capture("q", [], _fake_judge_ask)
        assert pair is None and info["ok"] is False

    def test_periodic_distill_defers_below_threshold(self):
        from nomorals.training.distillation import (
            DistillationStore, capture_owner_kept, periodic_distill)

        store = DistillationStore()
        store.add(capture_owner_kept("q", "", "a"))
        result = periodic_distill(store, min_pairs=32)
        assert result["status"] == "deferred"

    def test_train_local_adapter_via_real_pipeline(self, tmp_path):
        from nomorals.training.distillation import (
            capture_owner_kept, train_local_adapter)

        pairs = [capture_owner_kept(f"question {i}?", "", f"answer {i} word")
                 for i in range(4)]
        model, metrics, info = train_local_adapter(
            pairs, out_dir=tmp_path / "ckpt")
        assert metrics.examples == 4
        assert info["checkpoint_valid"] is True


# ── 3. continual ────────────────────────────────────────────────────────


class TestContinual:
    def test_replay_buffer_mixes_old_and_new(self):
        from nomorals.training.continual import ReplayBuffer

        buf = ReplayBuffer(capacity=100, replay_ratio=0.5)
        for i in range(60):
            buf.add(f"item-{i}")
        batch = buf.sample(20, seed=1, recent_n=10)
        assert len(batch) == 20
        assert any(b.startswith("item-") and int(b.split("-")[1]) < 50
                   for b in batch)  # old items replayed
        assert any(int(b.split("-")[1]) >= 50 for b in batch)

    def test_harness_fingerprint_pins_eval_set(self):
        from nomorals.training.continual import FixedEvalHarness

        pairs = _pref_pairs(4)
        harness = FixedEvalHarness(pairs)
        assert harness.check_fingerprint(harness.fingerprint) is True
        assert harness.check_fingerprint("deadbeef") is False
        assert harness.evaluate(lambda p, c: 1.0 if "concise" in c else 0.0)[
            "accuracy"] == 1.0

    def test_confined_update_clips_per_token_step(self):
        from nomorals.training.continual import confined_update

        pairs = _pref_pairs(1)
        pairs[0].weight = 100.0  # huge batch weight must not swing adapter
        new = confined_update({}, pairs, lr=1.0, max_step=1.0, shrink=0.0)
        assert max(abs(v) for v in new.values()) <= 1.0 + 1e-9

    def test_scheduled_update_accepts_improving_batch(self):
        from nomorals.training.continual import (
            ContinualUpdater, FixedEvalHarness, ReplayBuffer)
        from nomorals.training.preference import (
            PreferenceAdapter, train_preference_adapter)

        harness = FixedEvalHarness(_pref_pairs(4))
        updater = ContinualUpdater(harness=harness, buffer=ReplayBuffer(),
                                   interval_hours=0, min_batch=2)
        adapter = PreferenceAdapter.empty()
        updater.ingest(_pref_pairs(8))
        result = updater.scheduled_update(adapter, force=True, seed=0)
        assert result["status"] == "accepted"
        assert result["after"]["accuracy"] >= result["before"]["accuracy"]
        assert result["candidate"].pair_count > adapter.pair_count
        # Caller adapter untouched — candidate is a new object.
        assert adapter.token_weights == {}

    def test_scheduled_update_rejects_forgetting_batch(self):
        from nomorals.training.continual import (
            ContinualUpdater, FixedEvalHarness, ReplayBuffer)
        from nomorals.training.preference import train_preference_adapter

        harness = FixedEvalHarness(_pref_pairs(4))
        updater = ContinualUpdater(harness=harness, buffer=ReplayBuffer(),
                                   interval_hours=0, min_batch=2)
        # Weak adapter (1 pair): one strongly-opposed batch can flip it.
        adapter = train_preference_adapter(_pref_pairs(1))
        assert harness.evaluate(
            lambda p, c: adapter.score(p, c))["accuracy"] == 1.0
        # A batch teaching the OPPOSITE of the harness must be rejected.
        # shrink=0.6 lets one strongly-opposed batch pull the adapter
        # across zero (default shrink 0.05 would — by design — hold it).
        updater.ingest(_pref_pairs(8, reverse=True))
        result = updater.scheduled_update(adapter, force=True, seed=0,
                                          lr=1.0, shrink=0.6)
        assert result["status"] == "rejected"
        assert "forgetting" in result["reason"]

    def test_scheduled_update_not_due(self):
        from nomorals.training.continual import (
            ContinualUpdater, FixedEvalHarness)
        from nomorals.training.preference import PreferenceAdapter

        harness = FixedEvalHarness(_pref_pairs(2))
        updater = ContinualUpdater(harness=harness, interval_hours=24,
                                   last_run_at=10_000.0)
        result = updater.scheduled_update(PreferenceAdapter.empty(),
                                          now=10_100.0)
        assert result["status"] == "not_due"


# ── 4a. calibrated abstention ───────────────────────────────────────────


class TestAbstention:
    def test_calibrate_matches_research_weights(self):
        from nomorals.llm.abstention import calibrate_confidence

        assert calibrate_confidence(1.0, 1.0, source_quality=1.0) == 1.0
        assert calibrate_confidence(0.0, 0.0, source_quality=0.0) == 0.0
        # 0.45*1 + 0.35*0 + 0.20*0.8 = 0.61
        assert calibrate_confidence(1.0, 0.0) == 0.61

    def test_low_confidence_abstains_with_i_dont_know(self):
        from nomorals.llm.abstention import CalibratedAbstention

        gate = CalibratedAbstention()
        text, decision = gate.wrap("a guess", 0.3, evidence_count=2)
        assert decision.answered is False
        assert "I don't know" in text

    def test_no_evidence_abstains_even_when_confident(self):
        from nomorals.llm.abstention import CalibratedAbstention

        gate = CalibratedAbstention()
        _text, decision = gate.wrap("confident claim", 0.95, evidence_count=0)
        assert decision.answered is False
        assert "no evidence" in decision.reason

    def test_high_confidence_with_evidence_answers(self):
        from nomorals.llm.abstention import CalibratedAbstention

        gate = CalibratedAbstention()
        text, decision = gate.wrap("grounded answer", 0.9, evidence_count=3)
        assert decision.answered is True
        assert text == "grounded answer"

    def test_abstention_rate_is_measurable(self):
        from nomorals.llm.abstention import (
            CalibratedAbstention, abstention_rate)

        gate = CalibratedAbstention()
        gate.wrap("a", 0.9, evidence_count=1)
        gate.wrap("b", 0.1, evidence_count=1)
        assert gate.rate() == 0.5
        assert abstention_rate([]) == 0.0

    def test_threshold_rises_with_overconfidence(self):
        from nomorals.cognition.calibration import CalibrationStore
        from nomorals.llm.abstention import threshold_from_calibration

        store = CalibrationStore()
        for _ in range(5):
            store.record("code", "m1", 0.9, False)
        raised = threshold_from_calibration(store, "code", "m1")
        assert raised > 0.55
        assert threshold_from_calibration(store, "unseen", "m1") == 0.55


# ── 4b. counterfactual simulation ───────────────────────────────────────


class TestCounterfactual:
    def _graph(self):
        from nomorals.planning.graph import WorldGraph

        graph = WorldGraph(":memory:")
        a = graph.add_node("project", "Launch")
        b = graph.add_node("project", "Website")
        c = graph.add_node("project", "Ads")
        graph.add_edge(b.node_id, a.node_id, "depends_on")
        graph.add_edge(c.node_id, b.node_id, "depends_on")
        return graph, a, b, c

    def test_simulate_counterfactual_blast_radius(self):
        from nomorals.planning.counterfactual import (
            DecisionOption, simulate_counterfactual)

        graph, a, _b, _c = self._graph()
        result = simulate_counterfactual(
            graph, DecisionOption(node_id=a.node_id, label="Launch",
                                  new_state="delayed"))
        assert result.ok is True
        assert result.total_affected == 2  # Website + Ads depend on Launch
        assert "Website" in result.affected_labels

    def test_compare_options_ranks_least_disruptive_first(self):
        from nomorals.planning.counterfactual import (
            DecisionOption, compare_options, recommend_option)

        graph, a, _b, c = self._graph()
        options = [
            DecisionOption(node_id=a.node_id, label="Launch",
                           new_state="delayed"),
            DecisionOption(node_id=c.node_id, label="Ads",
                           new_state="paused"),
        ]
        results = compare_options(graph, options)
        assert results[0].option.label == "Ads"  # blast 0 < blast 2
        rec = recommend_option(graph, options)
        assert rec["recommendation"]["label"] == "Ads"

    def test_unknown_node_is_reported_not_dropped(self):
        from nomorals.planning.counterfactual import (
            DecisionOption, simulate_counterfactual)

        graph, _a, _b, _c = self._graph()
        result = simulate_counterfactual(
            graph, DecisionOption(node_id="nope", label="Ghost"))
        assert result.ok is False
        assert "not found" in result.reason


# ── 4c. uncertainty-driven learning ─────────────────────────────────────


class TestModelWeakness:
    def test_worst_slice_ranked_first(self):
        from nomorals.autonomy.model_weakness import learning_targets
        from nomorals.cognition.calibration import CalibrationStore

        store = CalibrationStore()
        for _ in range(5):  # confident + always wrong = worst
            store.record("code", "m1", 0.9, False)
        for _ in range(5):  # honest + always right = best
            store.record("chat", "m1", 0.8, True)
        targets = learning_targets(store)
        assert targets[0]["task"] == "code"
        assert targets[0]["uncertainty"] > targets[1]["uncertainty"]
        assert "collect targeted data" in targets[0]["suggested_action"]

    def test_model_weakness_filed_in_shared_ledger(self):
        from nomorals.autonomy.model_weakness import record_model_weakness

        db = sqlite3.connect(":memory:")
        wid = record_model_weakness(db, task="code", model="m1",
                                    evidence={"uncertainty": 1.4})
        assert wid
        row = db.execute(
            "SELECT subject FROM weaknesses WHERE id = ?", (wid,)).fetchone()
        assert row[0] == "model:code:m1"


# ── 4d. vision-language fusion ──────────────────────────────────────────


class TestFusion:
    def test_classify_invoice_vs_contract(self):
        from nomorals.vision.fusion import classify_document_kind

        invoice = classify_document_kind(
            "INVOICE No 42 — Amount due $1,200. Subtotal $1,000. Tax $200.")
        assert invoice["kind"] == "invoice"
        contract = classify_document_kind(
            "AGREEMENT — Whereas the parties hereby agree to the "
            "terms and conditions herein.")
        assert contract["kind"] == "contract"
        assert classify_document_kind("a photo of a dog")["kind"] == "other"

    def test_grounded_claims_flag_hallucinated_numbers(self):
        from nomorals.vision.fusion import grounded_claims

        claims = grounded_claims(
            "The total is $1,200. The total is $9,999.",
            "INVOICE total $1,200")
        assert claims[0]["grounded"] is True
        assert claims[1]["grounded"] is False
        assert "9,999" in claims[1]["missing_numbers"]

    def test_fuse_with_model_grounds_answer(self):
        from nomorals.vision.fusion import fuse_image_text

        report = {"metadata": {"format": "PNG", "width": 10, "height": 10},
                  "ocr": {"text": "INVOICE total $1,200"},
                  "colors": {}, "quality": {}}
        result = fuse_image_text(
            b"fake-bytes", "What is the total?",
            native_report=report,
            describe_fn=lambda prompt, img: "The total is $1,200.")
        assert result.document_kind == "invoice"
        assert result.model_used is True
        assert result.ungrounded == []

    def test_fuse_without_model_is_honest(self):
        from nomorals.vision.fusion import fuse_image_text

        report = {"metadata": {}, "ocr": {"text": ""},
                  "colors": {}, "quality": {}}
        result = fuse_image_text(b"fake-bytes", "What is this?",
                                  native_report=report)
        assert result.model_used is False
        assert result.model_answer == ""


# ── 4e. speculative decoding + KV ───────────────────────────────────────


class TestSpeculative:
    def test_expected_tokens_boundaries(self):
        from nomorals.llm.speculative import expected_tokens_per_step

        assert expected_tokens_per_step(0.0, 4) == 1.0
        assert expected_tokens_per_step(1.0, 4) == 5.0
        assert expected_tokens_per_step(0.5, 1) == 1.5

    def test_estimate_speedup_worth_flag(self):
        from nomorals.llm.speculative import estimate_speedup

        good = estimate_speedup(0.8, 4, draft_cost_ratio=0.15)
        assert good["worth_enabling"] is True
        assert good["estimated_speedup"] > 1.0
        bad = estimate_speedup(0.0, 4, draft_cost_ratio=0.15)
        assert bad["worth_enabling"] is False

    def test_draft_equal_target_rejected(self):
        from nomorals.llm.speculative import SpeculativeConfig

        cfg = SpeculativeConfig(enabled=True, draft_model="same.gguf",
                                target_model="same.gguf")
        assert cfg.validate()[0] is False
        assert cfg.to_server_args() == []  # invalid -> no speculative flags

    def test_local_server_emits_draft_flags(self):
        from nomorals.llm.local_server import GGUFServerManager

        mgr = GGUFServerManager(port=18081, draft_model="draft.gguf",
                                draft_max=3)
        args = mgr.build_args("target.gguf", "llama-server")
        assert "--draft-model" in args
        assert args[args.index("--draft-max") + 1] == "3"
        # Draft == target: no flags (pure overhead).
        mgr2 = GGUFServerManager(port=18082, draft_model="target.gguf")
        assert "--draft-model" not in mgr2.build_args("target.gguf",
                                                       "llama-server")
        # Off by default.
        mgr3 = GGUFServerManager(port=18083)
        assert "--draft-model" not in mgr3.build_args("target.gguf",
                                                      "llama-server")

    def test_kv_cache_args(self):
        from nomorals.llm.speculative import kv_cache_args

        assert kv_cache_args("q8_0") == ["--ctk", "q8_0", "--ctv", "q8_0"]
        assert kv_cache_args(None) == []


# ── 4f. diarized transcript ─────────────────────────────────────────────


class TestDiarizedTranscript:
    def test_attach_speakers_by_midpoint(self):
        from nomorals.voice.diarized_transcript import attach_speakers

        segments = [{"start": 0.0, "end": 2.0, "text": "hello"},
                    {"start": 2.0, "end": 4.0, "text": "hi back"}]
        diar = [{"start": 0.0, "end": 2.5, "speaker": "Ada"},
                {"start": 2.5, "end": 5.0, "speaker": "Bo"}]
        attached = attach_speakers(segments, diar)
        assert attached[0].speaker == "Ada"
        assert attached[1].speaker == "Bo"

    def test_attach_merges_adjacent_same_speaker(self):
        from nomorals.voice.diarized_transcript import attach_speakers

        segments = [{"start": 0.0, "end": 1.0, "text": "one"},
                    {"start": 1.0, "end": 2.0, "text": "two"}]
        diar = [{"start": 0.0, "end": 3.0, "speaker": "Ada"}]
        attached = attach_speakers(segments, diar)
        assert len(attached) == 1
        assert attached[0].text == "one two"

    def test_editor_rename_edit_merge_split(self):
        from nomorals.voice.diarized_transcript import (
            DiarizedSegment, DiarizedTranscript, TranscriptEditor)

        transcript = DiarizedTranscript(segments=[
            DiarizedSegment(0.0, 2.0, "hello world", "SPEAKER_00"),
            DiarizedSegment(2.0, 4.0, "hi there", "SPEAKER_01"),
        ])
        editor = TranscriptEditor(transcript)
        assert editor.rename_speaker("SPEAKER_00", "Ada") == 1
        assert editor.edit_text(0, "hello brave world") is True
        assert editor.reassign_speaker(1, "Ada") is True
        assert editor.merge_segments(1) is True
        assert len(editor.transcript.segments) == 1
        assert editor.split_segment(0, 2.0) is True
        assert len(editor.transcript.segments) == 2
        assert len(editor.history) >= 4
        assert "Ada:" in editor.export("text")
        assert "-->" in editor.export("srt")

    def test_diarize_and_transcribe_with_fakes(self):
        from nomorals.voice.diarized_transcript import diarize_and_transcribe

        def fake_transcribe(path, language="en"):
            return {"text": "hello hi",
                    "segments": [{"start": 0.0, "end": 1.0, "text": "hello"},
                                 {"start": 1.0, "end": 2.0, "text": "hi"}]}

        def fake_diarize(path, speaker_names=None):
            return {"segments": [
                {"start": 0.0, "end": 1.0, "speaker": "Ada"},
                {"start": 1.0, "end": 2.0, "speaker": "Bo"}]}

        result = diarize_and_transcribe("clip.wav",
                                        transcriber=fake_transcribe,
                                        diarizer=fake_diarize)
        assert result.speakers == ["Ada", "Bo"]
        assert result.to_text() == "Ada: hello\nBo: hi"


# ── 4g. translation quality bridge ──────────────────────────────────────


class TestTranslationQuality:
    def test_token_f1(self):
        from nomorals.llm.translation_quality import token_f1

        assert token_f1("hello world", "hello world") == 1.0
        assert token_f1("hello world", "goodbye moon") == 0.0

    def test_round_trip_passes(self):
        from nomorals.llm.translation_quality import translate_with_quality

        mapping = {("hello world", "en", "xx"): "bonjour monde",
                   ("bonjour monde", "xx", "en"): "hello world"}

        def fake_translate(text, src, tgt):
            return mapping[(text, src, tgt)]

        result = translate_with_quality(
            "hello world", source_lang="en", target_lang="xx",
            translate_fn=fake_translate)
        assert result.translated == "bonjour monde"
        assert result.back_translation_f1 == 1.0
        assert result.passed is True

    def test_bad_round_trip_fails(self):
        from nomorals.llm.translation_quality import translate_with_quality

        def fake_translate(text, src, tgt):
            if (src, tgt) == ("en", "xx"):
                return "nonsense words here"
            return "completely different meaning"

        result = translate_with_quality(
            "hello world", source_lang="en", target_lang="xx",
            translate_fn=fake_translate, threshold=0.5)
        assert result.passed is False

    def test_empty_source_never_passes(self):
        from nomorals.llm.translation_quality import translate_with_quality

        result = translate_with_quality(
            "  ", source_lang="en", target_lang="xx",
            translate_fn=lambda t, s, d: "x")
        assert result.passed is False

    def test_llm_judge_fn_prefers_real_translation(self):
        from nomorals.llm.translation_quality import (
            judge_with_llm_judge, translate_with_quality)

        mapping = {("hello world", "en", "xx"): "bonjour monde",
                   ("bonjour monde", "xx", "en"): "hello world"}

        def fake_translate(text, src, tgt):
            return mapping[(text, src, tgt)]

        result = translate_with_quality(
            "hello world", source_lang="en", target_lang="xx",
            translate_fn=fake_translate,
            judge_fn=judge_with_llm_judge(_fake_judge_ask))
        assert result.judge_score is not None
        assert result.judge_score >= 0.7
        assert result.passed is True


# ── 4h. audio understanding ─────────────────────────────────────────────


class TestAudioUnderstanding:
    def test_energy_curve_normalized(self):
        from nomorals.audio.understanding import energy_curve

        curve = energy_curve([0.5] * 8000, 8000, window_s=1.0)
        assert curve and all(v == 1.0 for v in curve)
        assert energy_curve([0.0] * 64000, 8000) == [0.0] * 8

    def test_detect_sections_finds_silence(self):
        from nomorals.audio.understanding import detect_sections

        sections = detect_sections([0.9, 0.9, 0.02, 0.02, 0.8])
        labels = [s.label for s in sections]
        assert "silence" in labels
        assert sections[0].start == 0.0

    def test_infer_mood_table(self):
        from nomorals.audio.understanding import infer_mood

        mood, _ = infer_mood({"music_score": 0.2, "tempo_bpm": 0.0,
                              "spectral_centroid_hz": 0.0, "rms_db": -30.0})
        assert mood == "speech"
        mood, _ = infer_mood({"music_score": 0.9, "tempo_bpm": 140.0,
                              "spectral_centroid_hz": 3000.0, "rms_db": -10.0,
                              "key": "C"})
        assert mood == "tense"
        mood, _ = infer_mood({"music_score": 0.9, "tempo_bpm": 60.0,
                              "spectral_centroid_hz": 800.0, "rms_db": -30.0,
                              "key": "Am"})
        assert mood == "melancholic"

    def test_understand_audio_with_injected_analysis(self):
        from nomorals.audio.understanding import understand_audio

        analysis = {"ok": True, "seconds": 5.0, "music_score": 0.9,
                    "tempo_bpm": 140.0, "key": "C", "rms_db": -10.0,
                    "spectral_centroid_hz": 3000.0}
        result = understand_audio("nonexistent.wav", analysis=analysis)
        assert result.ok is True
        assert result.kind == "music"
        assert result.mood == "tense"
        assert "tense" in result.describe()


# ── naming artifacts (verified, not rebuilt) ────────────────────────────


class TestNamingArtifactsExist:
    """The task's naming map, verified against the real modules."""

    def test_tool_loop_is_text_based_react(self):
        from nomorals.agents.partner import tool_loop

        assert hasattr(tool_loop, "ToolLoopResult")

    def test_skill_evolution_is_continual(self):
        from nomorals.agents import skill_evolution

        assert hasattr(skill_evolution, "SkillEvolutionLoop")

    def test_reasoning_has_cot_and_critique(self):
        from nomorals.agents import reasoning

        assert hasattr(reasoning, "ReasoningEngine")
        assert hasattr(reasoning.ReasoningEngine, "_critique")

    def test_adjudicate_is_grader(self):
        import importlib

        adjudicate_module = importlib.import_module(
            "nomorals.llm.adjudicate")
        assert hasattr(adjudicate_module, "Judge")
        assert hasattr(adjudicate_module.Judge, "adjudicate")

    def test_tool_params_listing_fix_present(self):
        from nomorals.tools.registry import _format_params

        schema = {"parameters": {
            "type": "object",
            "properties": {"q": {"type": "string"},
                           "limit": {"type": "integer", "default": 10}},
            "required": ["q"]}}
        assert _format_params(schema) == "q, limit=10"
