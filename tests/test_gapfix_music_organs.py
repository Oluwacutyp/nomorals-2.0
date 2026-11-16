"""Gapfix R2-24: Tier-2/3 music organs wired, not dormant.

fidelity.route() into the gen pipelines, critique_loop + check_consistency
from the song-draft path, and the three register() hooks (sonic_bible,
music_critique, genre_lora) on the tools/registry.py media list.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

# ── fixtures: a clean draft and a slop draft ──────────────────────────

def _line(text, contour):
    from nomorals.media.song_draft import DraftLine
    return DraftLine(text=text, syllables=len(text.split()),
                     contour=list(contour))


def _clean_draft():
    from nomorals.media.song_draft import DraftSection, SongDraft
    return SongDraft(
        title="Night Run", topic="night", style="pop", tempo=104, key="C",
        sections=[
            DraftSection(name="verse 1", bars=8, energy=0.4, lines=[
                _line("city lights are calling me",
                      ["hold", "up", "hold", "down", "hold", "up", "hold"]),
            ]),
            DraftSection(name="chorus", bars=8, energy=0.9, lines=[
                _line("we run all night long",
                      ["up", "up", "hold", "down", "hold"]),
            ]),
        ])


def _slop_draft():
    """Yoruba lyrics whose falling contour contradicts the rising tones —
    the exact slop the critique loop exists to catch."""
    from nomorals.media.song_draft import DraftSection, SongDraft
    return SongDraft(
        title="Owo", topic="owo", style="afrobeats", tempo=100, key="C",
        sections=[
            DraftSection(name="verse 1", bars=4, energy=0.4, lines=[
                _line("owó o", ["down", "down", "down"]),
            ]),
            DraftSection(name="chorus", bars=4, energy=0.9, lines=[
                _line("owó owó", ["down", "down", "down", "down"]),
            ]),
        ])


def _registry(tmp_path):
    from nomorals.tools.registry import ToolRegistry
    ctx = SimpleNamespace(
        db=None,
        settings=SimpleNamespace(workspace_dir=str(tmp_path),
                                 home_path=tmp_path),
        extras={}, router=None, tools=None)
    registry = ToolRegistry(ctx)
    ctx.tools = registry
    return ctx, registry


def _bible_dict():
    return {"artist": "Test", "album": "One", "keys": ["C", "G"],
            "mode": "any", "bpm_range": [95, 110], "genres": ["afrobeats"],
            "energy_range": [0.3, 0.9], "palette": ["log drum", "shaker"],
            "languages": ["yo", "en"]}


# ── 1. register() hooks on the media list ─────────────────────────────

class TestRegisterHooks:
    def test_sonic_bible_tool_registered(self, tmp_path):
        _, registry = _registry(tmp_path)
        registry.register_builtins()
        assert registry.get("sonic_bible") is not None
        assert "nomorals.media.sonic_bible" not in registry.failed_modules()

    def test_music_critique_tool_registered(self, tmp_path):
        _, registry = _registry(tmp_path)
        registry.register_builtins()
        assert registry.get("music_critique") is not None
        assert "nomorals.media.music_critique" not in registry.failed_modules()

    def test_genre_lora_tool_registered(self, tmp_path):
        _, registry = _registry(tmp_path)
        registry.register_builtins()
        assert registry.get("genre_lora") is not None
        assert "nomorals.media.lora_global" not in registry.failed_modules()


class TestToolsCallable:
    def test_sonic_bible_tool_check(self, tmp_path):
        _, registry = _registry(tmp_path)
        registry.register_builtins()
        out = registry.call(
            "sonic_bible", action="check", bible=json.dumps(_bible_dict()),
            track=json.dumps({"title": "t1", "key": "C", "bpm": 100,
                              "genre": "afrobeats",
                              "palette": ["log drum", "shaker"],
                              "language": "yo"}))
        assert out.ok, out
        assert out.value["consistent"] is True

    def test_music_critique_tool_loop_fixes_tone(self, tmp_path):
        _, registry = _registry(tmp_path)
        registry.register_builtins()
        sections = [{"name": "verse", "lyrics": ["owó"],
                     "melody": [[64, 0.0, 0.5], [60, 0.5, 0.5]]},
                    {"name": "chorus", "lyrics": ["owó owó"],
                     "melody": [[60, 0.0, 0.5], [64, 0.5, 0.5]]}]
        out = registry.call("music_critique", action="loop", title="t",
                            sections=json.dumps(sections), language="yo")
        assert out.ok, out
        assert out.value["fixes_applied"]

    def test_genre_lora_tool_spec(self, tmp_path):
        _, registry = _registry(tmp_path)
        registry.register_builtins()
        out = registry.call("genre_lora", action="spec", genre="amapiano")
        assert out.ok, out
        assert out.value["spec"]["activation_tag"] == "amapiano-style"


# ── 2. sonic bible persistence ────────────────────────────────────────

class TestBibleStore:
    def test_save_show_roundtrip(self, tmp_path):
        from nomorals.media.sonic_bible import SonicBible, load_bible, save_bible
        path = str(tmp_path / "bible.json")
        bible = SonicBible.from_dict(_bible_dict())
        saved = save_bible(bible, path)
        assert Path(saved).is_file()
        assert load_bible(path) == bible

    def test_load_missing_bible_is_none_not_raise(self, tmp_path):
        from nomorals.media.sonic_bible import load_bible
        assert load_bible(str(tmp_path / "nope.json")) is None

    def test_tool_save_show(self, tmp_path):
        _, registry = _registry(tmp_path)
        registry.register_builtins()
        out = registry.call("sonic_bible", action="save",
                            bible=json.dumps(_bible_dict()),
                            path=str(tmp_path / "b.json"))
        assert out.ok, out
        out2 = registry.call("sonic_bible", action="show",
                             path=str(tmp_path / "b.json"))
        assert out2.ok, out2
        assert out2.value["bible"]["artist"] == "Test"


# ── 3. draft → critique sections (beats → seconds, per section) ───────

class TestDraftCritiqueSections:
    def test_sections_grouped_per_section_in_seconds(self):
        from nomorals.media.draft_perform import draft_critique_sections
        secs = draft_critique_sections(_clean_draft())
        assert [s["name"] for s in secs] == ["verse 1", "chorus"]
        verse = secs[0]
        assert verse["lyrics"] == ["city lights are calling me"]
        assert verse["melody"]
        # seconds, not beats: at 104 BPM one beat is 60/104 ≈ 0.577s, so the
        # second note (4.571 beats in) starts at ≈2.64s, not 4.57s.
        starts = [m[1] for m in verse["melody"]]
        assert starts[0] == 0.0
        assert starts[1] == pytest.approx(32 / 7 * 60 / 104, abs=0.01)
        assert verse["melody"][0][2] == pytest.approx(
            32 / 7 * 0.92 * 60 / 104, abs=0.01)
        assert all(m[0] > 0 and m[2] > 0 for m in verse["melody"])

    def test_events_unchanged_by_refactor(self):
        """draft_melody_events must produce the same notes as before."""
        from nomorals.media.draft_perform import draft_melody_events
        evs = draft_melody_events(_clean_draft())
        assert [(e.note, round(e.start, 3)) for e in evs[:3]] == [
            (60, 0.0), (62, 4.571), (62, 9.143)]


# ── 4. judge_draft: the judge that catches slop ───────────────────────

class TestJudgeDraft:
    def test_clean_draft_ships_without_bible(self, tmp_path):
        from nomorals.media.song_draft import judge_draft
        res = judge_draft(_clean_draft(),
                          bible_path=str(tmp_path / "none.json"))
        assert res["critique"]["ship"] is True
        assert res["critique"]["fixes_applied"] == []
        assert res["consistency"]["available"] is False
        assert "no sonic bible" in res["consistency"]["reason"]
        assert "ships" in res["verdict"].lower()

    def test_slop_draft_caught_and_fixed(self, tmp_path):
        from nomorals.media.song_draft import judge_draft
        res = judge_draft(_slop_draft(), language="yo")
        before = res["critique"]["before"]
        assert before["ship"] is False
        assert any("tone" in i for i in before["issues"])
        assert res["critique"]["fixes_applied"]
        assert res["critique"]["ship"] is True
        assert res["critique"]["score"] >= before["score"]
        assert "tone-fix" in res["verdict"]

    def test_consistency_against_stored_bible(self, tmp_path):
        from nomorals.media.song_draft import judge_draft
        from nomorals.media.sonic_bible import SonicBible, save_bible
        bible_path = str(tmp_path / "bible.json")
        save_bible(SonicBible.from_dict(_bible_dict()), bible_path)
        good = judge_draft(_clean_draft(), bible_path=bible_path)
        # clean draft is pop/C/104bpm vs an afrobeats bible: genre drifts,
        # key/bpm do not — the verdict names the drift honestly.
        assert good["consistency"]["available"] is True
        assert "genre" in good["consistency"]["drift"]
        assert "key" not in good["consistency"]["drift"]
        assert "palette" in good["consistency"]["unscored"]
        bad = judge_draft(_slop_draft(), bible_path=bible_path)
        # the slop draft is consistent with the bible (same key/tempo/genre)
        # — its failing is the critique, not the bible.
        assert bad["consistency"]["consistent"] is True
        assert "bpm" not in bad["consistency"]["drift"]  # 100 in 95–110
        # a draft in the wrong key AND tempo AND genre is inconsistent
        from nomorals.media.song_draft import SongDraft
        wrong = SongDraft(title="x", topic="x", style="metal",
                          tempo=160, key="Eb",
                          sections=_slop_draft().sections)
        res = judge_draft(wrong, bible_path=bible_path)
        assert res["consistency"]["consistent"] is False
        assert "key" in res["consistency"]["drift"]
        assert "bpm" in res["consistency"]["drift"]
        assert "genre" in res["consistency"]["drift"]

    def test_corrupt_bible_never_breaks_judging(self, tmp_path):
        from nomorals.media.song_draft import judge_draft
        bad_path = tmp_path / "bad.json"
        bad_path.write_text("{not json", encoding="utf-8")
        res = judge_draft(_clean_draft(), bible_path=str(bad_path))
        assert res["critique"]["ship"] is True
        assert res["consistency"]["available"] is False
        assert res["consistency"]["reason"]


# ── 5. fidelity.route() in the gen pipeline ───────────────────────────

class TestRouteStageModels:
    def _req(self, tmp_path, **kw):
        from nomorals.media.musicgen_chain import ChainRequest
        return ChainRequest(title="t", lyrics="la la", language="en",
                            workdir=str(tmp_path), **kw)

    def test_no_backends_routes_to_no_candidate(self, tmp_path):
        from nomorals.media.musicgen_chain import route_stage_models
        routing = route_stage_models(
            self._req(tmp_path),
            bed_probe=lambda: [], vocal_probe=lambda: False,
            rvc_probe=lambda name: False)
        for task in ("bed", "vocal", "rvc"):
            assert routing[task]["model_id"] is None
            assert routing[task]["reason"]

    def test_turbo_routes_but_not_at_master_bar(self, tmp_path):
        from nomorals.media.musicgen_chain import route_stage_models
        routing = route_stage_models(
            self._req(tmp_path, quality_level="master"),
            bed_probe=lambda: ["turbo"], vocal_probe=lambda: False,
            rvc_probe=lambda name: False)
        assert routing["bed"]["model_id"] is None  # 0.6 < master bar 0.85
        routing_std = route_stage_models(
            self._req(tmp_path, quality_level="standard"),
            bed_probe=lambda: ["turbo"], vocal_probe=lambda: False,
            rvc_probe=lambda name: False)
        assert routing_std["bed"]["model_id"] == "acestep-turbo"

    def test_full_beats_turbo_when_both_available(self, tmp_path):
        from nomorals.media.musicgen_chain import route_stage_models
        routing = route_stage_models(
            self._req(tmp_path),
            bed_probe=lambda: ["turbo", "full"], vocal_probe=lambda: False,
            rvc_probe=lambda name: False)
        assert routing["bed"]["model_id"] == "acestep-full"
        assert routing["bed"]["quality"] == 0.9

    def test_rvc_routes_only_the_named_model(self, tmp_path):
        from nomorals.media.musicgen_chain import route_stage_models
        routing = route_stage_models(
            self._req(tmp_path, rvc_model="my-voice"),
            bed_probe=lambda: [], vocal_probe=lambda: False,
            rvc_probe=lambda name: name == "my-voice")
        assert routing["rvc"]["model_id"] == "rvc:my-voice"
        routing2 = route_stage_models(
            self._req(tmp_path, rvc_model="ghost"),
            bed_probe=lambda: [], vocal_probe=lambda: False,
            rvc_probe=lambda name: name == "my-voice")
        assert routing2["rvc"]["model_id"] is None

    def test_unknown_quality_level_fails_fast(self, tmp_path):
        from nomorals.media.musicgen_chain import route_stage_models
        with pytest.raises(ValueError, match="unknown quality level"):
            route_stage_models(self._req(tmp_path, quality_level="bogus"))


class TestRunChainRouting:
    def test_manifest_records_routing_with_fakes(self, tmp_path):
        import json as _json

        from nomorals.media.musicgen_chain import ChainRequest, fake_bed, fake_vocal, run_chain
        req = ChainRequest(title="t", lyrics="la la", language="en",
                           workdir=str(tmp_path))
        res = run_chain(req, bed_fn=fake_bed, vocal_fn=fake_vocal,
                        master_fn=lambda srcs, outdir, req=None: "")
        manifest = _json.loads(Path(res.manifest_path).read_text())
        routing = manifest["fidelity_routing"]
        assert routing["bed"]["note"] == "test double — not routed"
        assert routing["vocal"]["note"] == "test double — not routed"
        assert res.stage_status("bed") == "fake"

    def test_run_chain_invalid_level_raises_before_files(self, tmp_path):
        from nomorals.media.musicgen_chain import ChainRequest, run_chain
        req = ChainRequest(title="t", lyrics="la la", language="en",
                           quality_level="bogus", workdir=str(tmp_path))
        with pytest.raises(ValueError, match="unknown quality level"):
            run_chain(req)
        assert not list(Path(tmp_path).glob("*-manifest.json"))

    def test_run_chain_no_backends_skips_with_route_reason(self, tmp_path):
        from nomorals.media.musicgen_chain import ChainRequest, run_chain
        req = ChainRequest(title="t", lyrics="la la", language="en",
                           workdir=str(tmp_path))
        res = run_chain(req, bed_probe=lambda: [], vocal_probe=lambda: False,
                        rvc_probe=lambda name: False)
        assert res.ok is False
        bed = next(s for s in res.stages if s.name == "bed")
        assert bed.status == "skipped"
        assert "no candidate" in bed.detail


# ── 6. genre_lora train registers the finished adapter ────────────────

class TestGenreLoraTrainRegisters:
    def test_train_registers_adapter_on_success(self, tmp_path, monkeypatch):
        import nomorals.media.lora_global as lg
        from nomorals.media.afrobeats import AfrobeatsLoRA
        out_dir = tmp_path / "lora-amapiano"
        (out_dir / "lora").mkdir(parents=True)
        (out_dir / "lora" / "adapter_config.json").write_text("{}", encoding="utf-8")
        monkeypatch.setattr(lg.GenreLoRAJob, "run",
                            lambda self: {"ok": True, "output_dir": str(out_dir),
                                          "log": []})
        monkeypatch.setattr(lg.GenreLoRAJob, "validate", lambda self: [])
        import nomorals.media.afrobeats as ab
        reg_path = str(tmp_path / "loras.json")
        monkeypatch.setattr(ab, "DEFAULT_LORA_REGISTRY", reg_path)
        _, registry = _registry(tmp_path)
        registry.register_builtins()
        out = registry.call("genre_lora", action="train", genre="amapiano",
                            corpus_dir=str(tmp_path), dry_run=False,
                            output_dir=str(out_dir))
        assert out.ok, out
        assert out.value["registered"] is True
        assert AfrobeatsLoRA(registry_path=reg_path).registered()
