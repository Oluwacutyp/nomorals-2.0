"""Pass 6 — the Director: coverage → breakdown → design → shoot → QA.

Everything offline: the brain, the Seer, the videogen front door and
frame extraction are faked at their real boundaries (Brain.chat_json,
Seer.see, videogen.generate, cv_video.extract_frames shapes).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from nomorals.llm.base import LLMResponse
from nomorals.media.director import (
    Director, DirectorError, STYLE_PACKS, assemble_prompt, break_down,
    cover_script, design_shot, get_style_pack, parse_screenplay,
    refine_design,
)
from nomorals.media.director.design import (
    ANGLES, COLOR_TEMPS, EXPOSURE_INTENTS, FILL_RATIOS, FOCUS_INTENTS,
    KEY_DIRECTIONS, KEY_QUALITIES, KEY_SOURCES, LENS_FEELS, MOVEMENTS,
    SHOT_SIZES,
)
from nomorals.media.videogen import VideogenError

SCRIPT = """\
INT. COFFEE SHOP - DAY

MARA VOSS, nurse scrubs under a coat, stares at a letter on the table.

MARA
(quietly)
He signed it. After all these years, he signed it.

THEO
Then we start tonight.

MARA
Tonight? The ferries stop at nine.

Mara folds the letter into her pocket. The bell over the door rings.

CUT TO:

EXT. ROOFTOP - NIGHT

Mara stands at the ledge, the city glittering below her.

MARA
You ever wonder what falling feels like?

THEO
Every day since I met you.
"""

PROSE = ("Mara waited in the coffee shop until Theo arrived with the "
         "letter. That night they climbed to the rooftop and looked "
         "out over the city, neither saying what both were thinking.")


# ---------------------------------------------------------------------------
# Fakes at the real boundaries
# ---------------------------------------------------------------------------

COVERAGE_JSON = {
    "title": "The Letter", "genre": "drama",
    "logline": "A nurse learns her estranged father has signed away "
               "the family ferry line and plans one last crossing.",
    "synopsis": "Mara receives the signed letter in a coffee shop; "
                "Theo pushes her to act before the ferries stop.",
    "structure": {"acts": ["Act 1: the letter lands"],
                  "beats": ["inciting letter", "the rooftop vow"]},
    "assessments": {
        "premise": {"grade": "strong", "notes": "clean hook"},
        "character": {"grade": "adequate", "notes": "Theo is thin"},
        "dialogue": {"grade": "strong", "notes": "subtext carries it"},
        "visual": {"grade": "adequate", "notes": "two strong images"},
        "pacing": {"grade": "adequate", "notes": "tight"},
    },
    "verdict": "consider",
    "verdict_reasons": ["the hook is producible", "Theo needs a pass"],
    "strengths": ["the letter as engine"],
    "concerns": ["second act unseen in sample"],
}

SEGMENT_JSON = {
    "scenes": [
        {"location": "COFFEE SHOP", "time_of_day": "DAY",
         "interior": "INT",
         "beats": [{"kind": "action",
                    "text": "Mara waits until Theo arrives."},
                   {"kind": "dialogue", "speaker": "THEO",
                    "text": "We start tonight."}]},
        {"location": "ROOFTOP", "time_of_day": "NIGHT",
         "interior": "EXT",
         "beats": [{"kind": "action",
                    "text": "They look out over the city."}]},
    ]
}


class FakeBrain:
    """Routes chat_json by prompt marker, like the real stages ask."""

    def __init__(self, *, coverage=None, verdict=None, refine=None,
                 fail=False):
        self._coverage = coverage
        self._verdict = verdict if verdict is not None else {
            "subject_present": True, "lighting_match": True,
            "composition_ok": True, "notes": "frame matches intent"}
        self._refine = refine
        self._fail = fail
        self.prompts: list[str] = []

    def chat_json(self, messages, params=None, **kw):
        prompt = messages[-1].content
        self.prompts.append(prompt)
        if self._fail:
            return None, LLMResponse(text="", error="no providers up")
        if "script reader" in prompt:
            return self._coverage or COVERAGE_JSON, LLMResponse(text="{}")
        if "Segment this prose" in prompt:
            return SEGMENT_JSON, LLMResponse(text="{}")
        if "cinematographer refining" in prompt:
            return self._refine or {}, LLMResponse(text="{}")
        if "QA judge" in prompt:
            return self._verdict, LLMResponse(text="{}")
        return {}, LLMResponse(text="{}")


class FakeSeer:
    def __init__(self, text="A woman in a coat at a table, soft window "
                             "light from the left, medium framing."):
        self.text = text
        self.calls = 0

    def see(self, image, question=""):
        self.calls += 1
        return self.text


class BrokenSeer:
    def see(self, image, question=""):
        from nomorals.vision.seer import VisionUnavailable
        raise VisionUnavailable("no vision path on this machine")


def fake_generate_factory():
    """A videogen.generate stand-in: writes a real file at ``out``."""
    calls = []

    def generate(prompt, **kw):
        calls.append({"prompt": prompt, **kw})
        out = Path(kw["out"])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"fake-clip-bytes")
        return SimpleNamespace(path=str(out), engine="motion",
                               routed="motion",
                               note="fake generation for tests")

    generate.calls = calls
    return generate


def fake_extract(clip_path):
    src = Path(clip_path)
    p = src.with_name(src.stem + ".frame.jpg")
    p.write_bytes(b"fake-frame")
    return [str(p)]


# ---------------------------------------------------------------------------
# Screenplay parsing
# ---------------------------------------------------------------------------

def test_parse_standard_screenplay():
    play = parse_screenplay(SCRIPT, title="The Letter")
    assert play.is_standard
    assert len(play.scenes) == 2
    s1, s2 = play.scenes
    assert (s1.interior, s1.location, s1.time_of_day) == \
        ("INT", "COFFEE SHOP", "DAY")
    assert (s2.interior, s2.location, s2.time_of_day) == \
        ("EXT", "ROOFTOP", "NIGHT")
    assert play.characters == ["MARA", "THEO"]
    dlg = s1.dialogue
    assert dlg[0].character == "MARA"
    assert "He signed it" in dlg[0].text
    paren = [e for e in s1.elements if e.kind == "parenthetical"]
    assert paren and paren[0].text == "(quietly)"
    assert any(e.kind == "transition" for e in s1.elements)


def test_parse_prose_is_not_standard():
    play = parse_screenplay(PROSE)
    assert not play.is_standard
    assert play.scenes == []


# ---------------------------------------------------------------------------
# Breakdown grammar
# ---------------------------------------------------------------------------

def test_breakdown_grammar():
    play = parse_screenplay(SCRIPT)
    shots = break_down(play)
    ids = [s.id for s in shots]
    assert ids == sorted(ids, key=lambda x: (int(x.split(".")[0]),
                                             int(x.split(".")[1])))
    assert shots[0].kind == "establishing"
    assert shots[0].scene_number == 1
    dlg = [s for s in shots if s.kind == "dialogue"]
    assert dlg[0].speaker == "MARA" and "He signed it" in dlg[0].dialogue
    assert dlg[0].performance_note == "(quietly)"
    # MARA → THEO → MARA exchange earns a reaction shot of THEO
    reactions = [s for s in shots if s.kind == "reaction"]
    assert reactions and reactions[0].characters == ["THEO"]
    assert any(s.kind == "transition" for s in shots)
    # durations are plausible screen durations
    assert all(1.0 <= s.duration_s <= 9.0 for s in shots)


def test_breakdown_no_reestablish_on_continuation():
    text = ("INT. KITCHEN - DAY\n\nAnna cooks.\n\n"
            "INT. KITCHEN - DAY\n\nAnna serves the food.\n")
    shots = break_down(parse_screenplay(text))
    est = [s for s in shots if s.kind == "establishing"]
    assert len(est) == 1
    assert shots[-1].scene_number == 2


def test_breakdown_prose_needs_brain_and_uses_it(tmp_path):
    d = Director(root=tmp_path, brain=FakeBrain())
    prod = d.plan(PROSE, title="Prose Piece")
    assert len(prod.screenplay.scenes) == 2
    assert prod.shots[0].kind == "establishing"
    assert any(s.speaker == "THEO" for s in prod.shots)

    # a brain that cannot answer is an honest failure, not a guess
    d2 = Director(root=tmp_path / "x", brain=FakeBrain(fail=True))
    with pytest.raises(DirectorError):
        d2.plan(PROSE)


def test_breakdown_prose_brain_garbage_is_honest(tmp_path):
    brain = FakeBrain()
    brain._coverage = None
    orig = brain.chat_json

    def garbage(messages, params=None, **kw):
        if "Segment this prose" in messages[-1].content:
            return {"scenes": []}, LLMResponse(text="{}")
        return orig(messages, params=params, **kw)

    brain.chat_json = garbage
    d = Director(root=tmp_path, brain=brain)
    with pytest.raises(DirectorError):
        d.plan(PROSE)


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def test_coverage_contract_and_report():
    cov = cover_script(SCRIPT, brain=FakeBrain(),
                       play=parse_screenplay(SCRIPT))
    assert cov.verdict == "consider"
    assert cov.logline.startswith("A nurse")
    assert cov.assessments["dialogue"].grade == "strong"
    assert set(cov.assessments) == {"premise", "character", "dialogue",
                                     "visual", "pacing"}
    report = cov.report()
    assert "VERDICT: CONSIDER" in report
    assert "logline:" in report


def test_coverage_brain_failure_is_honest():
    with pytest.raises(DirectorError):
        cover_script(SCRIPT, brain=FakeBrain(fail=True))
    with pytest.raises(DirectorError):
        cover_script(SCRIPT, brain=None)
    bad = dict(COVERAGE_JSON, verdict="maybe")
    with pytest.raises(DirectorError):
        cover_script(SCRIPT, brain=FakeBrain(coverage=bad))


# ---------------------------------------------------------------------------
# Design: primitives, packs, refinement
# ---------------------------------------------------------------------------

def test_design_uses_only_vocabulary():
    for pack in STYLE_PACKS:
        for kind in ("establishing", "action", "dialogue", "reaction",
                     "detail", "transition"):
            spec = design_shot(kind, time_of_day="NIGHT",
                               interior="INT", style_pack=pack)
            cam, light = spec.camera, spec.lighting
            assert cam.shot_size in SHOT_SIZES
            assert cam.angle in ANGLES
            assert cam.movement in MOVEMENTS
            assert cam.lens in LENS_FEELS
            assert cam.focus in FOCUS_INTENTS
            assert light.key_source in KEY_SOURCES
            assert light.key_quality in KEY_QUALITIES
            assert light.key_direction in KEY_DIRECTIONS
            assert light.fill_ratio in FILL_RATIOS
            assert light.exposure in EXPOSURE_INTENTS
            assert light.color_temp in COLOR_TEMPS


def test_design_style_pack_biases_are_data():
    noir = design_shot("dialogue", time_of_day="DAY", interior="INT",
                       style_pack="noir")
    neutral = design_shot("dialogue", time_of_day="DAY", interior="INT",
                          style_pack="neutral")
    assert noir.lighting.fill_ratio == "1:8 low-key"
    assert noir.lighting.key_quality == "hard"
    assert neutral.lighting.fill_ratio != "1:8 low-key"
    night_neon = design_shot("action", time_of_day="NIGHT",
                             interior="EXT", style_pack="neon")
    assert night_neon.lighting.key_source == "neon signage"
    with pytest.raises(ValueError):
        get_style_pack("kubrick")


def test_refine_design_validates_against_vocab():
    base = design_shot("dialogue", time_of_day="DAY", interior="INT")
    brain = FakeBrain(refine={
        "camera": {"shot_size": "close-up", "movement": "crane",
                   "lens": "laser"},
        "lighting": {"key_source": "candlelight"},
        "mood": "dread", "reason": "tighten on her face"})
    out = refine_design(base, "MARA speaks", brain=brain)
    assert out.camera.shot_size == "close-up"
    assert out.camera.movement == "crane"
    assert out.camera.lens == base.camera.lens  # out-of-vocab ignored
    assert out.lighting.key_source == "candlelight"
    assert "tighten on her face" in out.rationale
    # a failing brain leaves the rule design untouched
    out2 = refine_design(base, "MARA speaks", brain=FakeBrain(fail=True))
    assert out2.camera.shot_size == base.camera.shot_size


def test_prompt_assembly_is_deterministic_and_complete():
    play = parse_screenplay(SCRIPT)
    shots = break_down(play)
    shot = next(s for s in shots if s.kind == "dialogue")
    shot.design = design_shot("dialogue", time_of_day="DAY",
                              interior="INT", style_pack="noir")
    p1 = assemble_prompt(shot)
    p2 = assemble_prompt(shot)
    assert p1 == p2
    assert '"He signed it' in p1
    assert "medium close-up shot" in p1
    assert "key light:" in p1
    assert "1:8 low-key" in p1


# ---------------------------------------------------------------------------
# Production: plan persistence, shoot, QA, regen
# ---------------------------------------------------------------------------

def _planned(tmp_path, brain=None):
    d = Director(root=tmp_path, brain=brain or FakeBrain())
    return d, d.plan(SCRIPT, title="The Letter", style_pack="noir")


def test_plan_persists_and_roundtrips(tmp_path):
    d, prod = _planned(tmp_path)
    plan_file = Path(prod.dir) / "plan.json"
    assert plan_file.is_file()
    assert (Path(prod.dir) / "report.md").is_file()
    loaded = d.load(prod.id)
    assert loaded.id == prod.id
    assert len(loaded.shots) == len(prod.shots)
    assert loaded.shots[0].design is not None
    assert loaded.style_pack == "noir"
    data = json.loads(plan_file.read_text())
    assert data["screenplay"]["scenes"][0]["location"] == "COFFEE SHOP"


def test_shoot_with_qa_pass(tmp_path):
    gen = fake_generate_factory()
    d = Director(root=tmp_path, brain=FakeBrain(), seer=FakeSeer(),
                 generate_fn=gen, extract_frames_fn=fake_extract)
    prod = d.plan(SCRIPT, title="The Letter")
    out = d.shoot(prod.id, shot_ids=["1.1"])
    shot = out.shot("1.1")
    assert shot.status == "qa_pass"
    assert shot.qa["subject_present"] is True
    assert Path(shot.clip_path).is_file()
    assert "clips" in shot.clip_path
    assert gen.calls and "Establish COFFEE SHOP" in gen.calls[0]["prompt"]


def test_shoot_qa_fail_then_regen_once(tmp_path):
    fail = {"subject_present": False, "lighting_match": True,
            "composition_ok": True, "notes": "subject missing"}
    brain = FakeBrain(verdict=fail)
    d = Director(root=tmp_path, brain=brain, seer=FakeSeer(),
                 generate_fn=fake_generate_factory(),
                 extract_frames_fn=fake_extract)
    prod = d.plan(SCRIPT, title="The Letter")
    out = d.shoot(prod.id, shot_ids=["1.1"])
    assert out.shot("1.1").status == "qa_fail"
    # the judge changes its mind on the regen's frames
    brain._verdict = {"subject_present": True, "lighting_match": True,
                      "composition_ok": True, "notes": "subject clear"}
    out = d.regen(prod.id, "1.1")
    shot = out.shot("1.1")
    assert shot.status == "regenerated"
    assert shot.regen_count == 1
    with pytest.raises(DirectorError):
        d.regen(prod.id, "1.1")


def test_qa_unavailable_is_never_a_pass(tmp_path):
    d = Director(root=tmp_path, brain=FakeBrain(), seer=BrokenSeer(),
                 generate_fn=fake_generate_factory(),
                 extract_frames_fn=fake_extract)
    prod = d.plan(SCRIPT, title="The Letter")
    out = d.shoot(prod.id, shot_ids=["1.1"])
    shot = out.shot("1.1")
    assert shot.status == "qa_unavailable"
    assert "vision unavailable" in shot.qa["notes"]

    d2 = Director(root=tmp_path / "b", brain=FakeBrain(), seer=FakeSeer(),
                  generate_fn=fake_generate_factory(),
                  extract_frames_fn=lambda p: [])
    prod2 = d2.plan(SCRIPT, title="The Letter")
    out2 = d2.shoot(prod2.id, shot_ids=["1.1"])
    assert out2.shot("1.1").status == "qa_unavailable"


def test_shoot_generation_failure_is_recorded_per_shot(tmp_path):
    def bad_generate(prompt, **kw):
        raise VideogenError("no CUDA GPU on this machine")

    d = Director(root=tmp_path, brain=FakeBrain(), seer=FakeSeer(),
                 generate_fn=bad_generate,
                 extract_frames_fn=fake_extract)
    prod = d.plan(SCRIPT, title="The Letter")
    out = d.shoot(prod.id, shot_ids=["1.1", "2.1"])
    assert out.shot("1.1").status == "failed"
    assert "no CUDA GPU" in out.shot("1.1").qa["notes"]
    assert "failed: 2" in d.summary(prod.id)


def test_shoot_without_qa_marks_generated(tmp_path):
    d = Director(root=tmp_path, brain=FakeBrain(),
                 generate_fn=fake_generate_factory())
    prod = d.plan(SCRIPT, title="The Letter")
    out = d.shoot(prod.id, shot_ids=["2.1"], qa=False)
    assert out.shot("2.1").status == "generated"


# ---------------------------------------------------------------------------
# Surfaces: tools, chat handler, CLI
# ---------------------------------------------------------------------------

def test_tools_registered():
    from nomorals.tools.registry import ToolRegistry
    r = ToolRegistry()
    r.register_builtins()
    assert not any(f["module"] == "media_director"
                   for f in r._failed_modules)
    for name in ("script_coverage", "movie_plan", "movie_shoot",
                 "movie_regen", "movie_show"):
        assert name in r._tools, f"{name} not registered"


def test_chat_handler_flow(tmp_path):
    from nomorals.agents.partner.runtime_media import RuntimeMediaMixin
    script_file = tmp_path / "letter.txt"
    script_file.write_text(SCRIPT)
    mixin = RuntimeMediaMixin()
    mixin.context = SimpleNamespace(workspace=str(tmp_path))
    usage = mixin._control_director("")
    assert "cover" in usage and "shoot" in usage
    styles = mixin._control_director("styles")
    assert "noir" in styles
    planned = mixin._control_director(f"plan {script_file} --style noir")
    assert "production" in planned and "shots designed" in planned
    prod_id = planned.split("production ")[1].split(" —")[0].strip()
    shown = mixin._control_director(f"show {prod_id}")
    assert prod_id in shown
    missing = mixin._control_director("cover /no/such/script.txt")
    assert "not found" in missing


def test_cli_parser_and_dispatch():
    from nomorals.cmdline.dispatch import _COMMANDS
    from nomorals.cmdline.parser import _parser
    assert "director" in _COMMANDS
    p = _parser()
    args = p.parse_args(["director", "plan", "x.txt", "--style", "neon"])
    assert args.command == "director"
    assert args.director_action == "plan"
    assert args.style == "neon"
    args = p.parse_args(["director", "shoot", "prod-9", "--no-qa"])
    assert args.director_action == "shoot" and args.no_qa is True
