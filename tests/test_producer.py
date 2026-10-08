"""Tests for nomorals.media.producer — the composition engine."""

import random

from nomorals.media.producer import (
    MotifNote,
    analyze_reference,
    describe_to_profile,
    develop_motif,
    energy_score,
    plan_arrangement,
    write_motif,
)


def _rng(seed=7):
    return random.Random(seed)


# ── motif writing ─────────────────────────────────────────────────────────

def test_motif_resolves_to_tonic():
    m = write_motif("F#", "minor", bars=2, rng=_rng())
    assert m, "motif must not be empty"
    assert m[-1].degree == 0, "motif must resolve to tonic"


def test_motif_stepwise_motion_dominates():
    m = write_motif("A", "minor", bars=2, rng=_rng())
    steps = [abs(m[i].degree - m[i - 1].degree)
             for i in range(1, len(m))]
    stepwise = sum(1 for s in steps if s <= 2)
    assert stepwise / max(1, len(steps)) >= 0.5, \
        f"expected mostly stepwise motion, got {steps}"


def test_motif_fills_bars():
    m = write_motif("C", "major", bars=2, rng=_rng())
    total = sum(n.dur for n in m)
    assert abs(total - 8.0) < 0.01, f"2-bar motif must be 8 beats, got {total}"


# ── developmental techniques ──────────────────────────────────────────────

def _sample_motif():
    return [MotifNote(0, 1.0), MotifNote(2, 0.5), MotifNote(1, 0.5),
            MotifNote(4, 2.0)]


def test_sequence_transposes():
    out = develop_motif(_sample_motif(), "sequence", steps=2)
    assert [n.degree for n in out] == [2, 4, 3, 6]


def test_inversion_flips_intervals():
    out = develop_motif(_sample_motif(), "inversion")
    # intervals +2,-1,+3 become -2,+1,-3 → degrees 0,-2,-1,-4
    assert [n.degree for n in out] == [0, -2, -1, -4]


def test_retrograde_reverses():
    out = develop_motif(_sample_motif(), "retrograde")
    assert [n.degree for n in out] == [4, 1, 2, 0]
    assert abs(sum(n.dur for n in out) - 4.0) < 0.01


def test_diminution_halves():
    out = develop_motif(_sample_motif(), "diminution")
    assert [n.dur for n in out] == [0.5, 0.25, 0.25, 1.0]


def test_augmentation_doubles():
    out = develop_motif(_sample_motif(), "augmentation")
    assert [n.dur for n in out] == [2.0, 1.0, 1.0, 4.0]


def test_octave_displacement():
    out = develop_motif(_sample_motif(), "octave", up=True)
    assert [n.degree for n in out] == [7, 9, 8, 11]


def test_develop_does_not_mutate_input():
    src = _sample_motif()
    before = [(n.degree, n.dur) for n in src]
    develop_motif(src, "inversion")
    develop_motif(src, "sequence", steps=3)
    assert [(n.degree, n.dur) for n in src] == before


# ── reference analysis ────────────────────────────────────────────────────

def test_analyze_reference_rejects_non_url():
    p = analyze_reference("just some words")
    assert not p.ok
    assert "spotify" in p.reason.lower()


def test_analyze_reference_never_raises_on_garbage():
    for bad in ("", None, "http://", "spotify.com/track/"):
        try:
            p = analyze_reference(bad)  # type: ignore[arg-type]
        except Exception as exc:  # noqa: BLE001
            raise AssertionError(f"analyze_reference raised on {bad!r}: {exc}")
        assert not p.ok


def test_describe_to_profile_dark_edm():
    p = describe_to_profile("dark driving edm like black out days")
    assert 145 <= p.bpm <= 160
    assert p.mode == "minor"
    assert "dark" in p.energy_words


def test_describe_to_profile_uk_drill():
    p = describe_to_profile("uk drill banger")
    assert 138 <= p.bpm <= 144


# ── arrangement ───────────────────────────────────────────────────────────

def test_high_energy_gets_short_intro():
    p = describe_to_profile("relentless aggressive euphoric hardstyle")
    plan = plan_arrangement(p, _rng())
    intro = next(b for n, b in plan if n == "intro")
    assert intro <= 2, f"high-energy intro should be short, got {intro}"


def test_melancholic_gets_longer_build():
    p = describe_to_profile("melancholic dark dreamy ballad")
    plan = plan_arrangement(p, _rng())
    assert any(n == "break" and b >= 4 for n, b in plan), \
        "melancholic plan should breathe"


def test_arrangement_varies_with_energy():
    hot = plan_arrangement(describe_to_profile("relentless aggressive"), _rng())
    sad = plan_arrangement(describe_to_profile("melancholic dreamy"), _rng())
    assert hot != sad, "arrangement must not be a fixed template"


def test_energy_score_ordering():
    hot = energy_score(describe_to_profile("relentless aggressive euphoric"))
    sad = energy_score(describe_to_profile("melancholic dreamy chill"))
    assert hot > sad


# ── end-to-end (offline) ──────────────────────────────────────────────────

def test_produce_from_description_never_raises(tmp_path):
    from unittest.mock import MagicMock
    from nomorals.media.producer import Producer

    ctx = MagicMock()
    # safe_path needs a real-ish context; give it a workspace root
    ctx.workspace = str(tmp_path)
    p = Producer(ctx)
    res = p.produce("dark driving edm test", seed=42, workdir="produced")
    # may fail on render deps, but must never raise and must be honest
    assert isinstance(res, dict) and "ok" in res


def test_motifs_differ_between_seeds():
    a = write_motif("F#", "minor", rng=random.Random(1))
    b = write_motif("F#", "minor", rng=random.Random(2))
    da = [(n.degree, n.dur) for n in a]
    db = [(n.degree, n.dur) for n in b]
    assert da != db, "motifs must not be a deterministic template"
