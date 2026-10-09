"""Tests for nomorals.media.rhythm — the generative rhythm engine.

No pattern tables: Euclidean distributions generate the hits, groove
keywords steer the feel, energy modulates density.  Same spec + same
seed → same drums (deterministic).
"""

import random

from nomorals.media.rhythm import (
    euclidean,
    generate_bass,
    generate_drums,
    humanize,
    parse_groove_keywords,
)
from nomorals.media.songspec import GrooveSpec, SectionSpec, SongSpec


def _spec(**kw):
    groove = GrooveSpec(
        feel=kw.pop("feel", "driving four-on-the-floor"),
        swing=kw.pop("swing", 0.0),
        kick_style=kw.pop("kick_style", ""),
        snare_style=kw.pop("snare_style", ""),
        hat_style=kw.pop("hat_style", ""),
        percussion=kw.pop("percussion", ""),
    )
    sec = SectionSpec(
        name=kw.pop("section", "chorus"),
        bars=kw.pop("bars", 4),
        chords=kw.pop("chords", ["i", "VI", "III", "VII"]),
        energy=kw.pop("energy", 0.7),
    )
    return SongSpec(title="t", key="A", mode="minor", tempo=120,
                    groove=groove, sections=[sec])


# ── Euclidean core ────────────────────────────────────────────────────────

def test_euclidean_tresillo():
    # Tresillo = E(3,8) — the fundamental Afro-Latin rhythm
    assert euclidean(3, 8) == [True, False, False, True, False, False,
                               True, False]


def test_euclidean_cinquillo():
    assert sum(euclidean(5, 8)) == 5
    assert len(euclidean(5, 8)) == 8


def test_euclidean_four_floor():
    assert euclidean(4, 4) == [True, True, True, True]


def test_euclidean_empty_and_full():
    assert euclidean(0, 8) == [False] * 8
    assert euclidean(8, 8) == [True] * 8


def test_euclidean_rotation():
    # rotation spins the pattern — tresillo vs son-clave
    a = euclidean(3, 8, 0)
    b = euclidean(3, 8, 2)
    assert sum(a) == sum(b) == 3
    assert a != b


def test_euclidean_even_distribution():
    # hits should be spread, not clumped: max gap <= min gap + 1
    for p, s in [(3, 8), (5, 8), (5, 16), (7, 16)]:
        pat = euclidean(p, s)
        idx = [i for i, h in enumerate(pat) if h]
        gaps = [(idx[(i + 1) % len(idx)] - idx[i]) % s for i in range(len(idx))]
        assert max(gaps) - min(gaps) <= 1, f"E({p},{s}) not even: {gaps}"


# ── groove keywords ───────────────────────────────────────────────────────

def test_parse_groove_keywords():
    g = GrooveSpec(feel="laid-back afrobeats bounce",
                   kick_style="syncopated, sparse",
                   percussion="shaker and congas")
    kw = parse_groove_keywords(g)
    assert kw["laid_back"]
    assert kw["syncopated"]
    assert kw["sparse"]
    assert kw["shaker"]
    assert kw["congas"]
    assert not kw["four_floor"]


def test_parse_four_floor():
    g = GrooveSpec(kick_style="four-on-floor driving")
    assert parse_groove_keywords(g)["four_floor"]


# ── drum generation ───────────────────────────────────────────────────────

def _drums(**kw):
    spec = _spec(**kw)
    rng = random.Random(kw.get("seed", 7))
    return generate_drums(spec, spec.sections[0], 0, rng,
                          next_energy=kw.get("next_energy"))


def test_drums_deterministic():
    d1 = _drums(seed=7)
    d2 = _drums(seed=7)
    # same seed → same event multiset (humanize jitter is seeded too)
    k1 = sorted((e.note, round(e.start, 4), e.velocity) for e in d1)
    k2 = sorted((e.note, round(e.start, 4), e.velocity) for e in d2)
    assert k1 == k2


def test_drums_different_seeds_differ():
    d1 = _drums(seed=7)
    d2 = _drums(seed=99)
    k1 = sorted((e.note, round(e.start, 4)) for e in d1)
    k2 = sorted((e.note, round(e.start, 4)) for e in d2)
    assert k1 != k2


def test_drums_have_kick_snare_hats():
    evs = _drums()
    notes = {e.note for e in evs}
    assert 36 in notes, "kick missing"
    assert 38 in notes, "snare missing"
    assert 42 in notes or 46 in notes, "hats missing"


def test_high_energy_denser_than_low():
    hi = _drums(energy=0.9, seed=7)
    lo = _drums(energy=0.2, seed=7)
    assert len(hi) > len(lo), \
        f"high energy should be denser: {len(hi)} vs {len(lo)}"


def test_four_floor_kick_on_beats():
    evs = _drums(feel="driving", kick_style="four-on-floor", seed=7)
    kicks = sorted(e.start % 4.0 for e in evs if e.note == 36)
    # four-on-the-floor: kick on every quarter note
    for beat in (0.0, 1.0, 2.0, 3.0):
        assert any(abs(k - beat) < 0.05 for k in kicks), \
            f"four-floor kick missing on beat {beat}"


def test_half_time_snare_on_three():
    evs = _drums(snare_style="half-time on 3", kick_style="half-time",
                 seed=7)
    snares = [e.start % 4.0 for e in evs
              if e.note in (38, 37)]  # snare or rim
    # half-time: snares cluster on beat 3 (2.0), not on 2 and 4
    on_three = sum(1 for s in snares if abs(s - 2.0) < 0.3)
    assert on_three >= len(snares) * 0.5, \
        f"half-time snare should favor beat 3: {snares}"


def test_fill_at_section_boundary():
    # next_energy jump → fill in the last bar
    evs = _drums(bars=4, energy=0.4, next_energy=0.9, seed=7)
    last_bar = [e for e in evs if 12.0 <= e.start < 16.0 and e.note == 38]
    assert len(last_bar) >= 4, \
        f"expected snare build into the drop, got {len(last_bar)} hits"


def test_no_fill_without_energy_change():
    evs = _drums(bars=4, energy=0.5, next_energy=0.52, seed=7)
    last_bar_snare = [e for e in evs
                      if 14.0 <= e.start < 16.0 and e.note == 38]
    # no big build when energy is flat
    assert len(last_bar_snare) < 8


def test_crash_on_section_start():
    evs = _drums(section="chorus", seed=7)
    crashes = [e for e in evs if e.note == 49 and e.start < 1.0]
    assert crashes, "crash expected on chorus start"


def test_no_crash_on_intro():
    evs = _drums(section="intro", seed=7)
    crashes = [e for e in evs if e.note == 49 and e.start < 1.0]
    assert not crashes, "no crash on intro"


def test_swing_shifts_offbeats():
    # use a syncopated groove that guarantees offbeat hat placements
    straight = _drums(swing=0.0, seed=7, feel="syncopated bounce",
                      hat_style="16ths with swing")
    swung = _drums(swing=0.5, seed=7, feel="syncopated bounce",
                   hat_style="16ths with swing")
    # swung offbeats should have different timing than straight
    sh = sorted(round(e.start, 4) for e in straight if e.note == 42)
    wh = sorted(round(e.start, 4) for e in swung if e.note == 42)
    # at least some hats should differ (the offbeats that swing shifts)
    diff = sum(1 for a, b in zip(sh, wh) if abs(a - b) > 0.01)
    assert diff > 0, f"swing should shift some hat timing: {sh[:5]} vs {wh[:5]}"


def test_never_raises_on_garbage():
    # garbage section → empty list, not an exception
    spec = SongSpec(title="x")
    evs = generate_drums(spec, None, 0, random.Random(1))
    assert evs == []


# ── bass ──────────────────────────────────────────────────────────────────

def test_bass_locks_to_kick():
    spec = _spec()
    sec = spec.sections[0]
    rng = random.Random(7)
    drums = generate_drums(spec, sec, 0, rng)
    # extract kick attacks
    kicks: list[list[float]] = [[] for _ in range(sec.bars)]
    for e in drums:
        if e.note == 36:
            b = int(e.start // 4.0)
            if 0 <= b < sec.bars:
                kicks[b].append(round(e.start % 4.0, 2))
    from nomorals.core.midi import chord_progression
    from nomorals.media.songspec import normalize_roman
    prog = [normalize_roman(c, "minor") for c in ["i", "VI", "III", "VII"]]
    roots = [min(c) - 12 for c in chord_progression("A", "minor", prog)]
    bass_evs, _ = generate_bass(spec, sec, 0, roots * 2, kicks,
                                random.Random(7))
    assert bass_evs, "bass should produce events"
    # bass should hit on at least some kick positions (the lock)
    bass_starts = {round(e.start % 4.0, 1) for e in bass_evs}
    kick_starts = {round(k, 1) for bar in kicks for k in bar}
    overlap = bass_starts & kick_starts
    assert overlap, \
        f"bass should lock with kick: bass={bass_starts}, kick={kick_starts}"


def test_bass_never_raises():
    evs, attacks = generate_bass(None, None, 0, [], [], random.Random(1))
    assert evs == [] and attacks == []


# ── humanize ──────────────────────────────────────────────────────────────

def test_humanize_preserves_downbeats():
    from nomorals.core.midi import NoteEvent
    evs = [NoteEvent(note=36, start=0.0, duration=0.25, velocity=110),
           NoteEvent(note=36, start=1.5, duration=0.25, velocity=100)]
    humanize(evs, random.Random(7))
    assert evs[0].start == 0.0, "downbeat must not move"
    assert evs[1].start != 1.5, "offbeat should get micro-shift"
    assert 1 <= evs[0].velocity <= 127
