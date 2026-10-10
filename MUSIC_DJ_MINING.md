# MUSIC/DJ SYSTEM — Mining Report

## What exists (repo survey)

### Composition
- **`composer_llm.py`** (410 lines) — LLM-first SongSpec composition. Brain writes lyrics/structure/chords/groove as JSON. Falls back to algorithmic. Tension/release rules in prompt.
- **`songspec.py`** (293 lines) — SongSpec schema: sections, chords, groove, melody ideas.
- **`producer.py`** (940 lines) — Renders SongSpec to WAV. Motivic composition (write motif → develop via sequence/inversion/retrograde), dynamic arrangement, independent bass, counter-melodies.
- **`afrobeats.py`** (909 lines) — Afrobeats-specific generation.
- **`abc_melody.py`** (261 lines) — ABC notation melody generation for LLM prompt.
- **`freestyle.py`** (136 lines) — Freestyle rap generation.
- **`rhythm.py`** (645 lines) — Rhythm pattern engine.
- **`song_draft.py`** — Draft → perform pipeline.
- **`synth.py`/`synth_backend.py`** — Profile-aware synth selection.
- **`vocals.py`/`vocal_lite.py`** — Vocal synthesis.

### DJ / Mixing
- **`dj_engine.py`** (604 lines) — BPM detection, key detection, harmonic mixing (Camelot), beat-matched transitions, energy arc planning.
- **`dj_live.py`** (811 lines) — Live DJ sets.
- **`dj.py`** (635 lines) — DJ interface.
- **`dj_mixdrop.py`** — Mix drops.

### Playback / Sources
- **`playback.py`** (1817 lines) — Playback engine, queue management.
- **`sources.py`** (417 lines) — Source adapters: SoundCloud, YouTube, Audiomack, Boomplay, NetNaija, Spotify.
- **`resolver.py`** (664 lines) — Multi-source resolution with fallback chain.
- **`downloader.py`** (906 lines) — Download with proxy/browser fallbacks.
- **`music.py`** (1637 lines) — Music library/organization.

### Audio
- **`nomorals/audio/`** — DSP, editing, fingerprinting, pipeline, characters, audiobook.

## Known weaknesses

### 1. SoundCloud preview-only tracks (CONFIRMED BUG)
SoundCloud returns ~30s preview transcodings for restricted tracks. The resolver downloads these silently — user gets a 29s clip instead of the full song. Fix: detect preview URLs (`/preview/` in path) in the SoundCloud connector, raise `SoundCloudError`, and let the resolver fall through to YouTube/Audiomack/Boomplay for the full track.

### 2. The 4 composition flaws (user-confirmed)
- **Aimless melodies** — motifs don't develop with intent; they wander.
- **Repetitive arrangements** — sections feel copy-pasted.
- **Weak drums/bass groove** — drums are metronomic, bass is root-note plodding.
- **Everything sounds the same** — no stylistic range.

### 3. DJ transitions are mechanical
`dj_engine.py` does beat-matching and harmonic mixing, but transitions are algorithmic crossfades. No phrasing awareness (dropping on the 1), no energy-matched EQ sculpting, no creative transitions (echo out, filter sweep, loop roll).

### 4. Playback error handling
`playback.py` is 1817 lines — likely has silent failures. The metadata dict bug (fixed in tools/media.py) suggests similar issues elsewhere.

### 5. Source coverage gaps
NetNaija, Audiomack, Boomplay are wired but may be stale. Need to verify they still work.

## Best-in-class outside (what to steal)

### Composition
- **Suno/Udio** — LLM writes the full song concept first, then renders. We do this, but our render is MIDI-quality.
- **AIVA** — Motivic development with emotional arc. We have motifs but the arc is weak.
- **Key insight:** The LLM should write MELODY CONTOURS (not just "ideas") — actual pitch sequences the producer renders faithfully.

### DJ
- **rekordbox/Serato** — Phrase-aware mixing (16/32-bar phrases), key shift, beat jump.
- **Algoriddim djay** — Neural Mix (stem separation for live remixing).
- **Key insight:** Transitions should be PLANNED by the LLM (creative direction) and EXECUTED by the engine (precise beat-matching).

### Playback
- **Spotify** — Never silently fails; always tells you why.
- **Key insight:** Every playback failure should be honest and actionable.

## The plan

1. **Fix SoundCloud previews** — Detect and fall through. (Known issue, do first.)
2. **Composition upgrade** — LLM writes actual melody contours, not just ideas. Stronger motivic development. Bass/drum groove from the song's energy.
3. **DJ upgrade** — Phrase-aware transitions, LLM-planned creative mixes.
4. **Playback hardening** — Audit for silent failures, fix them.
5. **Source verification** — Test all sources, fix stale ones.

Even the "trash" builds have gold: the algorithmic fallback composer has solid music theory (circle of fifths, voice leading) that the LLM path should use as guardrails.
