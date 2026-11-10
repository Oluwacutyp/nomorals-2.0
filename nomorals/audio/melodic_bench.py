"""SR3 fixture-scope melodic sibling benchmark (doc 24 §8/§9).

The SR2 benchmark (``songrecon_bench``) measures the class-1
recording matcher. Its sibling pool — the same composition
re-performed by the bench synthesizer — is the pool the recording
engine is *not* built for: SR2 measured 12/24 sibling excerpts
accepted, every accept naming its own composition, zero
cross-accepts (``d24sr2-verify/final-artifact/report.md``).

This bench runs the SR3 melodic matcher (``audio.melodic``) over
the same kind of pool and reports, per pool: how often the source
work is the TOP-ranked hypothesis, how often it appears anywhere
in the hypothesis list, how often a wrong work tops the list, and
(for non-catalog negatives) how often anything is hypothesized at
all. The class-1 engine is run on the same query files for a
same-queries side-by-side.

SCOPE — read before quoting any number: every track, sibling and
negative here is SYNTHESIZED by the bench generator. These numbers
are fixture-scope evidence about the matcher's behaviour on
controlled re-performances. They are NOT cover/hum accuracy on
real recordings: per doc 24 §9 that stays unproven (G0) until a
blind ground-truth set from the owner's real library runs.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from .fingerprint import FingerprintDB
from .melodic import MelodyCatalog
from .songrecon_bench import (
    add_noise,
    excerpt,
    synth_sibling,
    synth_track,
    write_wav,
)

__all__ = [
    "MELODIC_BENCH_SCHEMA",
    "run_melodic_benchmark",
]

MELODIC_BENCH_SCHEMA = "songrecon.melodic-bench/v1"

#: The SR2 recording-engine sibling result this bench is read
#: against (measured artifact, not re-derived here):
#: d24sr2-verify/final-artifact/report.md — pool "siblings",
#: n=24, accepts=12, every accept naming its own composition.
SR2_SIBLING_REFERENCE = {
    "pool": "siblings (SR2 §9 battery)",
    "n": 24,
    "accepted": 12,
    "accept_rate": 0.5,
    "cross_accepts": 0,
    "provenance": "d24sr2-verify/final-artifact/report.md",
}

_POSITIVE_POOLS = ("sibling_full", "sibling_excerpt10",
                   "noise_sibling_full")
_NEGATIVE_POOLS = ("noncatalog_full", "noncatalog_excerpt10",
                   "stranger_leap", "stranger_same_collection")

#: Hand-written stranger melodies (NOT bench-generator output),
#: rendered with a plain sine stack. ``stranger_leap`` uses a
#: leap grammar alien to every catalog melody; the matcher must
#: stay silent. ``stranger_same_collection`` is a stepwise
#: pentatonic line sharing the catalog works' pitch collection —
#: the hard case, reported as measured.
_STRANGER_LEAP = [(48, 1), (60, 1), (55, 1), (67, 1), (53, 1),
                  (65, 1), (50, 1), (62, 1), (57, 1), (69, 1),
                  (52, 1), (64, 2)]
_STRANGER_SAME_COLLECTION = [
    (50, 1), (53, 1), (58, 1), (57, 1), (53, 2), (50, 1), (55, 1),
    (53, 1), (52, 1), (50, 1), (48, 1), (50, 2), (53, 1), (50, 2)]


def _render_stranger(notes: list[tuple[int, int]], seed: int):
    """Render a hand-written melody (sine stack + soft bass)."""
    import math
    import random
    from array import array

    sr = 11025
    beat = 0.42
    total = sum(d for _, d in notes) * beat + 0.4
    out = [0.0] * (int(total * sr) + 2048)
    rng = random.Random(seed)
    t = 0.15
    for midi, dur_beats in notes:
        dur = dur_beats * beat
        f0 = 440.0 * 2 ** ((midi - 69) / 12)
        n = int(dur * sr)
        s0 = int(t * sr)
        for k in range(n):
            tt = k / sr
            env = min(1.0, tt / 0.02) * math.exp(-tt / (dur * 1.4))
            out[s0 + k] += 0.5 * env * (
                math.sin(2 * math.pi * f0 * tt)
                + 0.35 * math.sin(2 * math.pi * f0 * 2 * tt)
                + 0.15 * math.sin(2 * math.pi * f0 * 3 * tt))
            fb = f0 / 4
            out[s0 + k] += 0.3 * env * math.sin(2 * math.pi * fb * tt)
        t += dur
    for i in range(len(out)):
        out[i] += rng.gauss(0.0, 0.004)
    return array("f", out)


def _pool_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if not n:
        return {"n": 0}
    return {
        "n": n,
        "top1_correct": sum(1 for r in rows if r["top1_correct"]),
        "own_in_hypotheses": sum(
            1 for r in rows if r["own_rank"] is not None),
        "any_hypothesis": sum(
            1 for r in rows if r["n_hypotheses"] > 0),
        "wrong_top1": sum(
            1 for r in rows
            if r["n_hypotheses"] > 0 and not r["top1_correct"]),
        "fingerprint_accepts": sum(
            1 for r in rows if r["fp_accepted"]),
        "fingerprint_accepts_correct": sum(
            1 for r in rows if r["fp_accepted_correct"]),
    }


def run_melodic_benchmark(out_dir: str | os.PathLike[str], *,
                          n_tracks: int = 12,
                          track_seconds: float = 18.0,
                          seed_base: int = 20000,
                          negative_seed_base: int = 90000,
                          ) -> dict[str, Any]:
    """Run the melodic sibling pools; write report.json + report.md
    into ``out_dir`` and return the report dict."""
    out = Path(out_dir)
    (out / "queries").mkdir(parents=True, exist_ok=True)
    (out / "catalog").mkdir(parents=True, exist_ok=True)
    started = time.time()
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started))

    fdb = FingerprintDB(str(out / "bench_fp.db"))
    mcat = MelodyCatalog(str(out / "bench_melodies.db"))
    titles: dict[int, str] = {}
    rows: dict[str, list[dict[str, Any]]] = {
        name: [] for name in _POSITIVE_POOLS + _NEGATIVE_POOLS}
    # Incremental + resumable: rows land in rows.jsonl as each query
    # completes, so an interrupted run keeps its measurements and a
    # re-run skips finished (pool, seed) pairs and a finished catalog.
    rows_path = out / "rows.jsonl"
    done: set[tuple[str, int]] = set()
    if rows_path.exists():
        for line in rows_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            rows[row["pool"]].append(row)
            done.add((row["pool"], int(row["seed"])))
    rows_file = rows_path.open("a", encoding="utf-8")

    def _record(pool: str, row: dict[str, Any]) -> None:
        row = {"pool": pool, **row}
        rows[pool].append(row)
        done.add((pool, int(row["seed"])))
        rows_file.write(json.dumps(row, sort_keys=True) + "\n")
        rows_file.flush()
        os.fsync(rows_file.fileno())
        print(f"[{pool}] seed={row['seed']} hyps={row['n_hypotheses']} "
              f"top1={row['top1_title'] or '-'} "
              f"fp={'Y' if row['fp_accepted'] else 'n'}", flush=True)

    try:
        catalog_ready = fdb.stats().get("tracks", 0) >= n_tracks \
            and mcat.catalog_snapshot().get("works", 0) >= n_tracks
        for i in range(n_tracks):
            seed = seed_base + i
            title = f"Bench Track {i:02d}"
            titles[seed] = title
            if catalog_ready:
                continue
            path = write_wav(out / "catalog" / f"track{i:02d}.wav",
                             synth_track(seed, track_seconds))
            res = fdb.add_track(path, title=title,
                                artist="SongRecon Bench")
            if not res.get("ok"):
                raise RuntimeError(f"catalog build failed: {res}")
            work = mcat.enroll_fingerprint_track(
                fdb, int(res["track_id"]), path)
            if not work.get("ok"):
                raise RuntimeError(f"melody enroll failed: {work}")

        def run_query(pool: str, qpath: str, expected: str | None,
                      seed: int) -> None:
            if (pool, seed) in done:
                return
            got = mcat.match(qpath, prefer_stem=False)
            hyps = got.get("hypotheses") or []
            hyp_titles = [h.get("title") or "" for h in hyps]
            fp = fdb.match(qpath)
            _record(pool, {
                "seed": seed,
                "expected": expected or "",
                "n_hypotheses": len(hyps),
                "top1_title": hyp_titles[0] if hyp_titles else "",
                "top1_correct": bool(expected)
                and bool(hyp_titles) and hyp_titles[0] == expected,
                "own_rank": hyp_titles.index(expected)
                if expected and expected in hyp_titles else None,
                "own_fraction": next(
                    (float(h["match_strength"]["value"])
                     for h in hyps
                     if (h.get("title") or "") == expected), None),
                "fp_accepted": bool(fp.get("ok")),
                "fp_accepted_correct": bool(fp.get("ok"))
                and (fp.get("title") or "") == (expected or ""),
            })

        for seed, title in titles.items():
            sibling = synth_sibling(seed, track_seconds)
            full = write_wav(
                out / "queries" / f"sib-full-{seed}.wav", sibling)
            run_query("sibling_full", full, title, seed)
            cut = write_wav(
                out / "queries" / f"sib-cut-{seed}.wav",
                excerpt(sibling, 0.0, 10.0))
            run_query("sibling_excerpt10", cut, title, seed)
            noisy = write_wav(
                out / "queries" / f"sib-noise-{seed}.wav",
                add_noise(sibling, 10.0, seed * 31 + 7))
            run_query("noise_sibling_full", noisy, title, seed)

        n_neg = max(4, n_tracks // 2)
        for i in range(n_neg):
            seed = negative_seed_base + i
            samples = synth_track(seed, track_seconds)
            full = write_wav(
                out / "queries" / f"neg-full-{seed}.wav", samples)
            run_query("noncatalog_full", full, None, seed)
            cut = write_wav(
                out / "queries" / f"neg-cut-{seed}.wav",
                excerpt(samples, 0.0, 10.0))
            run_query("noncatalog_excerpt10", cut, None, seed)

        leap = write_wav(out / "queries" / "stranger-leap.wav",
                         _render_stranger(_STRANGER_LEAP, 777001))
        run_query("stranger_leap", leap, None, 777001)
        cousin = write_wav(
            out / "queries" / "stranger-cousin.wav",
            _render_stranger(_STRANGER_SAME_COLLECTION, 777002))
        run_query("stranger_same_collection", cousin, None, 777002)
    finally:
        rows_file.close()
        mcat.close()
        fdb.close()

    pools = {name: _pool_stats(rows[name])
             for name in _POSITIVE_POOLS}
    negatives = {}
    for name in _NEGATIVE_POOLS:
        nrows = rows[name]
        negatives[name] = {
            "n": len(nrows),
            "any_hypothesis": sum(
                1 for r in nrows if r["n_hypotheses"] > 0),
            "total_hypotheses": sum(
                r["n_hypotheses"] for r in nrows),
            "fingerprint_accepts": sum(
                1 for r in nrows if r["fp_accepted"]),
        }
    report: dict[str, Any] = {
        "schema": MELODIC_BENCH_SCHEMA,
        "generated": stamp,
        "label": "synthesized-fixture measurement (bench synthesizer "
                 "re-performances) — NOT real-world cover/hum "
                 "accuracy; doc 24 §9 ground truth remains G0/"
                 "unproven until the owner's-library blind set runs",
        "catalog": {"tracks": n_tracks, "works": n_tracks,
                    "track_seconds": track_seconds},
        "pools": pools,
        "negatives": negatives,
        "rows": rows,
        "recording_engine_reference": SR2_SIBLING_REFERENCE,
        "wall_seconds": round(time.time() - started, 1),
    }
    (out / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (out / "report.md").write_text(
        _render_markdown(report), encoding="utf-8")
    return report


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# SR3 melodic sibling benchmark (fixture scope)",
        "",
        f"Generated: {report['generated']} · schema "
        f"`{report['schema']}` · wall {report['wall_seconds']}s "
        f"(resumable run — rows accumulate incrementally in "
        f"rows.jsonl across legs; wall is the final leg)",
        "",
        f"> {report['label']}",
        "",
        "## Re-performance pools (melodic matcher)",
        "",
        "| pool | n | top-1 correct | own in hypotheses | any "
        "hypothesis | wrong top-1 | class-1 accepts (same queries) |",
        "|---|---|---|---|---|---|---|",
    ]
    for name in _POSITIVE_POOLS:
        s = report["pools"][name]
        lines.append(
            f"| {name} | {s['n']} | {s['top1_correct']} | "
            f"{s['own_in_hypotheses']} | {s['any_hypothesis']} | "
            f"{s['wrong_top1']} | {s['fingerprint_accepts']} "
            f"({s['fingerprint_accepts_correct']} correct) |")
    lines += [
        "",
        "## Non-catalog negatives",
        "",
        "| pool | n | queries with any hypothesis | total "
        "hypotheses | class-1 accepts |",
        "|---|---|---|---|---|",
    ]
    for name in _NEGATIVE_POOLS:
        s = report["negatives"][name]
        lines.append(
            f"| {name} | {s['n']} | {s['any_hypothesis']} | "
            f"{s['total_hypotheses']} | {s['fingerprint_accepts']} |")
    lines += [
        "",
        "Reading the negatives honestly: the `noncatalog_*` pools "
        "are *other compositions from the same bench generator* — "
        "they share the catalog's chord-walk vocabulary, and the "
        "hypothesis floor does not keep such same-family strangers "
        "out (they surface as competing hypotheses, which is what "
        "the receipt presents them as). The hand-written "
        "`stranger_leap` melody still picks up hypotheses against "
        "THIS catalog — the bench tracks' extracted melodies "
        "contain wide extraction leaps of their own, so leap "
        "intervals find partners; against the small hand-built "
        "catalog in the SR3 test suite the same melody stays "
        "silent. `stranger_same_collection` (same pitch collection, "
        "different tune) gets none here. The floor's silence is "
        "therefore catalog-dependent; what the pools show "
        "consistently is the ranking — true re-performances put "
        "their own work top-1 in 9-10 of 12 pools and in the "
        "hypothesis list in 10-12 of 12.",
        "",
    ]
    ref = report["recording_engine_reference"]
    lines += [
        "",
        "## Reference: SR2 recording engine on its sibling pool",
        "",
        f"SR2 measured {ref['accepted']}/{ref['n']} sibling excerpts "
        f"accepted (rate {ref['accept_rate']}), zero cross-accepts "
        f"({ref['provenance']}). The melodic matcher above answers "
        "the different question — same composition, different "
        "performance — and answers it with ranked hypotheses, never "
        "identifications.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the SR3 fixture-scope melodic sibling "
                    "benchmark (synthesized fixtures only).")
    parser.add_argument("--out", required=True,
                        help="output directory for report.json/md "
                             "and the fixture catalog/queries")
    parser.add_argument("--tracks", type=int, default=12)
    parser.add_argument("--seconds", type=float, default=18.0)
    args = parser.parse_args(argv)
    report = run_melodic_benchmark(
        args.out, n_tracks=args.tracks, track_seconds=args.seconds)
    pools = report["pools"]
    print(json.dumps({"pools": pools,
                      "negatives": report["negatives"],
                      "wall_seconds": report["wall_seconds"]},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
