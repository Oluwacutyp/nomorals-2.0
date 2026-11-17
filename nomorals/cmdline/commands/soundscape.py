"""``nm soundscape`` — compose, tune and play ambient soundscapes.

    nm soundscape list
    nm soundscape rain --duration 60
    nm soundscape "rain sounds, more thunder" --timer 40
    nm soundscape sleep [--json]
    nm soundscape mixes
    nm soundscape save "deep focus" brown [--duration 120] [--animate]
    nm soundscape tune "deep focus" thunder 3
    nm soundscape tone 1000
    nm soundscape calibrate 1000 6

The same organ chat /soundscape uses: every stem synthesised in
code, nothing faked. An unknown generator prints the honest reason
and exits 1 — never an invented soundscape. A bare ``nm soundscape``
prints usage and exits 2.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

__all__ = ["_cmd_soundscape"]


def _home(context: Any) -> str | None:
    settings = getattr(context, "settings", None) if context else None
    if settings is None:
        return None
    home = getattr(settings, "home_path", None) or getattr(
        settings, "home", None)
    return str(home) if home else None


def _print(answer: dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(answer, indent=2, default=str))
    else:
        print(answer.get("reply") or answer.get("error") or "")


def _cmd_soundscape(args: argparse.Namespace, context: Any) -> int:
    from ...audio.soundscapes.engine import BAND_FREQS, render_tone
    from ...audio.soundscapes.generators import (
        GENERATORS,
        PRESETS,
        parse_soundscape_query,
        resolve_generator,
    )
    from ...audio.soundscapes.render import render_soundscape
    from ...audio.soundscapes.store import open_store

    home = _home(context) or "~/.nomorals"
    store = open_store(home)
    words = list(getattr(args, "soundtrack_query", None)
                 or getattr(args, "soundscape_query", None) or [])
    as_json = bool(getattr(args, "json", False))
    verb = (words[0].lower() if words else "")
    try:
        if not words:
            print("soundscape: give me a soundscape — "
                  "nm soundscape <rain|forest|cafe|brown noise|sleep|…> "
                  "| list | mixes | save <name> <generator> | "
                  "tune <mix> <stem|band> <dB> | tone <freq> | "
                  "calibrate <band Hz> <gain dB>", file=sys.stderr)
            return 2
        if verb == "list":
            lines = ["soundscapes — generators:"]
            for spec in GENERATORS.values():
                lines.append(f"• {spec.name} ({spec.id}) — {spec.description}")
            lines.append("presets: " + ", ".join(sorted(PRESETS)))
            mixes = store.list_mixes()
            if mixes:
                lines.append("your mixes: " + ", ".join(mixes))
            print("\n".join(lines))
            return 0
        if verb == "mixes":
            mixes = store.list_mixes()
            if not mixes:
                print("no saved mixes yet")
            else:
                for name in mixes:
                    mix = store.load_mix(name)
                    print(f"• {mix['name']} — {mix['generator']}"
                          if mix else f"• {name}")
            return 0
        if verb == "save":
            if len(words) < 3:
                print("soundscape: save needs a name and a generator — "
                      "nm soundscape save <name> <generator>",
                      file=sys.stderr)
                return 2
            spec = resolve_generator(words[2])
            store.save_mix(words[1], spec.id, tuple(spec.defaults),
                           animate=bool(getattr(args, "animate", False)),
                           duration_s=float(
                               getattr(args, "duration", None) or 60.0))
            print(f"saved mix “{words[1]}” — {spec.name}")
            return 0
        if verb == "tune":
            # dB is the last word; the label is the word(s) before it
            # ("1000 Hz" is two words); the target is the rest.
            if len(words) < 4:
                print("soundscape: tune needs a mix, a stem or band, and "
                      "a dB change — nm soundscape tune <mix> <stem> <dB>",
                      file=sys.stderr)
                return 2
            try:
                delta = float(words[-1])
            except (TypeError, ValueError):
                print("soundscape: the dB change must be a number",
                      file=sys.stderr)
                return 2
            if words[-2] == "Hz" and len(words) >= 5:
                label = words[-3] + " Hz"
                target = " ".join(words[1:-3])
            else:
                label = words[-2]
                target = " ".join(words[1:-2])
            mix = store.load_mix(target)
            if mix is not None:
                spec = resolve_generator(str(mix["generator"]))
                values = list(mix["sliders"])
                if label not in spec.sliders:
                    print(f"“{label}” is not a slider of "
                          f"“{mix['name']}” — try: "
                          + ", ".join(spec.sliders), file=sys.stderr)
                    return 1
                idx = spec.sliders.index(label)
                values[idx] = max(-30.0, min(12.0, values[idx] + delta))
                store.save_mix(mix["name"], spec.id, tuple(values),
                               animate=bool(mix.get("animate", False)),
                               duration_s=float(mix.get("duration_s") or 60.0))
                print(f"“{mix['name']}”: {label} {values[idx]:+.1f} dB")
                return 0
            spec = resolve_generator(target)
            if label not in spec.sliders:
                print(f"“{label}” is not a slider of {spec.name} — "
                      f"try: " + ", ".join(spec.sliders), file=sys.stderr)
                return 1
            ans = render_soundscape(
                home, spec.id,
                duration_s=float(getattr(args, "duration", None) or 60.0),
                adjust={label: delta})
            _print(ans, as_json)
            return 0 if ans.get("ok") else 1
        if verb == "tone":
            if len(words) != 2:
                print("soundscape: tone needs a frequency — "
                      "nm soundscape tone <freq Hz>", file=sys.stderr)
                return 2
            try:
                freq = float(words[1])
            except (TypeError, ValueError):
                print("soundscape: the frequency must be a number",
                      file=sys.stderr)
                return 2
            path = render_tone(freq, duration_s=2.0)
            print(f"calibration tone — {freq:g} Hz: {path}")
            return 0
        if verb == "calibrate":
            if len(words) != 3:
                bands = ", ".join(f"{f:g}" for f in BAND_FREQS)
                print("soundscape: calibrate needs a band and a gain — "
                      "nm soundscape calibrate <band Hz> <gain dB>\n"
                      f"bands: {bands}", file=sys.stderr)
                return 2
            try:
                band_hz = float(words[1])
                gain = float(words[2])
            except (TypeError, ValueError):
                print("soundscape: band and gain must be numbers",
                      file=sys.stderr)
                return 2
            if band_hz not in BAND_FREQS:
                bands = ", ".join(f"{f:g}" for f in BAND_FREQS)
                print(f"{band_hz:g} Hz is not a calibration band — "
                      f"try: {bands}", file=sys.stderr)
                return 1
            store.set_hearing_band(band_hz, gain)
            print(f"hearing: {band_hz:g} Hz {gain:+.1f} dB — applied to "
                  f"every render from now on")
            return 0
        text = " ".join(words).strip()
        if store.load_mix(text) is not None:
            # The owner's own mix name wins over any preset word
            # inside it ("deep focus" is his mix, not the "focus"
            # preset) — render_soundscape loads the mix itself.
            ans = render_soundscape(home, text)
            _print(ans, as_json)
            return 0 if ans.get("ok") else 1
        parsed = parse_soundscape_query(text)
        target = parsed.get("generator") or text
        duration = getattr(args, "duration", None)
        timer = getattr(args, "timer", None)
        if timer is None and parsed.get("timer_min") is not None:
            timer = parsed["timer_min"]
        animate = getattr(args, "animate", False)
        if getattr(args, "no_animate", False):
            animate = False
        elif parsed.get("animate") is not None:
            animate = parsed["animate"]
        ans = render_soundscape(
            home, target,
            duration_s=float(duration) if duration is not None else None,
            timer_min=float(timer) if timer is not None else None,
            animate=bool(animate),
            fade_s=float(getattr(args, "fade", None))
            if getattr(args, "fade", None) is not None else None,
            adjust=dict(parsed.get("adjust") or {}))
        _print(ans, as_json)
        return 0 if ans.get("ok") else 1
    except ValueError as exc:
        print(f"soundscape: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - CLI surfaces as text
        print(f"soundscape: {exc}", file=sys.stderr)
        return 1
