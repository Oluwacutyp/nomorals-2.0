"""Soundscape tool: ``soundscape`` — Devon's own MyNoise core.

She composes ambient sound herself: noise colours with measured
spectra (white flat, pink −3 dB/oct, brown −6 dB/oct, blue
+3 dB/oct, grey equal-loudness), binaural beats and drones, and
layered scenes (rain, forest, ocean, fire, cafe, night) — each on
ten sliders, with a hearing calibration, an animate drift, and named
mixes she remembers. Every stem is synthesised in code; no recording
is ever faked, and a render that cannot be built is refused in
words. The tool renders the WAV; the chat surface sends it.
"""

from __future__ import annotations

from typing import Any

from ..core.policy import Capability

__all__ = ["register"]


def _home_from_context(context: Any) -> str | None:
    settings = getattr(context, "settings", None) if context else None
    if settings is None:
        return None
    home = getattr(settings, "home_path", None) or getattr(
        settings, "home", None)
    return str(home) if home else None


def register(registry: Any) -> None:
    """Attach the ``soundscape`` tool to a registry."""
    context = registry.context

    @registry.register(
        "soundscape",
        description=(
            "Compose and render an ambient soundscape — noise colours "
            "(white, pink, brown, grey, blue), binaural beats, drones, "
            "and layered scenes (rain, forest, ocean, fire, cafe, "
            "night) — each on ten sliders, with a hearing calibration "
            "and named mixes. action=<render|list|mixes|save|tune|"
            "calibrate|tone>; render takes query=<name, preset or "
            "natural words like 'rain sounds, more thunder'>, "
            "duration_s=<seconds, max 300>, timer_min=<minutes>, "
            "animate=<true|false>; list returns the generators, "
            "presets and saved mixes; save takes name=<mix name> and "
            "generator=<id> with sliders=<label: dB, ...>; tune takes "
            "name=<mix or generator>, band=<slider label> and "
            "db=<change in dB>; calibrate takes band=<Hz> and "
            "db=<gain>; tone takes freq=<Hz> and renders a calibration "
            "tone. Returns ok, the render receipt (path, render_hash, "
            "peak, rms, duration_s, reply) or error. An unknown "
            "generator or layer is refused honestly, never invented."
        ),
        capability=Capability.MEDIA,
    )
    def soundscape_tool(
        query: str = "",
        action: str = "render",
        name: str = "",
        generator: str = "",
        duration_s: Any = None,
        timer_min: Any = None,
        animate: Any = None,
        sliders: Any = None,
        band: str = "",
        db: Any = None,
        freq: Any = None,
    ) -> dict[str, Any]:
        from ..audio.soundscapes.engine import BAND_FREQS, render_tone
        from ..audio.soundscapes.generators import (
            GENERATORS,
            PRESETS,
            parse_soundscape_query,
        )
        from ..audio.soundscapes.render import render_soundscape
        from ..audio.soundscapes.store import open_store

        home = _home_from_context(context)
        if home is None:
            return {"ok": False,
                    "error": "soundscape needs a home directory"}
        store = open_store(home)
        act = (action or "render").strip().lower()
        if act == "list":
            return {"ok": True,
                    "generators": [
                        {"id": spec.id, "name": spec.name,
                         "kind": spec.kind, "sliders": list(spec.sliders),
                         "description": spec.description}
                        for spec in GENERATORS.values()],
                    "presets": sorted(PRESETS),
                    "mixes": store.list_mixes(),
                    "reply": ("generators: "
                              + ", ".join(sorted(GENERATORS))
                              + ". presets: " + ", ".join(sorted(PRESETS))
                              + (". mixes: " + ", ".join(store.list_mixes())
                                 if store.list_mixes() else ""))}
        if act == "mixes":
            mixes = store.list_mixes()
            return {"ok": True, "mixes": mixes,
                    "reply": ("saved mixes: " + ", ".join(mixes)
                              if mixes else "no saved mixes yet")}
        if act == "save":
            mix_name = (name or "").strip()
            gen_id = (generator or query or "").strip()
            if not mix_name or not gen_id:
                raise ValueError(
                    "soundscape save needs name=<mix name> and "
                    "generator=<id>")
            from ..audio.soundscapes.generators import resolve_generator

            spec = resolve_generator(gen_id)
            given = dict(sliders or {})
            values = []
            for idx, label in enumerate(spec.sliders):
                raw = given.get(label)
                if raw is None:
                    try:
                        raw = given.get(float(label.split()[0]))
                    except (TypeError, ValueError):
                        raw = None
                if raw is None:
                    raw = spec.defaults[idx] if spec.defaults else 0.0
                values.append(float(raw))
            store.save_mix(mix_name, spec.id, tuple(values),
                           animate=bool(animate)
                           if animate is not None else False,
                           duration_s=float(duration_s)
                           if duration_s is not None else 60.0)
            return {"ok": True, "name": mix_name, "generator": spec.id,
                    "reply": f"saved mix “{mix_name}” — {spec.name}"}
        if act == "tune":
            target = (name or query or "").strip()
            if not target:
                raise ValueError("soundscape tune needs a mix or generator")
            if db is None:
                raise ValueError("soundscape tune needs db=<change in dB>")
            mix = store.load_mix(target)
            if mix is not None:
                from ..audio.soundscapes.generators import resolve_generator

                spec = resolve_generator(str(mix["generator"]))
                values = list(mix["sliders"])
                label = (band or "").strip()
                if label in spec.sliders:
                    idx = spec.sliders.index(label)
                    values[idx] = max(-30.0, min(12.0, values[idx]
                                                 + float(db)))
                    store.save_mix(mix["name"], spec.id, tuple(values),
                                   animate=bool(mix.get("animate", False)),
                                   duration_s=float(
                                       mix.get("duration_s") or 60.0))
                    return {"ok": True, "name": mix["name"],
                            "reply": (f"“{mix['name']}”: {label} "
                                      f"{values[idx]:+.1f} dB")}
                return {"ok": False,
                        "error": f"“{label}” is not a slider of "
                                 f"“{mix['name']}” — try: "
                                 + ", ".join(spec.sliders)}
            from ..audio.soundscapes.generators import resolve_generator

            spec = resolve_generator(target)
            label = (band or "").strip()
            if label not in spec.sliders and label not in (
                    f"{f:g} Hz" for f in BAND_FREQS):
                return {"ok": False,
                        "error": f"“{label}” is not a slider of "
                                 f"{spec.name} — try: "
                                 + ", ".join(spec.sliders)}
            ans = render_soundscape(
                home, spec.id,
                duration_s=float(duration_s) if duration_s is not None
                else 60.0,
                animate=bool(animate) if animate is not None else False,
                adjust={label: float(db)})
            return ans
        if act == "calibrate":
            if not band or db is None:
                raise ValueError(
                    "soundscape calibrate needs band=<Hz> and db=<gain>")
            try:
                band_hz = float(band)
                gain = float(db)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "soundscape calibrate needs numeric band and db"
                    ) from exc
            if band_hz not in BAND_FREQS:
                return {"ok": False,
                        "error": f"{band_hz:g} Hz is not a calibration band "
                                 f"— try: "
                                 + ", ".join(f"{f:g}" for f in BAND_FREQS)}
            store.set_hearing_band(band_hz, gain)
            return {"ok": True, "band": band_hz, "db": gain,
                    "reply": (f"hearing: {band_hz:g} Hz "
                              f"{gain:+.1f} dB — applied to every render "
                              f"from now on")}
        if act == "tone":
            if freq is None:
                raise ValueError("soundscape tone needs freq=<Hz>")
            try:
                freq_hz = float(freq)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "soundscape tone needs a numeric freq") from exc
            path = render_tone(freq_hz, duration_s=2.0)
            return {"ok": True, "path": path, "freq": freq_hz,
                    "reply": f"calibration tone — {freq_hz:g} Hz"}
        text = (query or name or "").strip()
        if not text:
            raise ValueError(
                "soundscape needs a name — 'rain', 'forest', 'cafe', "
                "'brown noise' or a preset like 'sleep'")
        if store.load_mix(text) is not None:
            # The owner's own mix name wins over any preset word
            # inside it ("deep focus" is his mix, not the "focus"
            # preset) — render_soundscape loads the mix itself.
            return render_soundscape(home, text)
        parsed = parse_soundscape_query(text)
        target = parsed.get("generator") or text
        duration = None
        if duration_s is not None:
            duration = float(duration_s)
        if timer_min is not None:
            timer = float(timer_min)
        elif parsed.get("timer_min") is not None:
            timer = float(parsed["timer_min"])
        else:
            timer = None
        animate_flag = animate
        if animate_flag is None:
            animate_flag = parsed.get("animate")
        ans = render_soundscape(
            home, target,
            duration_s=duration,
            timer_min=timer,
            animate=(bool(animate_flag)
                     if animate_flag is not None else None),
            adjust=dict(parsed.get("adjust") or {}))
        return ans
