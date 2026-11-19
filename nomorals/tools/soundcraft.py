"""SoundCraft: open, read, edit, mix and bounce real .scraft audio
sessions — plus the realtime engine and quarantined plugin hosts
(SoundCraft port of storytold/soundcraft — an open-source digital
audio workstation).

A thin surface over :mod:`nomorals.media.sound` — the organ owns
the session model, the mixer, the effects chain, the realtime
callback engine, the plugin hosts and the offline render; the tool
only routes arguments and shapes the reply. Pure local work, stdlib
+ numpy only, no upload, no retention. Writes output files, so
FS_WRITE.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..core.policy import Capability
from ..media.sound import (
    HostKind,
    MixEngine,
    PluginHost,
    RealtimeEngine,
    SessionDocument,
    SessionEditor,
    render_session,
    run_parity,
    scan_plugins,
    verify_attribution,
)

__all__ = ["register"]

_ACTIONS = ("inspect", "create", "add", "set", "mix", "automate",
            "plugin", "render", "realtime", "parity", "attribution")


def register(registry: Any) -> None:
    """Attach the soundcraft family to a registry."""
    @registry.register(
        "soundcraft",
        description=(
            "Open, read, edit, mix and bounce real .scraft audio "
            "sessions, local and uncapped — inspect tracks, clips, "
            "buses and sources; add tracks, clips, notes and buses; "
            "set gains, pan, mute and solo; mix down with honest "
            "clip reporting; automate parameters sample-accurately; "
            "run the realtime callback engine; scan and load CLAP / "
            "VST3 / AU plugins in a quarantined subprocess; bounce a "
            "session to WAV. Edits preserve the untouched clips "
            "byte-for-byte. SoundCraft actions (session model + "
            "mixer + effects + realtime engine + plugin hosts): "
            "inspect=<read the session inventory, path=<.scraft>>; "
            "create=<blank session, out=<path>, title=<t>>; add=<add "
            "a track, clip, note or bus, kind=<track|clip|note|bus>, "
            "name=<n>, track=<id>, source=<wav path>, start=<samples>, "
            "duration=<samples>, pitch=<midi>, velocity=<0-1>>; "
            "set=<set a gain, pan, mute, solo or clip position, "
            "kind=<track_gain|track_pan|clip_gain|move|trim|mute|solo>, "
            "node=<id>, value=<number>>; mix=<mix down in memory, "
            "path=<.scraft>, peak and clip report>; automate=<add a "
            "parameter lane, track=<id>, param=<gain_db>, "
            "points=<\"sample:value,...\">>; plugin=<scan or load a "
            "plugin, kind=<scan|load>, host=<clap|vst3|au>, "
            "target=<dotted.path:Class>>; render=<bounce to WAV, "
            "path=<.scraft>, out=<wav path>>; realtime=<run the "
            "realtime callback engine over the session, 20 blocks, "
            "beats/underruns/faults>; parity=<run the "
            "machine-checked parity TOML>; attribution=<verify the "
            "SHA-256 attribution DB>. action=<inspect|create|add|set|"
            "mix|automate|plugin|render|realtime|parity|attribution>; "
            "path=<.scraft path>; out=<path to write>."
        ),
        capability=Capability.FS_WRITE,
    )
    def soundcraft_tool(action: str = "inspect",
                        path: str = "", out: str = "",
                        sources: list[str] | None = None,
                        output: str = "", title: str = "",
                        kind: str = "", name: str = "",
                        track: int = 0, node: int = 0,
                        source: str = "", start: int = 0,
                        duration: int = 0, pitch: int = 69,
                        velocity: float = 0.8,
                        value: float = 0.0, param: str = "gain_db",
                        points: str = "", host: str = "clap",
                        target: str = "", bus: int = 0,
                        ) -> dict[str, Any]:
        action = (action or "inspect").lower()
        if action not in _ACTIONS:
            raise ValueError(f"unknown soundcraft action {action!r}; use "
                             f"one of {', '.join(_ACTIONS)}")
        sources = list(sources or [])
        if path and not sources:
            sources = [path]
        if out and not output:
            output = out
        if action == "parity":
            result = run_parity()
            failed_ids = list(result["failed"])
            return {"ok": True, "passed": result["passed"],
                    "failed": len(failed_ids),
                    "failed_ids": failed_ids,
                    "reply": f"soundcraft parity: {result['passed']} "
                             f"passed, {len(failed_ids)} failed"
                             + (f" ({', '.join(failed_ids)})" if failed_ids else "")
                             + "."}
        if action == "attribution":
            result = verify_attribution()
            return {"ok": True, "verified": result["verified"],
                    "failed": len(result["failed"]),
                    "skipped": len(result["skipped"]),
                    "failed_items": result["failed"],
                    "skipped_items": result["skipped"],
                    "reply": f"soundcraft attribution: {result['verified']} "
                             f"verified, {len(result['failed'])} failed, "
                             f"{len(result['skipped'])} skipped."}
        if action == "create":
            if not output:
                raise ValueError("soundcraft create needs out=<path>")
            doc = SessionDocument.create(title=title or "Untitled")
            out_path = Path(output).expanduser()
            out_path.write_bytes(doc.to_bytes())
            return {"ok": True, "output": str(out_path),
                    "tracks": 0, "size_after": len(doc.to_bytes()),
                    "reply": f"created blank .scraft at {out_path}."}
        if action == "plugin":
            host_kind = HostKind(host.lower()) if host else HostKind.CLAP
            plugin_host = PluginHost(host_kind)
            if (kind or "scan").lower() == "scan":
                found = scan_plugins(host_kind, [str(Path.home())])
                return {"ok": True, "host": host_kind.value,
                        "plugins": [f.to_dict() for f in found],
                        "reply": f"{host_kind.value} scan: "
                                 f"{len(found)} plugin(s) found."}
            if not target:
                raise ValueError("soundcraft plugin load needs target=<dotted.path:Class>")
            inst = plugin_host.load(target)
            try:
                return {"ok": True, "host": host_kind.value,
                        "plugin": inst.info(), "state": inst.state,
                        "reply": f"loaded {inst.name} ({host_kind.value}) "
                                 f"in quarantine: {inst.state}."}
            finally:
                inst.close()
        if not sources:
            raise ValueError(f"soundcraft {action} needs a source .scraft")
        if action == "inspect":
            doc = SessionDocument.from_bytes(Path(sources[0]).read_bytes())
            inv = doc.inventory()
            inv["reply"] = (
                f"{sources[0]}: .scraft, {inv['tracks']} track(s), "
                f"{inv['clips']} clip(s), buses "
                f"{', '.join(inv['buses'])}, "
                f"{inv['length_samples']} samples.")
            return inv
        if action == "mix":
            doc = SessionDocument.from_bytes(Path(sources[0]).read_bytes())
            _, _, report = MixEngine(doc).render(0, doc.length_samples())
            return {"ok": True, "master_peak": report["master_peak"],
                    "clipped": report["clipped"],
                    "clip_first_sample": report["clip_first_sample"],
                    "strip_peaks": report["strip_peaks"],
                    "bus_peaks": report["bus_peaks"],
                    "reply": f"mix: master peak {report['master_peak']:.3f}, "
                             + ("CLIPPED at sample "
                                f"{report['clip_first_sample']}."
                                if report["clipped"] else "no clipping.")}
        if action == "render":
            if not output:
                raise ValueError("soundcraft render needs out=<wav path>")
            doc = SessionDocument.from_bytes(Path(sources[0]).read_bytes())
            result = render_session(doc, Path(output).expanduser())
            return {"ok": True, "output": str(output),
                    "peak": result["peak"], "rms": result["rms"],
                    "clipped": result["clipped"],
                    "frames": result["frames"],
                    "reply": f"bounced {result['frames']} frames to "
                             f"{output} (peak {result['peak']:.3f})."}
        if action == "realtime":
            doc = SessionDocument.from_bytes(Path(sources[0]).read_bytes())
            engine = RealtimeEngine(doc, block_size=512)
            engine.play()
            engine.drive(20)
            engine.stop()
            return {"ok": True, "beats": engine.beats,
                    "underruns": engine.underruns,
                    "faults": engine.faults,
                    "reply": f"realtime: {engine.beats} blocks, "
                             f"{engine.underruns} underruns, "
                             f"{len(engine.faults)} faults."}
        if not output:
            raise ValueError(f"soundcraft {action} needs out=<path>")
        original = Path(sources[0]).read_bytes()
        doc = SessionDocument.from_bytes(original)
        editor = SessionEditor(doc)
        if action == "add":
            add_kind = (kind or "track").lower()
            if add_kind == "track":
                nid = editor.add_track(name or "Track")
                detail = f"added track {nid}"
            elif add_kind == "clip":
                if not track or not source:
                    raise ValueError("soundcraft add clip needs track=<id> and source=<wav>")
                nid = editor.add_clip(track, name or "Clip", source, start)
                detail = f"added clip {nid} to track {track}"
            elif add_kind == "note":
                if not track:
                    raise ValueError("soundcraft add note needs track=<id>")
                nid = editor.add_note(track, start, duration or 24000,
                                      pitch, velocity)
                detail = f"added note {nid} (pitch {pitch}) to track {track}"
            elif add_kind == "bus":
                nid = editor.add_bus(name or "Aux")
                detail = f"added bus {nid}"
            else:
                raise ValueError(f"soundcraft add kind {add_kind!r}; use track, clip, note or bus")
        elif action == "set":
            set_kind = (kind or "track_gain").lower()
            if set_kind == "track_gain":
                editor.set_track_gain(node, value)
            elif set_kind == "track_pan":
                editor.set_track_pan(node, value)
            elif set_kind == "clip_gain":
                editor.set_clip_gain(node, value)
            elif set_kind == "move":
                editor.move_clip(node, int(value))
            elif set_kind == "trim":
                editor.trim_clip(node, end=int(value))
            elif set_kind == "mute":
                editor.mute_track(node, bool(value))
            elif set_kind == "solo":
                editor.solo_track(node, bool(value))
            else:
                raise ValueError(f"soundcraft set kind {set_kind!r}")
            detail = f"set {set_kind} on {node} to {value}"
        elif action == "automate":
            if not track or not points:
                raise ValueError("soundcraft automate needs track=<id> and points=<\"sample:value,...\">")
            parsed = []
            for part in points.split(","):
                sample, _, val = part.partition(":")
                parsed.append((int(sample), float(val)))
            editor.add_automation(track, param, parsed)
            detail = f"added {param} automation to track {track}"
        else:  # pragma: no cover — guarded by the _ACTIONS check above
            raise ValueError(f"unknown soundcraft action {action!r}")
        out_path = Path(output).expanduser()
        out_path.write_bytes(doc.to_bytes())
        fidelity = doc.fidelity_report()
        return {"ok": True, "output": str(out_path), "detail": detail,
                "changed": fidelity["changed"],
                "untouched_changed": fidelity["untouched_changed"],
                "reply": f"{detail}; saved to {out_path}."}
