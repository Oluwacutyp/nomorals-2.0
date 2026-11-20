"""RuntimeMocapMixin: PartnerRuntime command group (mocap + rig)."""

from __future__ import annotations

import json
from typing import Any

from ...core.logging_setup import get_logger
from .failures import record_failure

_log = get_logger(__name__)


class RuntimeMocapMixin:
    """RuntimeMocapMixin for :class:`PartnerRuntime`."""

    # ── mocap + rig: /mocap ─────────────────────────────────────────────
    def _control_mocap(self, tail: str, chat_key: str,
                       message: Any = None) -> str:
        """Motion capture and the VTuber rig, live from any chat.

        /mocap status                             store channels + engines
        /mocap rig list                           loaded rig sessions
        /mocap rig load [name]                    load a rig (default rig)
        /mocap rig status [name]                  a session's rig + remap
        /mocap rig pose [name] [frame JSON]       drive one frame
        /mocap rig render [name]                  render the pose (PNG)
        /mocap rig drive <bvh> [text]             drive a take, render it
        /mocap rig export <bvh> [text]            drive a take, BVH back out
        /mocap ingest start [live|fixture] [name] start polling a source
        /mocap ingest stop [name]                 stop + drop a session
        /mocap ingest status [name]               frames/points/drops
        /mocap ingest tick [name] [n]             poll n frames by hand
        /mocap ingest publish [live|fixture]      one frame, no session
        """
        from ...media.mocap.bvh import SkeletonFrame
        from ...media.mocap.errors import MocapError
        from ...media.rig import (
            FaceInput,
            get_session,
            list_sessions,
            load_session,
            load_take_bvh,
        )
        from ...media.rig.errors import RigError
        from ...tools.mocap import _STORE

        parts = (tail or "").split()
        if not parts:
            return self._control_mocap_help()
        if parts[0] == "status":
            return ("mocap channels: "
                    + ", ".join(_STORE.channels())
                    + " | engines: "
                    + ", ".join(e["name"] for e in _STORE.engines()))
        if parts[0] == "ingest":
            return self._control_mocap_ingest(parts[1:])
        if parts[0] != "rig":
            return self._control_mocap_help()
        verb = parts[1] if len(parts) > 1 else ""
        name = "default"
        rest = parts[2:]
        if verb in ("load", "status", "pose", "render") and rest \
                and not rest[0].startswith("{"):
            name = rest[0]
            rest = rest[1:]
        try:
            if verb == "list":
                sessions = list_sessions()
                lines = ["rig sessions:"]
                for entry in sessions:
                    lines.append(
                        f"  {entry['name']}: {entry['rig']} "
                        f"({entry['bones']} bones, "
                        f"remap {entry['remap_dialect']}, "
                        f"{entry['width']}x{entry['height']} "
                        f"@{entry['fps']}fps)")
                return "\n".join(lines)
            if verb == "load":
                session = load_session(name)
                return (f"rig {session.rig.name} loaded as {name} "
                        f"({len(session.rig.bones)} bones, remap "
                        f"{session.remap.dialect})")
            if verb == "status":
                session = get_session(name)
                info = dict({"name": name}, **session.to_dict())
                return json.dumps(info, indent=2)
            session = get_session(name)
            if verb == "pose":
                face = None
                frame = None
                for token in rest:
                    if token.startswith("{"):
                        try:
                            raw = json.loads(token)
                        except ValueError:
                            return "pose frame/face must be JSON"
                        if isinstance(raw, dict) and (
                                "blendshapes" in raw
                                or "viseme" in raw
                                or "text" in raw):
                            face = FaceInput(
                                blendshapes=raw.get("blendshapes"),
                                viseme=raw.get("viseme"),
                                text=raw.get("text"),
                                laughter=float(
                                    raw.get("laughter") or 0.0))
                        elif isinstance(raw, dict):
                            frame = SkeletonFrame(values={
                                str(k): tuple(float(v) for v in vals)
                                for k, vals in raw.items()})
                pose = session.drive_frame(frame, face)
                return (f"pose f:{pose.seq} viseme {pose.viseme} "
                        f"mouth={tuple(round(v, 2) for v in pose.mouth)}"
                        + (f" unmapped: {', '.join(pose.unmapped)}"
                           if pose.unmapped else "")
                        + (" STALE" if pose.stale else ""))
            if verb == "render":
                pose = session.drive_frame(None)
                rgb = session.render_frame(pose)
                return (f"pose f:{pose.seq} rendered "
                        f"({len(rgb)} bytes RGB, "
                        f"{session.width}x{session.height}) — "
                        f"use nm mocap rig render --out PATH.png "
                        f"for the file")
            if verb in ("drive", "export"):
                if not rest:
                    return f"/mocap rig {verb} needs a BVH path"
                take = load_take_bvh(rest[0])
                text = " ".join(rest[1:]) or None
                posed = session.drive_take(
                    take["skeleton"], take["frames"], text=text)
                if verb == "export":
                    result = session.export_bvh(
                        posed, f"rig-drive-{name}.bvh", store=_STORE)
                    return (f"exported {result['frames']} frames to "
                            f"{result['path']}")
                return (f"drove {len(posed)} frames "
                        f"(visemes: "
                        f"{', '.join(p.viseme for p in posed[:8])}"
                        f"{'…' if len(posed) > 8 else ''})")
        except RigError as exc:
            record_failure(
                "runtime_mocap.RuntimeMocapMixin._control_mocap", exc)
            return f"rig: {exc}"
        except MocapError as exc:
            record_failure(
                "runtime_mocap.RuntimeMocapMixin._control_mocap", exc)
            return f"ingest: {exc}"
        except Exception as exc:  # noqa: BLE001 - chat never dies
            record_failure(
                "runtime_mocap.RuntimeMocapMixin._control_mocap", exc)
            return f"mocap failed: {exc}"
        return self._control_mocap_help()

    def _control_mocap_ingest(self, parts: list[str]) -> str:
        """``/mocap ingest …`` — the polling pipeline from chat.

        Chat takes the simple shape only: a source name and a
        session name. The full option set lives on the ``mocap``
        brain tool and ``nm mocap ingest``. Named refusals (no
        device, no pyzmq, unknown session) come back as plain
        words, never a traceback.
        """
        from ...hardware.kinect import KinectError
        from ...media.mocap.errors import MocapError
        from ...media.mocap.ingest import (
            IngestOptions,
            drop_session,
            get_session,
            list_sessions,
            open_session,
            open_source,
            publish_one_frame,
        )
        from ...tools.mocap import _STORE

        if not parts:
            return self._control_mocap_help()
        verb = parts[0]
        rest = parts[1:]
        try:
            if verb == "status":
                if rest:
                    status = get_session(rest[0]).status()
                    return self._ingest_status_line(status)
                sessions = list_sessions()
                if not sessions:
                    return "no ingest sessions running"
                return "\n".join(
                    self._ingest_status_line(s)
                    for s in sessions)
            if verb == "publish":
                source = rest[0] if rest else "fixture"
                receipt = publish_one_frame(
                    _STORE, source, IngestOptions())
                if receipt.get("frame") is None:
                    return (f"ingest publish dropped: "
                            f"{receipt.get('reason', 'unknown')}")
                return (f"ingest publish: frame {receipt['frame']} "
                        f"{receipt['points']} points "
                        f"({receipt['color']})")
            if verb == "stop":
                name = rest[0] if rest else "default"
                drop_session(name)
                return f"ingest session {name} stopped"
            if verb == "tick":
                name = rest[0] if rest else "default"
                count = int(rest[1]) if len(rest) > 1 else 1
                session = get_session(name)
                receipts = session.run(count)
                done = [r for r in receipts
                        if r.get("frame") is not None]
                return (f"ingest {name}: {len(done)}/{count} frames, "
                        f"{sum(r['points'] for r in done)} points")
            if verb == "start":
                source = rest[0] if rest else "fixture"
                name = rest[1] if len(rest) > 1 else "default"
                options = IngestOptions()
                src = open_source(source, options)
                try:
                    session = open_session(name, src, options,
                                           store=_STORE)
                except Exception:
                    src.close()
                    raise
                status = session.start()
                return (f"ingest session {name} started "
                        f"({source} @ {options.fps}fps, "
                        f"ZeroMQ {options.zmq_endpoint}) — "
                        f"state {status['state']}")
        except MocapError as exc:
            record_failure(
                "runtime_mocap.RuntimeMocapMixin"
                "._control_mocap_ingest", exc)
            return f"ingest: {exc}"
        except KinectError as exc:
            record_failure(
                "runtime_mocap.RuntimeMocapMixin"
                "._control_mocap_ingest", exc)
            return f"ingest: {exc}"
        except (ValueError, TypeError) as exc:
            return f"ingest: {exc}"
        return self._control_mocap_help()

    @staticmethod
    def _ingest_status_line(status: dict) -> str:
        drops = status["drops"]
        return (f"{status['name']}: {status['state']} "
                f"({status['source']}) frames={status['frames']} "
                f"points={status['points_total']} "
                f"last={status['points_last']} "
                f"drops timeouts={drops['timeouts']} "
                f"conversion={drops['conversion_errors']} "
                f"store={drops['store_errors']} "
                f"zmq={drops['zmq_errors']} "
                f"fps {status['fps_achieved']}/{status['fps_target']}"
                + (f" last error: {status['last_error']}"
                   if status["last_error"] else ""))

    def _control_mocap_help(self) -> str:
        return (
            "/mocap status | rig list | rig load [name] | "
            "rig status [name] | rig pose [name] [frame JSON] | "
            "rig render [name] | rig drive <bvh> [text] | "
            "rig export <bvh> [text] | ingest start [live|fixture] "
            "[name] | ingest stop [name] | ingest status [name] | "
            "ingest tick [name] [n] | ingest publish [live|fixture]")

    # ── cloudcam: /cloudcam ───────────────────────────────────────────
    def _control_cloudcam(self, tail: str, chat_key: str,
                          message: Any = None) -> str:
        """Streamed point clouds, live from any chat.

        /cloudcam start [name] [source]    open a stream
        /cloudcam stop [name]               stop the poller
        /cloudcam status [name]             one session's state
        /cloudcam list                      loaded sessions
        /cloudcam tick [name]               stage the latest frame
        /cloudcam export [name]             latest frame as PLY
        /cloudcam render [name]             latest frame as a PNG
        """
        from ...media.cloudcam import (
            get_session,
            list_sessions,
            load_session,
            render_projection,
            write_ply,
        )
        from ...media.cloudcam.errors import CloudCamError

        parts = (tail or "").split()
        if not parts:
            return self._control_cloudcam_help()
        verb = parts[0]
        rest = parts[1:]
        name = "default"
        if verb in ("start", "stop", "status", "tick", "export",
                    "render") and rest:
            name = rest[0]
            rest = rest[1:]
        try:
            if verb == "list":
                sessions = list_sessions()
                lines = ["cloudcam sessions:"]
                for entry in sessions:
                    lines.append(
                        f"  {entry['name']}: {entry['source']} "
                        f"{'running' if entry['running'] else 'stopped'} "
                        f"({entry['ring_frames_received']} frames, "
                        f"{entry['ring_dropped_frames']} dropped, "
                        f"last frame {entry['latest_points']} points)")
                return "\n".join(lines)
            if verb == "start":
                source = (rest[0] if rest else "idle").strip().lower()
                store = None
                if source == "store":
                    from ...tools.mocap import _STORE
                    store = _STORE
                session = load_session(name, source=source, store=store)
                info = session.start()
                return (f"cloudcam {name} streaming from "
                        f"{info['source']} "
                        f"({info['grid']}, ring {info['ring_capacity']})")
            if verb == "stop":
                session = get_session(name)
                session.stop()
                return f"cloudcam {name} stopped"
            if verb == "status":
                return json.dumps(get_session(name).status(), indent=2)
            if verb in ("tick", "stage"):
                receipt = get_session(name).tick()
                if receipt["placeholder"]:
                    return (f"cloudcam {name}: no frame yet — "
                            f"placeholder grid "
                            f"({receipt['width']}x{receipt['height']})")
                return (f"cloudcam {name}: frame {receipt['frame_seq']} "
                        f"staged — {receipt['packed']} points, "
                        f"{receipt['hidden']} hidden, "
                        f"{receipt['dropped']} dropped "
                        f"({receipt['ring_dropped_frames']} dropped in "
                        f"the ring)")
            if verb in ("export", "render"):
                session = get_session(name)
                points = session.latest_points()
                if not points:
                    return (f"cloudcam {name} has staged no frame yet "
                            f"— start a source and tick first")
                if verb == "export":
                    result = write_ply(points, f"cloudcam-{name}.ply")
                    return (f"exported {result['points']} points to "
                            f"{result['path']}")
                result = render_projection(points,
                                           f"cloudcam-{name}.png")
                return (f"rendered {result['plotted']} points to "
                        f"{result['path']} ({result['view']} view)")
        except CloudCamError as exc:
            record_failure(
                "runtime_mocap.RuntimeMocapMixin._control_cloudcam", exc)
            return f"cloudcam: {exc}"
        except Exception as exc:  # noqa: BLE001 - chat never dies
            record_failure(
                "runtime_mocap.RuntimeMocapMixin._control_cloudcam", exc)
            return f"cloudcam failed: {exc}"
        return self._control_cloudcam_help()

    def _control_cloudcam_help(self) -> str:
        return (
            "/cloudcam start [name] [source zmq|store|file|idle] | "
            "stop [name] | status [name] | list | tick [name] | "
            "export [name] | render [name]")

    # ── show: /live ───────────────────────────────────────────────
    def _control_live(self, tail: str, chat_key: str,
                      message: Any = None) -> str:
        """One live run of mocap + rig + cloudcam, through named
        states.

        /live create [name]                load a show
        /live list                         loaded shows
        /live status [name]                one show's state
        /live start [name]                 arm, then go live
        /live pause [name]                 hold the show
        /live resume [name]                resume a paused show
        /live stop [name]                  stop the show
        /live tick [name]                  one schedule step
        /live run [name] [ticks]           run bounded ticks (max 150)
        /live events [name]                the show's journal
        /live export [name]                driven poses as BVH
        /live load-take [name] <path>      load a BVH/JSON take
        /live attach-rig [show] <rig>      attach a rig session
        /live attach-cloudcam [show] <cam> attach a cloudcam session
        /live drop [name]                  stop and drop a show
        """
        from ...media.show import (
            drop_show,
            get_show,
            list_shows,
            load_show,
        )
        from ...media.show.errors import ShowError
        from ...tools.mocap import _STORE
        from ...tools.show import ensure_defaults

        parts = (tail or "").split()
        if not parts:
            return self._control_live_help()
        verb = parts[0]
        rest = parts[1:]
        name = "default"
        if verb in ("create", "start", "pause", "resume", "stop",
                    "tick", "status", "events", "export", "load-take",
                    "drop", "attach-rig", "attach-cloudcam") and rest:
            name = rest[0]
            rest = rest[1:]
        try:
            if verb == "list":
                shows = list_shows()
                lines = ["shows:"]
                for entry in shows:
                    lines.append(
                        f"  {entry['name']}: {entry['state']} "
                        f"(tick {entry['tick']}, "
                        f"{entry['posed']} poses, "
                        f"rig {entry['rig'] or 'none'})")
                return "\n".join(lines)
            if verb == "create":
                load_show(name, _STORE)
                return (f"show {name} created (idle — /live start "
                        f"{name} to go live)")
            if verb == "run":
                # /live run [name] [ticks] — a bare number is ticks
                # for the default show, not a show name.
                ticks = 30
                if rest and rest[0].isdigit():
                    ticks = int(rest[0])
                elif rest:
                    name = rest[0]
                    if len(rest) > 1:
                        ticks = int(rest[1])
                if ticks > 150:
                    return ("show run is capped at 150 ticks in "
                            "chat — use nm show run for longer runs")
                show = get_show(name)
                attached = ensure_defaults(show)
                status = show.run(ticks)
                extra = (f" (auto-attached {', '.join(attached)})"
                         if attached else "")
                return (f"show {name} ran {ticks} ticks — "
                        f"{status['state']} (tick {status['tick']}, "
                        f"{status['posed']} poses){extra}")
            if verb not in ("status", "start", "pause", "resume",
                            "stop", "tick", "events", "export",
                            "load-take", "attach-rig", "attach-cloudcam",
                            "drop"):
                return self._control_live_help()
            show = get_show(name)
            if verb == "status":
                return json.dumps(show.status(), indent=2)
            if verb == "start":
                attached = ensure_defaults(show)
                status = show.start()
                extra = (f" (auto-attached {', '.join(attached)})"
                         if attached else "")
                return (f"show {name} live{extra} — "
                        f"tick {status['tick']}")
            if verb == "pause":
                show.pause()
                return f"show {name} paused"
            if verb == "resume":
                show.resume()
                return f"show {name} live again"
            if verb == "stop":
                show.stop()
                return f"show {name} stopped"
            if verb == "tick":
                receipt = show.step()
                pose = receipt["pose"]
                return (
                    f"show {name}: tick {receipt['tick']} "
                    f"({receipt['state']})"
                    + (f" — pose f:{pose['seq']} viseme "
                       f"{pose['viseme']}" if pose else "")
                    + (" — holding" if receipt["held"] else ""))
            if verb == "events":
                events = show.events(10)
                lines = [f"show {name} events:"]
                for event in events:
                    lines.append(f"  t{event['tick']} {event['kind']}")
                return "\n".join(lines)
            if verb == "export":
                result = show.export(f"show-{name}.bvh")
                return (f"exported {result['frames']} frames to "
                        f"{result['path']}")
            if verb == "load-take":
                if not rest:
                    return "/live load-take [name] <path.bvh|path.json>"
                info = show.load_take(rest[0])
                return (f"show {name} take loaded: "
                        f"{info['total']} frames from {info['name']}")
            if verb == "attach-rig":
                if not rest:
                    return "/live attach-rig [show] <rig session>"
                from ...media.rig import (
                    get_session as get_rig_session,
                )
                result = show.attach_rig(get_rig_session(rest[0]))
                return f"show {name} rig: {result['rig']}"
            if verb == "attach-cloudcam":
                if not rest:
                    return ("/live attach-cloudcam [show] "
                            "<cloudcam session>")
                from ...media.cloudcam import (
                    get_session as get_cc_session,
                )
                result = show.attach_cloudcam(
                    get_cc_session(rest[0]))
                return f"show {name} cloudcam: {result['cloudcam']}"
            if verb == "drop":
                drop_show(name)
                return f"show {name} dropped"
        except ShowError as exc:
            record_failure(
                "runtime_mocap.RuntimeMocapMixin._control_live", exc)
            return f"live: {exc}"
        except Exception as exc:  # noqa: BLE001 - chat never dies
            record_failure(
                "runtime_mocap.RuntimeMocapMixin._control_live", exc)
            return f"live failed: {exc}"
        return self._control_live_help()

    def _control_live_help(self) -> str:
        return (
            "/live create [name] | list | status [name] | start [name] "
            "| pause [name] | resume [name] | stop [name] | tick [name] "
            "| run [name] [ticks] | events [name] | export [name] | "
            "load-take [name] <path> | attach-rig [show] <rig> | "
            "attach-cloudcam [show] <cam> | drop [name]")
