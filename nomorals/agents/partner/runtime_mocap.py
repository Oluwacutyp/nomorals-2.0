"""RuntimeMocapMixin: PartnerRuntime command group (mocap + rig)."""

from __future__ import annotations

import json
from pathlib import Path
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
        /mocap mvn listen [name] [host] [port]    listen for MVN datagrams
        /mocap mvn send [name] [host] [port]      send one store frame
        /mocap mvn replay <pcap> [name]           replay a capture to a host
        /mocap mvn stop [name]                    stop an MVN session
        /mocap mvn status [name]                  MVN sessions
        /mocap mvn decode <pcap>                  decode a capture, no UDP
        /mocap mvn names                          the MVN segment tables
        /mocap kinect record <out.mkv> [frames N] [tag KEY=VALUE ...]
                                                  record a live Kinect
        /mocap kinect playback <file.mkv> [op] [count N]
                                                  read a recording back
        /mocap kinect status                      probe the Kinect SDKs
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
        if parts[0] == "mvn":
            return self._control_mocap_mvn(parts[1:], chat_key)
        if parts[0] == "kinect":
            return self._control_mocap_kinect(parts[1:], chat_key)
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
            "ingest tick [name] [n] | ingest publish [live|fixture] | mvn listen [name] [host] [port] | mvn send [name] [host] [port] | mvn replay <pcap> [name] | mvn stop [name] | mvn status [name] | mvn decode <pcap> | mvn names | kinect record <out.mkv> [frames N] [tag KEY=VALUE ...] | kinect playback <file.mkv> [op] [count N] | kinect status")

    def _control_mocap_mvn(self, parts: list[str],
                           chat_key: str) -> str:
        """Xsens MVN wire, live from any chat — the same sessions
        the ``mocap`` brain tool and ``nm mocap mvn`` drive."""
        from ...media.mocap.errors import MocapError
        from ...tools.mocap import _mvn_action

        if not parts:
            return self._control_mocap_help()
        verb = parts[0]
        rest = parts[1:]
        name = "default"
        if verb in ("listen", "send", "replay", "stop", "status") \
                and rest:
            name = rest[0]
            rest = rest[1:]
        try:
            if verb == "listen":
                host = rest[0] if rest else "0.0.0.0"
                port = int(rest[1]) if len(rest) > 1 else 9763
                info = _mvn_action("mvn_listen", name,
                                   {"host": host, "port": port}, [])
                return (f"mvn {name} listening on "
                        f"{info['target']}")
            if verb == "send":
                host = rest[0] if rest else "127.0.0.1"
                port = int(rest[1]) if len(rest) > 1 else 9763
                receipt = _mvn_action(
                    "mvn_send", name,
                    {"host": host, "port": port}, [])
                note = (f" unmapped: {', '.join(receipt['unmapped'])}"
                        if receipt["unmapped"] else "")
                return (f"mvn {name} sent sample "
                        f"{receipt['sample']} "
                        f"({receipt['segments']} segments, "
                        f"{receipt['sent']} bytes){note}")
            if verb == "replay":
                if not rest:
                    return "/mocap mvn replay needs a pcap path"
                info = _mvn_action("mvn_replay", name, {}, [rest[0]])
                return (f"mvn {name} replaying {rest[0]} to "
                        f"{info['target']}")
            if verb == "stop":
                _mvn_action("mvn_stop", name, {}, [])
                return f"mvn {name} stopped"
            if verb == "status":
                if name != "default" or rest:
                    info = _mvn_action("mvn_status", name, {}, [])
                    return json.dumps(info, indent=2)
                info = _mvn_action("mvn_status", "", {}, [])
                sessions = info["sessions"]
                if not sessions:
                    return "mvn sessions: none"
                lines = ["mvn sessions:"]
                for entry in sessions:
                    lines.append(
                        f"  {entry['name']}: {entry['kind']} "
                        f"{entry['target']} "
                        f"({entry['counts']})")
                return "\n".join(lines)
            if verb == "decode":
                if not rest:
                    return "/mocap mvn decode needs a pcap path"
                info = _mvn_action("mvn_decode", "", {}, [rest[0]])
                return (f"mvn decode {rest[0]}: "
                        f"{info['pcap_counts']['packets']} packets, "
                        f"{info['frames']} frames, "
                        f"{info['tables']} name tables, "
                        f"{info['dropped']} dropped")
            if verb == "names":
                info = _mvn_action("mvn_names", "", {}, [])
                return (f"mvn wire table: "
                        f"{len(info['wire_2022'])} segments "
                        f"(Pelvis 1 … LeftFifthTip 63); spec table: "
                        f"{len(info['spec_23_segment'])} segments "
                        f"(LeftFoot 23, LeftToe 24, no ID 22)")
        except MocapError as exc:
            record_failure(
                "runtime_mocap.RuntimeMocapMixin._control_mocap_mvn",
                exc)
            return f"mvn: {exc}"
        except (OSError, ValueError) as exc:
            record_failure(
                "runtime_mocap.RuntimeMocapMixin._control_mocap_mvn",
                exc)
            return f"mvn: {exc}"
        except Exception as exc:  # noqa: BLE001 - chat never dies
            record_failure(
                "runtime_mocap.RuntimeMocapMixin._control_mocap_mvn",
                exc)
            return f"mocap failed: {exc}"
        return self._control_mocap_help()

    def _control_mocap_kinect(self, parts: list[str],
                              chat_key: str) -> str:
        """Kinect record/playback, live from any chat — the same
        sessions the ``mocap`` brain tool and ``nm mocap kinect``
        drive."""
        from ...hardware.kinect.errors import KinectError
        from ...tools.mocap import _kinect_action

        if not parts:
            return self._control_mocap_help()
        verb = parts[0]
        rest = parts[1:]

        def _kv(tokens: list[str]) -> dict[str, str]:
            values: dict[str, str] = {}
            for token in tokens:
                if "=" not in token:
                    raise ValueError(
                        f"expected KEY=VALUE, got {token!r}")
                key, value = token.split("=", 1)
                values[key.strip()] = value
            return values

        try:
            if verb == "record":
                if not rest:
                    return "/mocap kinect record needs an .mkv path"
                out = rest[0]
                kv = _kv(rest[1:])
                tags: dict[str, str] = {}
                for token in rest[1:]:
                    if token.startswith("tag="):
                        key, value = token[4:].split("=", 1)
                        tags[key.strip()] = value
                params = {"frames": int(kv.get("frames") or 30),
                          "device_index": int(kv.get("device") or 0),
                          "timeout_ms": int(kv.get("timeout") or 1000),
                          "imu": kv.get("imu", "on").lower() != "off",
                          "tags": tags,
                          "color_format": kv.get("color-format",
                                                 "MJPG"),
                          "color_resolution": kv.get(
                              "color-resolution", "720P"),
                          "depth_mode": kv.get(
                              "depth-mode", "WFOV_2X2BINNED"),
                          "camera_fps": kv.get("camera-fps", "30")}
                summary = _kinect_action("kinect_record", "", params,
                                         [], out)
                return (f"recorded {summary['frames_written']} "
                        f"frames ({summary['imu_samples']} IMU) to "
                        f"{summary['path']}")
            if verb == "playback":
                if not rest:
                    return "/mocap kinect playback needs an .mkv path"
                path = rest[0]
                tokens = rest[1:]
                op = "info"
                if tokens and "=" not in tokens[0]:
                    op = tokens[0]
                    tokens = tokens[1:]
                kv = _kv(tokens)
                params = {"op": op,
                          "count": int(kv.get("count") or 5),
                          "direction": kv.get("direction", "next"),
                          "origin": kv.get("origin", "begin"),
                          "name": kv.get("name", ""),
                          "track": kv.get("track", "")}
                if "seek" in kv:
                    params["seek_usec"] = int(kv["seek"])
                result = _kinect_action("kinect_playback", "", params,
                                        [path], "")
                if op == "info":
                    return json.dumps(result, indent=2)
                if op == "tracks":
                    lines = [f"tracks in {path} "
                             f"({result['track_count']}):"]
                    for entry in result["tracks"]:
                        line = f"  {entry['name']}"
                        if entry.get("builtin"):
                            line += " (builtin)"
                        if entry.get("video_settings"):
                            vs = entry["video_settings"]
                            line += (f" {vs['width']}x{vs['height']} "
                                     f"@{vs['framerate']}fps")
                        lines.append(line)
                    return "\n".join(lines)
                if op == "read":
                    lines = [f"read {result['captures_returned']} "
                             f"captures from {path}"
                             + (" (EOF)" if result["eof"] else "")]
                    for capture in result["captures"]:
                        depth = capture["depth"]
                        color = capture["color"]
                        stamp = (depth["device_timestamp_usec"]
                                 if depth else None)
                        bits = [f"@{stamp}us"]
                        if color:
                            bits.append(f"color {color['width']}x"
                                        f"{color['height']}")
                        if depth:
                            bits.append(f"depth {depth['width']}x"
                                        f"{depth['height']}")
                        lines.append("  " + " ".join(bits))
                    return "\n".join(lines)
                if op == "imu":
                    lines = [f"read {result['samples_returned']} "
                             f"IMU samples from {path}"
                             + (" (EOF)" if result["eof"] else "")]
                    for sample in result["samples"]:
                        lines.append(
                            f"  @{sample['device_timestamp_usec']}us "
                            f"acc={sample['acc']} "
                            f"gyro={sample['gyro']}")
                    return "\n".join(lines)
                if op == "data":
                    lines = [f"read {result['blocks_returned']} "
                             f"blocks from {result['track']} in "
                             f"{path}"
                             + (" (EOF)" if result["eof"] else "")]
                    for block in result["blocks"]:
                        lines.append(
                            f"  @{block['device_timestamp_usec']}us "
                            f"{block['size_bytes']} bytes "
                            f"sha256:{block['sha256'][:12]}")
                    return "\n".join(lines)
                if op == "tag":
                    return (f"{result['name']}: "
                            f"{result['value'] or '(not set)'}")
                if op == "attachment":
                    if result.get("value") is None:
                        return (f"attachment {result['name']}: "
                                "(not present)")
                    return (f"attachment {result['name']}: "
                            f"{result['size_bytes']} bytes, "
                            f"sha256:{result['sha256'][:12]}")
                return json.dumps(result, indent=2)
            if verb == "status":
                from ...hardware.kinect import probe
                report = probe()
                return report.summary()
        except KinectError as exc:
            record_failure(
                "runtime_mocap.RuntimeMocapMixin._control_mocap_kinect",
                exc)
            return f"kinect: {exc}"
        except (OSError, ValueError) as exc:
            record_failure(
                "runtime_mocap.RuntimeMocapMixin._control_mocap_kinect",
                exc)
            return f"kinect: {exc}"
        except Exception as exc:  # noqa: BLE001 - chat never dies
            record_failure(
                "runtime_mocap.RuntimeMocapMixin._control_mocap_kinect",
                exc)
            return f"mocap failed: {exc}"
        return self._control_mocap_help()

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

    # ── facemcap: /facemcap ───────────────────────────────────────
    def _control_facemcap(self, tail: str, chat_key: str,
                          message: Any = None) -> str:
        """ARKit LiveLink CSV facial capture, as face takes.

        /facemcap load <path> [name]   load a take (manifest,
                                        face-take JSON or CSV)
        /facemcap list                  loaded takes
        /facemcap status [name]         one take's state
        /facemcap drive <name> [rig]    drive the whole take
                                        through a rig session
        /facemcap frame <name> [n]      one frame as data
        /facemcap export <name>         the take as a face-take
                                        JSON file
        /facemcap render <name> [n]     one frame as a PNG
        /facemcap attach <name> <show>  attach the take to a show
        /facemcap drop [name]           drop a take
        """
        from ...media.facemcap import (
            drive_take,
            drop_take,
            get_take,
            list_takes,
            load_take,
            render_frame,
        )
        from ...media.facemcap.errors import FaceMocapError

        parts = (tail or "").split()
        if not parts:
            return self._control_facemcap_help()
        verb = parts[0]
        rest = parts[1:]
        name = "default"
        if verb in ("status", "drive", "frame", "export", "render",
                    "attach", "drop") and rest:
            name = rest[0]
            rest = rest[1:]
        try:
            if verb == "list":
                takes = list_takes()
                lines = ["facemcap takes:"]
                for entry in takes:
                    lines.append(
                        f"  {entry['name']}: {entry['frames']} "
                        f"frames, {entry['fps']}fps, "
                        f"{entry['duration_s']}s, root "
                        f"{entry['root'] or '(none)'}, "
                        f"{len(entry['morphs'])} morphs"
                        + (f", {len(entry['unmapped'])} unmapped"
                           if entry['unmapped'] else ""))
                return "\n".join(lines)
            if verb == "load":
                if not rest:
                    return self._control_facemcap_help()
                path = rest[0]
                if len(rest) > 1:
                    name = rest[1]
                take = load_take(name, path)
                return (f"facemcap {take.name} loaded: "
                        f"{take.total} frames, {take.fps}fps, "
                        f"{take.duration_s:.2f}s, root "
                        f"{take.root_name or '(none)'}, "
                        f"{len(take.morph_names)} morphs"
                        + (f", {len(take.unmapped)} unmapped: "
                           f"{', '.join(take.unmapped)}"
                           if take.unmapped else ""))
            if verb == "status":
                take = get_take(name)
                return json.dumps(take.to_dict(), indent=2)
            if verb == "drive":
                take = get_take(name)
                from ...media.rig import get_session as get_rig
                rig_name = rest[0] if rest else "default"
                session = get_rig(rig_name)
                posed = drive_take(session, take)
                return (f"facemcap {take.name} driven through rig "
                        f"{session.rig.name}: {len(posed)} frames, "
                        f"{take.fps}fps")
            if verb == "frame":
                take = get_take(name)
                index = int(rest[0]) if rest else 0
                if index < 0 or index >= take.total:
                    return (f"facemcap {take.name} has frames "
                            f"0..{take.total - 1}, got {index}")
                return json.dumps(
                    dict(take.frames[index].to_dict(),
                         take=take.name, index=index,
                         total=take.total), indent=2)
            if verb == "export":
                take = get_take(name)
                target = f"facemcap-{name}.json"
                Path(target).write_text(
                    json.dumps(take.to_json(), indent=1),
                    encoding="utf-8")
                return (f"facemcap {take.name} exported: "
                        f"{take.total} frames to "
                        f"{Path(target).resolve()}")
            if verb == "render":
                take = get_take(name)
                from ...media.rig import get_session as get_rig
                index = int(rest[0]) if rest else 0
                rig_name = rest[1] if len(rest) > 1 else "default"
                session = get_rig(rig_name)
                result = render_frame(
                    session, take, index,
                    f"facemcap-{name}-{index}.png")
                return (f"facemcap {take.name} frame {index} "
                        f"rendered to {result['path']}")
            if verb == "attach":
                if not rest:
                    return self._control_facemcap_help()
                take = get_take(name)
                from ...media.show import get_show
                show = get_show(rest[0])
                result = show.attach_face_take(take.player())
                return (f"facemcap {take.name} attached to show "
                        f"{show.name}: {result['face_take']}")
            if verb == "drop":
                result = drop_take(name)
                return (f"facemcap {name} dropped"
                        + (f" ({result['frames']} frames)"
                           if result['frames'] else ""))
        except FaceMocapError as exc:
            record_failure(
                "runtime_mocap.RuntimeMocapMixin._control_facemcap",
                exc)
            return f"facemcap: {exc}"
        except Exception as exc:  # noqa: BLE001 - chat never dies
            record_failure(
                "runtime_mocap.RuntimeMocapMixin._control_facemcap",
                exc)
            return f"facemcap failed: {exc}"
        return self._control_facemcap_help()

    def _control_facemcap_help(self) -> str:
        return (
            "/facemcap load <path> [name] | list | status [name] | "
            "drive <name> [rig] | frame <name> [n] | export <name> | "
            "render <name> [n] | attach <name> <show> | drop [name]")

    # ── facemesh: /facemesh ───────────────────────────────────────
    def _control_facemesh(self, tail: str, chat_key: str,
                          message: Any = None) -> str:
        """MediaPipe Face Mesh capture, as face takes.

        /facemesh capture <path> [name]  run a video through
                                        Face Mesh, register the
                                        take in the shared registry
        /facemesh landmarks <path> [n]   landmarks + blendshapes
                                        from an image or the first
                                        n frames of a video
        /facemesh status                 mediapipe / model /
                                        ffmpeg availability
        """
        from ...media.facemcap.registry import register_take
        from ...media.facemesh import (
            FaceMeshError,
            capture_image,
            capture_video,
            mediapipe_status,
        )

        parts = (tail or "").split()
        if not parts:
            return self._control_facemesh_help()
        verb = parts[0]
        rest = parts[1:]
        try:
            if verb == "status":
                status = mediapipe_status()
                lines = ["facemesh status:"]
                lines.append(
                    "  mediapipe: "
                    + (f"{status['mediapipe_version']}"
                       if status["mediapipe"] else "not installed"))
                lines.append(
                    "  model: "
                    + (f"{status['model_bytes']} bytes"
                       if status["model_ok"] else "not downloaded"))
                lines.append(
                    "  ffmpeg: "
                    + ("yes" if status["ffmpeg"] else "no"))
                return "\n".join(lines)
            if verb == "capture":
                if not rest:
                    return self._control_facemesh_help()
                name = rest[1] if len(rest) > 1 else "default"
                result = capture_video(rest[0], name=name)
                register_take(name, result.take)
                return (f"facemesh {result.take.name} captured: "
                        f"{result.take.total} frames, "
                        f"{result.take.fps}fps, "
                        f"{result.take.duration_s:.2f}s, "
                        f"{result.frames_read} read, "
                        f"{result.frames_skipped} skipped "
                        "(no face)")
            if verb == "landmarks":
                if not rest:
                    return self._control_facemesh_help()
                path = rest[0]
                suffix = path.rsplit(".", 1)[-1].lower()
                if suffix in ("mp4", "mov", "mkv", "avi", "webm"):
                    n = int(rest[1]) if len(rest) > 1 else 1
                    result = capture_video(path, max_frames=n)
                    frame = result.take.frames[0]
                    return json.dumps({
                        "source": path,
                        "landmarks": result.first_landmarks,
                        "blendshapes": frame.blendshapes,
                        "bones": frame.bones}, indent=2)
                result = capture_image(path)
                frame = result.take.frames[0]
                return json.dumps({
                    "source": path,
                    "landmarks": result.first_landmarks,
                    "blendshapes": frame.blendshapes,
                    "bones": frame.bones}, indent=2)
        except FaceMeshError as exc:
            record_failure(
                "runtime_mocap.RuntimeMocapMixin._control_facemesh",
                exc)
            return f"facemesh: {exc}"
        except Exception as exc:  # noqa: BLE001 - chat never dies
            record_failure(
                "runtime_mocap.RuntimeMocapMixin._control_facemesh",
                exc)
            return f"facemesh failed: {exc}"
        return self._control_facemesh_help()

    def _control_facemesh_help(self) -> str:
        return ("/facemesh capture <path> [name] | "
                "landmarks <path> [n] | status")
    # ── vtuber show: /vtuber ───────────────────────────────────────
    def _control_vtuber(self, tail: str, chat_key: str,
                        message: Any = None) -> str:
        """A scripted character performance on a stage.

        /vtuber character <path.json> [name]  load a character
        /vtuber act <path.vtact.json> [name]   load an act
        /vtuber perform <show> <act> <character>  perform an act
        /vtuber cue <show> <kind> <value> [seconds]  an audience cue
        /vtuber live <show> <receiver>        attach a live MVN
        /vtuber live-detach <show>            detach the live MVN
        /vtuber status [show]                  one show's performance
        /vtuber tick [show]                     one schedule step
        /vtuber stop [show]                     detach the act, stop
        /vtuber list                           characters, acts, shows
        /vtuber drop [show]                     stop and drop a show
        """
        from ...media.show import (
            drop_show,
            get_show,
            list_shows,
            load_show,
        )
        from ...media.show.errors import ShowError
        from ...media.vtuber import (
            attach_live,
            attach_performance,
            detach_live,
            detach_performance,
            get_act,
            get_character,
            list_acts,
            list_characters,
            load_act,
            load_character,
        )
        from ...media.vtuber.errors import VtuberError
        from ...tools.mocap import _STORE
        from ...tools.show import ensure_defaults

        parts = (tail or "").split()
        if not parts:
            return self._control_vtuber_help()
        verb = parts[0]
        rest = parts[1:]
        name = "default"
        if verb in ("status", "tick", "stop", "drop", "live",
                    "live-detach", "cue", "perform") and rest:
            name = rest[0]
            rest = rest[1:]
        try:
            if verb == "character":
                if not rest:
                    return self._control_vtuber_help()
                load_name = rest[1] if len(rest) > 1 else ""
                character = load_character(load_name, rest[0])
                return (f"vtuber {character.name} loaded: "
                        f"{len(character.expressions)} expressions, "
                        f"{len(character.gestures)} gestures, "
                        f"rig {character.rig}")
            if verb == "act":
                if not rest:
                    return self._control_vtuber_help()
                load_name = rest[1] if len(rest) > 1 else ""
                act = load_act(load_name, rest[0])
                return (f"vtuber {act.name} loaded: "
                        f"{len(act.beats)} beats, character "
                        f"{act.character}, {act.fps}fps")
            if verb == "list":
                characters = [c["name"] for c in list_characters()]
                acts = [a["name"] for a in list_acts()]
                lines = ["vtuber characters: "
                         + (", ".join(characters) or "(none)"),
                         "vtuber acts: "
                         + (", ".join(acts) or "(none)"),
                         "shows:"]
                for entry in list_shows():
                    lines.append(
                        f"  {entry['name']}: {entry['state']} "
                        f"(tick {entry['tick']}, "
                        f"{entry['posed']} poses, "
                        f"rig {entry['rig'] or 'none'})")
                return "\n".join(lines)
            if verb == "perform":
                if len(rest) < 2:
                    return self._control_vtuber_help()
                try:
                    show = get_show(name)
                except ShowError:
                    show = load_show(name, _STORE)
                attached = ensure_defaults(show)
                if getattr(show, "_vtuber_player", None) is not None:
                    detach_performance(show)
                act = get_act(rest[0])
                character = get_character(rest[1])
                attach_performance(show, act, character)
                status = show.start()
                extra = (f" (auto-attached "
                         f"{', '.join(attached)})" if attached else "")
                return (f"vtuber {name} performing {act.name} as "
                        f"{character.name} — {status['state']}"
                        f" (tick {status['tick']}){extra}")
            if verb not in ("status", "tick", "cue", "live",
                            "live-detach", "stop", "drop"):
                return self._control_vtuber_help()
            show = get_show(name)
            if verb == "status":
                status = show.status()
                player = getattr(show, "_vtuber_player", None)
                live = getattr(show, "_live_mvn", None)
                if player is not None:
                    status.update(player.state())
                status["live"] = (
                    live["name"] if live is not None else None)
                return json.dumps(status, indent=2)
            if verb == "tick":
                receipt = show.step()
                return (f"vtuber {name}: tick {receipt['tick']} "
                        f"({receipt['state']})"
                        + (" — holding" if receipt["held"] else ""))
            if verb == "cue":
                if len(rest) < 2:
                    return self._control_vtuber_help()
                player = getattr(show, "_vtuber_player", None)
                if player is None or not player._attached:
                    return (f"vtuber {name} performs nothing — "
                            f"perform an act first")
                duration = None
                value_parts = list(rest[1:])
                if len(value_parts) > 1:
                    try:
                        duration = float(value_parts[-1])
                        value_parts = value_parts[:-1]
                    except ValueError:
                        duration = None
                result = player.cue(rest[0],
                                    " ".join(value_parts), duration)
                return (f"vtuber {name}: cue {result['cue']} "
                        f"{result['value']}")
            if verb == "live":
                if not rest:
                    return self._control_vtuber_help()
                from ...media.mocap.mvn import get_receiver
                receiver = get_receiver(rest[0])
                attach_live(show, receiver, rest[0])
                return (f"vtuber {name} live from MVN receiver "
                        f"{rest[0]} — the recorded take holds")
            if verb == "live-detach":
                detach_live(show)
                return (f"vtuber {name} live detached — the recorded "
                        f"take resumes")
            if verb == "stop":
                if getattr(show, "_vtuber_player", None) is not None:
                    detach_performance(show)
                show.stop()
                return f"vtuber {name} stopped"
            if verb == "drop":
                drop_show(name)
                return f"vtuber {name} dropped"
        except (ShowError, VtuberError) as exc:
            record_failure(
                "runtime_mocap.RuntimeMocapMixin._control_vtuber", exc)
            return f"vtuber: {exc}"
        except Exception as exc:  # noqa: BLE001 - chat never dies
            record_failure(
                "runtime_mocap.RuntimeMocapMixin._control_vtuber", exc)
            return f"vtuber failed: {exc}"
        return self._control_vtuber_help()

    def _control_vtuber_help(self) -> str:
        return (
            "/vtuber perform <show> <act> <character> | character "
            "<path.json> [name] | act <path.vtact.json> [name] | "
            "cue <show> <kind> <value> [seconds] | live <show> "
            "<receiver> | live-detach <show> | status [show] | "
            "tick [show] | stop [show] | list | drop [show]")

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
        /live web [name] <dir> [ticks]      export the show as a hosted demo (static site)
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
                    "tick", "status", "events", "export", "web",
                    "load-take",
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
                            "stop", "tick", "events", "export", "web",
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
            if verb == "web":
                if not rest:
                    return "/live web [name] <dir> [ticks]"
                from ...media.show.web import export_web
                ticks = 30
                if len(rest) > 1 and rest[1].isdigit():
                    ticks = int(rest[1])
                attached = ensure_defaults(show)
                result = export_web(show, rest[0], ticks=ticks)
                extra = (f" (auto-attached {', '.join(attached)})"
                         if attached else "")
                return (f"show {name} as a hosted demo: "
                        f"{result['frames']} frames, "
                        f"{result['duration_s']}s, {result['index']}"
                        f"{extra}")
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
            "web [name] <dir> [ticks] | load-take [name] <path> | "
            "attach-rig [show] <rig> | "
            "attach-cloudcam [show] <cam> | drop [name]")

    # ── scene project: /project ────────────────────────────────────
    def _control_project(self, tail: str, chat_key: str,
                         message: Any = None) -> str:
        """Scene projects — one directory tying scenes, takes, rigs
        and assets together.

        /project create <name> [root]        scaffold a project
        /project list                        loaded projects
        /project status [name]               one project's state
        /project open <path>                 open a .dvproject/dir
        /project import-uproject <path> [root] import a UE .uproject
        /project validate [name]             every problem on disk
        /project add-scene <name> <path>     validate + add a scene
        /project add-take <name> <path>      validate + add a take
        /project add-rig <name> <path>       validate + add a rig
        /project add-asset <name> <path>     record a file as asset
        /project remove <kind> <entry> [name] drop an entry
        /project set-default <scene> [name]  name the default scene
        /project open-scene <scene> [name]   load a scene entry
        /project take-path <take> [name]     a take entry's path
        /project export <out> [name]         sealed bundle copy
        /project bind-show [name] [show] [take] show over the take
        /project drop [name]                 forget a loaded project
        """
        from ...media.project import (
            Project,
            drop_project,
            get_project,
            list_projects,
            load_project,
        )
        from ...media.project.errors import ProjectError
        from ...tools.mocap import _STORE

        parts = (tail or "").split()
        if not parts:
            return self._control_project_help()
        verb = parts[0]
        rest = parts[1:]
        name = "default"
        if verb in ("status", "validate", "add-scene", "add-take",
                    "add-rig", "add-asset", "set-default",
                    "open-scene", "take-path", "export", "bind-show",
                    "drop") and rest:
            name = rest[0]
            rest = rest[1:]
        try:
            if verb == "list":
                projects = list_projects()
                lines = ["projects:"]
                for entry in projects:
                    lines.append(
                        f"  {entry['name']}: "
                        f"{len(entry['scenes'])} scenes, "
                        f"{len(entry['takes'])} takes, "
                        f"{len(entry['rigs'])} rigs, "
                        f"default {entry['default_scene'] or 'none'}")
                return "\n".join(lines)
            if verb == "create":
                if not rest:
                    return "/project create <name> [root]"
                project = Project.create(
                    rest[0], rest[1] if len(rest) > 1 else ".")
                load_project(project.manifest_path)
                return (f"project {rest[0]} created at "
                        f"{project.root}")
            if verb == "open":
                if not rest:
                    return "/project open <path>"
                project = load_project(rest[0])
                return (f"project {project.name} opened "
                        f"({len(project.entries['scenes'])} scenes, "
                        f"{len(project.entries['takes'])} takes)")
            if verb == "import-uproject":
                if not rest:
                    return ("/project import-uproject "
                            "<path.uproject> [root]")
                project, receipt = Project.from_uproject(
                    rest[0], rest[1] if len(rest) > 1 else ".")
                load_project(project.manifest_path)
                unmapped = receipt["unmapped_plugins"]
                return (f"project {project.name} imported from UE "
                        f"{receipt['source_engine']} "
                        f"(modules: "
                        f"{', '.join(receipt['modules'])}; "
                        f"plugins: "
                        f"{', '.join(receipt['plugins']) or 'none'}"
                        + (f"; unmapped, disabled: "
                           f"{', '.join(unmapped)}" if unmapped
                           else "") + ")")
            if verb == "remove":
                if len(rest) < 2:
                    return ("/project remove <kind> <entry> [name]")
                name = rest[2] if len(rest) > 2 else "default"
                project = get_project(name)
                receipt = project.remove(rest[0], rest[1])
                project.save()
                return (f"project {name}: removed "
                        f"{receipt['kind']} {receipt['removed']}")
            if verb not in ("status", "validate", "add-scene",
                            "add-take", "add-rig", "add-asset",
                            "set-default", "open-scene", "take-path",
                            "export", "bind-show", "drop"):
                return self._control_project_help()
            project = get_project(name)
            if verb == "status":
                return json.dumps(project.status(), indent=2)
            if verb == "validate":
                problems = project.validate()
                if not problems:
                    return f"project {name} is whole"
                return (f"project {name} has {len(problems)} "
                        f"problem(s):\n"
                        + "\n".join(f"  - {p}" for p in problems))
            if verb in ("add-scene", "add-take", "add-rig",
                        "add-asset"):
                if not rest:
                    return (f"/project {verb} <name> <path>")
                path = rest[0]
                entry = None
                if verb == "add-scene":
                    receipt = project.add_scene(
                        Path(path).stem, path)
                elif verb == "add-take":
                    receipt = project.add_take(
                        Path(path).stem, path)
                elif verb == "add-rig":
                    receipt = project.add_rig(
                        Path(path).stem, path)
                else:
                    receipt = project.add_asset(
                        Path(path).stem, path)
                project.save()
                return (f"project {name}: added "
                        f"{receipt['kind']} {receipt['entry']} "
                        f"({receipt['stored']})")
            if verb == "set-default":
                if not rest:
                    return "/project set-default <scene> [name]"
                project.set_default_scene(rest[0])
                project.save()
                return (f"project {name}: default scene "
                        f"{rest[0]}")
            if verb == "open-scene":
                if not rest:
                    return "/project open-scene <scene> [name]"
                scene = project.open_scene(rest[0])
                return (f"project {name} scene {rest[0]}: "
                        f"{scene.title!r} — {len(scene.nodes)} "
                        f"nodes, {len(scene.tracks)} tracks")
            if verb == "take-path":
                if not rest:
                    return "/project take-path <take> [name]"
                return (f"project {name} take {rest[0]}: "
                        f"{project.take_path(rest[0])}")
            if verb == "export":
                if not rest:
                    return "/project export <out> [name]"
                receipt = project.export_bundle(rest[0])
                return (f"project {name} exported to "
                        f"{receipt['out']} "
                        f"({receipt['files']} files)")
            if verb == "bind-show":
                show = rest[0] if rest else name
                take = rest[1] if len(rest) > 1 else ""
                status = project.bind_show(show, _STORE, take=take)
                return (f"project {name}: show {show} bound to "
                        f"take {status['project_take']} — "
                        f"default scene "
                        f"{status['project_default_scene'] or 'none'}")
            if verb == "drop":
                drop_project(name)
                return f"project {name} dropped"
        except ProjectError as exc:
            record_failure(
                "runtime_mocap.RuntimeMocapMixin._control_project",
                exc)
            return f"project: {exc}"
        except Exception as exc:  # noqa: BLE001 - chat never dies
            record_failure(
                "runtime_mocap.RuntimeMocapMixin._control_project",
                exc)
            return f"project failed: {exc}"
        return self._control_project_help()

    def _control_project_help(self) -> str:
        return (
            "/project create <name> [root] | list | status [name] | "
            "open <path> | import-uproject <path.uproject> [root] | "
            "validate [name] | add-scene <name> <path> | add-take "
            "<name> <path> | add-rig <name> <path> | add-asset "
            "<name> <path> | remove <kind> <entry> [name] | "
            "set-default <scene> [name] | open-scene <scene> [name] "
            "| take-path <take> [name] | export <out> [name] | "
            "bind-show [name] [show] [take] | drop [name]")
