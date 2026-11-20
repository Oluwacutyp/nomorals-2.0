"""``nm voice`` — voice call/say/listen/transcribe and catalogue."""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any
from pathlib import Path



def _voice_data_dir(args: argparse.Namespace, settings: Any) -> str:
    return str(settings.audio.voice_data_dir or "voice_data")


def _voice_read_key_file(path: str) -> bytes:
    """Key file: raw 16/24/32-byte key, or a passphrase (PBKDF2-derived)."""
    raw = open(path, "rb").read().strip()
    if len(raw) in (16, 24, 32):
        return raw
    from ...core.cipher import derive_key

    return derive_key(raw.decode("utf-8", "replace"),
                      salt=b"nomorals-voice-audio", iterations=200_000)


def _voice_stt_from_tools(context: Any) -> Any:
    """STT callable reusing the exact transcription path as inbound voice
    notes (the ``transcribe`` tool). One-shot per utterance — see
    ``stt_supports_partial``."""

    def _stt(wav_path: str) -> str:
        outcome = context.tools.call("transcribe", path=wav_path,
                                     provider="auto")
        if not outcome.ok:
            return ""
        value = outcome.value or {}
        return str(value.get("text", "") or "").strip()

    _stt.supports_partial = False  # type: ignore[attr-defined]
    return _stt


def _voice_think_runtime(context: Any, device: str) -> Any:
    """A ``think`` callable wired to the real partner pipeline.

    One PartnerRuntime for the whole call, one stable ChatRef
    (``voice:<device>``) for every turn — the conversation identity
    persists in memory/DB across sessions, exactly like a chat DM.
    """
    from ...os.resources import default_manager
    from ...social.chat.base import ChatMessage, ChatRef
    from ...social.chat.gateway import ChatGateway
    from ...agents.partner_runtime import PartnerRuntime

    chat = ChatRef(platform="voice", chat_id=device, kind="dm",
                   title="Voice call", peer="owner")
    gateway = ChatGateway({}, db=context.db, owner_chats={chat.key})
    gateway.register_chat(chat)  # marks is_owner in the chat registry
    # Resource advisor injected (runtime is L5, may not import os/L6).
    runtime = PartnerRuntime(context, gateway=gateway,
                             resource_advisor=default_manager)
    counter = [0]

    def think(transcript: str) -> str:
        counter[0] += 1
        msg = ChatMessage(chat=chat, incoming=True, text=transcript,
                          sender="owner",
                          message_id=f"voice-{device}-{counter[0]}")
        outcome = runtime.handle_message(msg)
        return "\n\n".join(p for p in outcome.parts if p).strip()

    return think


def _cmd_voice(args: argparse.Namespace, context: Any) -> int:
    """Route `nm voice` subcommands."""
    action = args.voice_action
    if action == "call":
        return _cmd_voice_call(args, context)
    if action == "say":
        return _cmd_voice_say(args, context)
    if action == "listen":
        return _cmd_voice_listen(args, context)
    if action == "transcribe":
        return _cmd_voice_transcribe(args, context)
    if action == "stats":
        return _cmd_voice_stats(args, context)
    if action == "consent":
        return _cmd_voice_consent(args, context)
    if action == "purge":
        return _cmd_voice_purge(args, context)
    if action == "decrypt":
        return _cmd_voice_decrypt(args, context)
    if action == "fetch":
        return _cmd_voice_fetch(args, context)
    if action == "clone":
        return _cmd_voice_clone(args, context)
    if action == "list":
        return _cmd_voice_catalogue_list(args, context)
    if action == "use":
        return _cmd_voice_use(args, context)
    if action == "current":
        return _cmd_voice_current(args, context)
    if action == "rm":
        return _cmd_voice_rm(args, context)
    if action == "describe":
        return _cmd_voice_describe(args, context)
    if action == "transcript":
        return _cmd_voice_transcript(args, context)
    if action == "match":
        return _cmd_voice_match(args, context)
    if action == "probe":
        return _cmd_voice_probe(args, context)
    if action == "convert":
        return _cmd_voice_convert(args, context)
    if action == "live":
        return _cmd_voice_live(args, context)
    if action == "service":
        return _cmd_voice_service(args, context)
    if action == "model":
        return _cmd_voice_model(args, context)
    print(f"unknown voice action: {action}", file=sys.stderr)
    return 2


def _cmd_voice_service(args: argparse.Namespace, context: Any) -> int:
    """Route `nm voice service` subcommands (storyteller-ml §3B.11)."""
    from ...voice.services import (
        ContainerSpec,
        ModelArtifact,
        ModelServer,
        ServiceDeclaration,
        ServiceRegistry,
        TensorSpec,
        export_vits_probe_service,
    )
    from ...voice.services import (
        health as service_health,
    )

    action = args.voice_service_action
    registry = ServiceRegistry()

    def _emit(payload: Any) -> int:
        if getattr(args, "json", False):
            print(json.dumps(payload, indent=2, default=str))
        elif isinstance(payload, dict) and "error" in payload:
            print(f"error: {payload['error']}", file=sys.stderr)
            return 1
        elif isinstance(payload, dict) and payload.get("ok") is False:
            print(f"error: {payload.get('error', 'failed')}",
                  file=sys.stderr)
            return 1
        else:
            print(json.dumps(payload, indent=2, default=str))
        return 0

    if action == "list":
        rows = registry.list(task=getattr(args, "task", "") or "")
        if getattr(args, "json", False):
            return _emit({"services": [d.to_dict() for d in rows]})
        if not rows:
            print("no model services registered yet — "
                  "nm voice service export-vits-probe")
            return 0
        for d in rows:
            print(f"{d.slug}@{d.version} [{d.task}] "
                  f"{d.artifact.format} {d.artifact.path}")
        return 0

    if action == "register":
        import hashlib as _hashlib
        import os as _os
        path = _os.path.abspath(args.artifact)
        if not _os.path.isfile(path):
            print(f"no artifact file: {path}", file=sys.stderr)
            return 1
        digest = _hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
        try:
            inputs = [TensorSpec.from_dict(t)
                      for t in json.loads(args.inputs or "[]")]
            outputs = [TensorSpec.from_dict(t)
                       for t in json.loads(args.outputs or "[]")]
        except ValueError as exc:
            print(f"bad inputs/outputs JSON: {exc}", file=sys.stderr)
            return 1
        name = args.name or _os.path.splitext(_os.path.basename(path))[0]
        artifact = ModelArtifact(
            name=name, version=args.version, task=args.task,
            format="torchscript", path=path, sha256=digest.hexdigest(),
            inputs=inputs, outputs=outputs, entry=args.entry)
        decl = ServiceDeclaration(
            slug=name, version=args.version, task=args.task,
            artifact=artifact,
            container=ContainerSpec(owner=args.owner or "",
                                    base_image=args.base_image))
        try:
            registry.register(decl)
        except (ValueError, FileNotFoundError) as exc:
            print(f"register failed: {exc}", file=sys.stderr)
            return 1
        return _emit({"registered": decl.ref, "task": decl.task,
                      "artifact": decl.artifact.path,
                      "sha256": decl.artifact.sha256})

    if action == "serve":
        try:
            inputs = json.loads(args.inputs)
        except ValueError as exc:
            print(f"bad --inputs JSON: {exc}", file=sys.stderr)
            return 1
        server = ModelServer(registry)
        try:
            result = server.execute(args.ref, inputs)
        except Exception as exc:  # noqa: BLE001 — CLI reports, never traces
            print(f"serve failed: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
            return 1
        if getattr(args, "json", False):
            return _emit(result)
        print(f"{result['service']}@{result['version']} served: "
              + ", ".join(f"{n} {s}"
                          for n, s in result["output_shapes"].items()))
        return 0

    if action == "health":
        result = service_health(args.ref, registry=registry)
        return _emit(result)

    if action == "export-vits-probe":
        print("building the VITS probe service "
              "(real architecture, random weights)…", file=sys.stderr)
        try:
            decl = export_vits_probe_service(
                owner=getattr(args, "owner", "") or "")
        except Exception as exc:  # noqa: BLE001 — CLI reports, never traces
            print(f"export failed: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
            return 1
        return _emit({"registered": decl.ref, "task": decl.task,
                      "artifact": decl.artifact.path,
                      "sha256": decl.artifact.sha256,
                      "note": "probe service — random weights, "
                              "not a trained voice"})

    if action == "status":
        server = ModelServer(registry)
        return _emit(server.status())

    print(f"unknown voice service action: {action}", file=sys.stderr)
    return 2


def _voice_model_secret(args: argparse.Namespace) -> str:
    """The bundle secret: --secret, then --secret-file, then the
    NM_VOICE_BUNDLE_SECRET env var. Empty when none is given — the
    caller refuses, naming it (a bundle is never sealed or opened
    with an invented secret)."""
    import os
    if getattr(args, "secret", ""):
        return str(args.secret)
    secret_file = getattr(args, "secret_file", "") or ""
    if secret_file:
        try:
            return Path(secret_file).read_text(
                encoding="utf-8").strip()
        except OSError as exc:
            print(f"cannot read secret file {secret_file}: {exc}",
                  file=sys.stderr)
            return ""
    return os.environ.get("NM_VOICE_BUNDLE_SECRET", "")


def _voice_model_rfc3339(text: str) -> int:
    """RFC3339 timestamp -> epoch seconds (the bundle's expiry unit)."""
    from datetime import UTC, datetime
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"bad RFC3339 timestamp {text!r}: {exc}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return int(parsed.timestamp())


def _cmd_voice_model(args: argparse.Namespace, context: Any) -> int:
    """Route `nm voice model` subcommands (fakeyou-recast §3B.13)."""
    from ...voice.model_bundle import (
        ModelBundleError,
        VoiceModel,
        read_fast,
        read_metadata,
        register_model_bundle,
        unpack_bundle,
        write_bundle,
    )

    action = args.voice_model_action

    def _emit(payload: Any) -> int:
        if getattr(args, "json", False):
            print(json.dumps(payload, indent=2, default=str))
        elif isinstance(payload, dict) and "error" in payload:
            print(f"error: {payload['error']}", file=sys.stderr)
            return 1
        elif isinstance(payload, dict) and payload.get("ok") is False:
            print(f"error: {payload.get('error', 'failed')}",
                  file=sys.stderr)
            return 1
        else:
            print(json.dumps(payload, indent=2, default=str))
        return 0

    if action == "pack":
        if args.expires_at and args.expires_in_days is not None:
            print("cannot set both --expires-at and --expires-in-days",
                  file=sys.stderr)
            return 1
        secret = _voice_model_secret(args)
        if not secret:
            print("a bundle needs a secret — pass --secret, "
                  "--secret-file, or NM_VOICE_BUNDLE_SECRET",
                  file=sys.stderr)
            return 1
        try:
            expires_at = None
            if args.expires_at:
                expires_at = _voice_model_rfc3339(args.expires_at)
            elif args.expires_in_days is not None:
                expires_at = int(time.time()) + args.expires_in_days * 86400
            acoustic = Path(args.acoustic).read_bytes()
            vocoder = Path(args.vocoder).read_bytes()
        except (OSError, ValueError) as exc:
            print(f"pack failed: {exc}", file=sys.stderr)
            return 1
        model = VoiceModel(
            format_version=args.version,
            token=args.token, name=args.name, author=args.author,
            model_version=args.model_version or args.version,
            expires_at_s=expires_at,
            password=args.password or None,
            acoustic_model_bytes=acoustic, vocoder_model_bytes=vocoder)
        try:
            receipt = write_bundle(model, secret, args.out,
                                   replace=args.replace)
        except (ModelBundleError, FileExistsError, OSError) as exc:
            print(f"pack failed: {exc}", file=sys.stderr)
            return 1
        if getattr(args, "json", False):
            return _emit(receipt)
        print(f"packed {receipt['token']} — {receipt['name']} "
              f"({receipt['bytes']} bytes) -> {receipt['path']}")
        print(f"sha256 {receipt['sha256']}")
        return 0

    if action == "inspect":
        try:
            if not args.full:
                fast = read_fast(args.file)
                return _emit(fast.to_dict())
            secret = _voice_model_secret(args)
            if not secret:
                print("inspect --full needs the secret — pass --secret, "
                      "--secret-file, or NM_VOICE_BUNDLE_SECRET",
                      file=sys.stderr)
                return 1
            meta = read_metadata(args.file, secret)
            return _emit(meta.to_dict())
        except (ModelBundleError, FileNotFoundError, OSError) as exc:
            print(f"inspect failed: {exc}", file=sys.stderr)
            return 1

    if action == "unpack":
        secret = _voice_model_secret(args)
        if not secret:
            print("unpack needs the secret — pass --secret, "
                  "--secret-file, or NM_VOICE_BUNDLE_SECRET",
                  file=sys.stderr)
            return 1
        try:
            receipt = unpack_bundle(
                args.file, secret, args.out_dir,
                password=args.password or None,
                allow_expired=args.allow_expired)
        except (ModelBundleError, FileNotFoundError, OSError) as exc:
            print(f"unpack failed: {exc}", file=sys.stderr)
            return 1
        if getattr(args, "json", False):
            return _emit(receipt)
        print(f"unpacked {receipt['token']} — {receipt['name']}")
        print(f"acoustic {receipt['acoustic_bytes']} bytes -> "
              f"{receipt['acoustic_path']}")
        print(f"vocoder {receipt['vocoder_bytes']} bytes -> "
              f"{receipt['vocoder_path']}")
        return 0

    if action == "register":
        try:
            decl = register_model_bundle(args.file)
        except (ModelBundleError, ValueError, FileNotFoundError,
                OSError) as exc:
            print(f"register failed: {exc}", file=sys.stderr)
            return 1
        return _emit({"registered": decl.ref, "task": decl.task,
                      "artifact": decl.artifact.path,
                      "sha256": decl.artifact.sha256})

    print(f"unknown voice model action: {action}", file=sys.stderr)
    return 2


def _voice_tts_callable(args: argparse.Namespace, settings: Any,
                        data_dir: str) -> Any:
    """The TTS callable the live loops share: (text, profile) -> wav.

    Same engine as ``nm voice say`` (UniversalTTS, settings backend);
    ``--no-tts`` (call only) degrades to a text-only stub.
    """
    if getattr(args, "no_tts", False):
        def _silent(text: str, profile: str) -> dict:  # noqa: ARG001
            return {"path": ""}

        return _silent
    from ...voice.tts import UniversalTTS

    engine = UniversalTTS(backend=settings.audio.tts_engine or "auto")

    def _tts(text: str, profile: str) -> dict:
        out_path = str(Path(data_dir) / "tmp" /
                       f"tts-{int(time.time() * 1000)}.wav")
        return engine.speak(text, voice_name=profile or None,
                            out_path=out_path)

    return _tts


def _cmd_voice_call(args: argparse.Namespace, context: Any) -> int:
    from ...voice.session import VoiceSession

    settings = context.settings
    data_dir = _voice_data_dir(args, settings)
    audio_key = (_voice_read_key_file(args.audio_key_file)
                 if args.audio_key_file else None)

    tts = _voice_tts_callable(args, settings, data_dir)

    def _deliver(turn_id: str, text: str) -> None:
        print(f"\n━━━ {turn_id} ━━━\n{text}\n")

    def _ask() -> bool:
        try:
            ans = input("Allow microphone recording on this device? [y/N] ")
        except EOFError:
            return False
        return ans.strip().lower() in ("y", "yes")

    stt_backend = getattr(args, "stt", "tools")
    if stt_backend == "bridge":
        from ...voice.session import make_default_bridge_stt

        stt_fn = make_default_bridge_stt()
    elif stt_backend == "local":
        from ...voice.session import make_local_stt

        stt_fn = make_local_stt()
    else:
        stt_fn = _voice_stt_from_tools(context)

    session = VoiceSession(
        think=_voice_think_runtime(context, args.device),
        stt=stt_fn,
        tts=tts,
        deliver_text=_deliver,
        data_dir=data_dir,
        device_id=args.device,
        profile=args.profile,
        silence_ms=settings.audio.voice_silence_ms,
        keep_audio=args.keep_audio,
        audio_key=audio_key,
        speech_cap_secs=settings.audio.voice_speech_cap_secs,
    )
    print("Voice call starting — say \"goodbye\" to hang up.\n")
    report = session.run(max_turns=args.turns, ask_consent=_ask)
    if args.json:
        print(json.dumps({
            "session_id": report.session_id,
            "turns": report.turns,
            "barge_ins": report.barge_ins,
            "end_reason": report.end_reason,
            "error": report.error,
        }, indent=2))
    else:
        print(f"\nCall ended ({report.end_reason}): "
              f"{report.turns} turns, {report.barge_ins} barge-ins.")
        if report.error:
            print(f"error: {report.error}")
    return 0 if report.end_reason != "error" else 1


def _cmd_voice_live(args: argparse.Namespace, context: Any) -> int:
    """Route `nm voice live` subcommands."""
    action = args.voice_live_action
    if action == "start":
        return _cmd_voice_live_start(args, context)
    if action == "provision":
        return _cmd_voice_live_provision(args, context)
    if action == "status":
        return _cmd_voice_live_status(args, context)
    if action == "video":
        return _cmd_voice_live_video(args, context)
    print(f"unknown voice live action: {action}", file=sys.stderr)
    return 2


def _cmd_voice_live_video(args: argparse.Namespace, context: Any) -> int:
    """Route `nm voice live video` subcommands."""
    action = args.voice_live_video_action
    if action == "start":
        return _cmd_voice_live_video_start(args, context)
    if action == "status":
        return _cmd_voice_live_video_status(args, context)
    if action == "end":
        return _cmd_voice_live_video_end(args, context)
    if action == "snapshot":
        return _cmd_voice_live_video_snapshot(args, context)
    print(f"unknown voice live video action: {action}", file=sys.stderr)
    return 2


def _cmd_voice_live_video_start(args: argparse.Namespace,
                                context: Any) -> int:
    from ...voice.livekit_call import (
        LiveKitConfig,
        LiveKitUnavailable,
        LiveKitVideoSession,
        image_frame_source,
        scene_frame_source,
        title_frame_source,
    )

    settings = context.settings
    data_dir = _voice_data_dir(args, settings)
    design = None
    if args.source == "scene":
        if not (args.design or "").strip():
            print("video source 'scene' needs --design '<json>'",
                  file=sys.stderr)
            return 2
        try:
            design = json.loads(args.design)
        except json.JSONDecodeError as exc:
            print(f"--design is not valid JSON: {exc}", file=sys.stderr)
            return 2
        if not isinstance(design, dict) or not design:
            print("--design must be a JSON object (a motion-studio "
                  "design)", file=sys.stderr)
            return 2
    try:
        if args.source == "title":
            frame_source = title_frame_source(
                args.room, width=args.width, height=args.height)
            source_name = "title"
        elif args.source == "image":
            if not (args.image or "").strip():
                print("video source 'image' needs --image <path>",
                      file=sys.stderr)
                return 2
            frame_source = image_frame_source(
                args.image, width=args.width, height=args.height)
            source_name = f"image:{Path(args.image).name}"
        else:
            duration = args.duration if args.duration > 0 else 30.0
            frame_source = scene_frame_source(
                design, width=args.width, height=args.height,
                duration=duration, text=args.text, loop=args.loop)
            source_name = f"scene:{design.get('kind', 'scene')}"
        session = LiveKitVideoSession(
            frame_source=frame_source,
            config=LiveKitConfig.from_env(),
            room=args.room,
            data_dir=data_dir,
            width=args.width, height=args.height, fps=args.fps,
            source_name=source_name,
        )
    except LiveKitUnavailable as exc:
        print(f"cannot start the live video: {exc}", file=sys.stderr)
        if exc.missing:
            print(f"missing: {exc.missing}", file=sys.stderr)
        return 1
    print(f"Publishing video in LiveKit room {args.room!r} "
          f"({source_name}) — Ctrl-C to stop.\n")
    try:
        report = session.start(max_duration_s=args.max_duration)
    except LiveKitUnavailable as exc:
        print(f"cannot start the live video: {exc}", file=sys.stderr)
        if exc.missing:
            print(f"missing: {exc.missing}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.format())
    return 0 if report.end_reason != "error" else 1


def _cmd_voice_live_video_status(args: argparse.Namespace,
                                context: Any) -> int:
    from ...voice.livekit_call import live_video_status

    status = live_video_status(_voice_data_dir(args, context.settings))
    if args.json:
        print(json.dumps(status, indent=2))
        return 0
    if status["sdk_installed"]:
        print(f"livekit SDK: installed ({status['sdk_version']})")
    else:
        print("livekit SDK: NOT installed — "
              "nm voice live provision")
    if status["configured"]:
        print("LiveKit server: configured")
    else:
        print("LiveKit server: NOT configured — missing: "
              + "; ".join(status["missing"]))
    last = status.get("last_video")
    if last:
        print(f"last video: room {last.get('room')} — "
              f"{last.get('frames_published')} frames published, "
              f"ended {last.get('end_reason')}")
    return 0


def _cmd_voice_live_video_end(args: argparse.Namespace,
                             context: Any) -> int:  # noqa: ARG001
    from ...voice.livekit_call import end_video_session

    report = end_video_session()
    if report is None:
        print("no live video session is running", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.format())
    return 0 if report.end_reason != "error" else 1


def _cmd_voice_live_video_snapshot(args: argparse.Namespace,
                                  context: Any) -> int:  # noqa: ARG001
    from ...voice.livekit_call import get_video_session

    session = get_video_session()
    if session is None:
        print("no live video session is running", file=sys.stderr)
        return 1
    snap = session.snapshot(args.identity)
    if snap is None:
        who = f" from {args.identity!r}" if args.identity else ""
        print(f"no video frame received yet{who}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(snap, indent=2))
    else:
        print(f"snapshot from {snap['identity']}: {snap['path']} "
              f"({snap['frames_received']} frames received, "
              f"{snap['age_s']}s old)")
    return 0


def _cmd_voice_live_start(args: argparse.Namespace, context: Any) -> int:
    from ...voice.livekit_call import (
        LiveKitCallSession,
        LiveKitConfig,
        LiveKitUnavailable,
    )
    from ...voice.session import (
        make_default_bridge_stt,
        make_local_stt,
    )

    settings = context.settings
    data_dir = _voice_data_dir(args, settings)
    stt_backend = getattr(args, "stt", "tools")
    if stt_backend == "bridge":
        stt_fn = make_default_bridge_stt()
    elif stt_backend == "local":
        stt_fn = make_local_stt()
    else:
        stt_fn = _voice_stt_from_tools(context)
    session = LiveKitCallSession(
        think=_voice_think_runtime(
            context, args.device or f"livekit:{args.room}"),
        stt=stt_fn,
        tts=_voice_tts_callable(args, settings, data_dir),
        config=LiveKitConfig.from_env(),
        room=args.room,
        data_dir=data_dir,
        profile=args.profile,
        silence_ms=settings.audio.voice_silence_ms,
        speech_cap_secs=settings.audio.voice_speech_cap_secs,
    )
    print(f"Joining LiveKit room {args.room!r} — say \"goodbye\" to hang "
          f"up.\n")
    try:
        report = session.start(max_turns=args.turns)
    except LiveKitUnavailable as exc:
        print(f"cannot start the live call: {exc}", file=sys.stderr)
        if exc.missing:
            print(f"missing: {exc.missing}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.format())
    return 0 if report.end_reason != "error" else 1


def _cmd_voice_live_provision(args: argparse.Namespace,
                              context: Any) -> int:  # noqa: ARG001
    from ...voice.livekit_call import provision_livekit

    result = provision_livekit()
    if args.json:
        print(json.dumps(result, indent=2))
    elif result.get("ok"):
        if result.get("installed"):
            print(f"livekit SDK installed ({result.get('version')}) — "
                  f"set LIVEKIT_URL and LIVEKIT_API_KEY + "
                  f"LIVEKIT_API_SECRET (or LIVEKIT_TOKEN), then "
                  f"nm voice live start --room <name>")
        else:
            print(f"livekit SDK already installed "
                  f"({result.get('version')})")
    else:
        print(f"provision failed: {result.get('error')}", file=sys.stderr)
    return 0 if result.get("ok") else 1


def _cmd_voice_live_status(args: argparse.Namespace,
                           context: Any) -> int:
    from ...voice.livekit_call import livekit_status

    status = livekit_status(_voice_data_dir(args, context.settings))
    if args.json:
        print(json.dumps(status, indent=2))
        return 0
    if status["sdk_installed"]:
        print(f"livekit SDK: installed ({status['sdk_version']})")
    else:
        print("livekit SDK: NOT installed — "
              "nm voice live provision")
    if status["configured"]:
        print("LiveKit server: configured")
    else:
        print("LiveKit server: NOT configured — missing: "
              + "; ".join(status["missing"]))
    last = status.get("last_call")
    if last:
        print(f"last call: room {last.get('room')} — "
              f"{last.get('turns')} turns, {last.get('barge_ins')} "
              f"barge-ins, ended {last.get('end_reason')}")
    return 0


def _cmd_voice_say(args: argparse.Namespace, context: Any) -> int:
    from ...voice.catalogue import default_catalogue
    from ...voice.tts import UniversalTTS

    # --voice names a catalogue voice (resolves backend + profile);
    # --profile is the raw profile path for direct engine use.
    cat = default_catalogue()
    entry = cat.get(args.voice) if args.voice else None
    if entry is not None:
        out = cat.speak(args.text, out_path=args.out or "", mood=args.mood,
                        intensity=args.intensity, seed=args.seed,
                        effect=args.effect or None, perform=args.perform)
    else:
        backend = args.backend or context.settings.audio.tts_engine or "auto"
        engine = UniversalTTS(backend=backend)
        if args.perform:
            out = engine.perform(args.text,
                                 voice_name=(args.voice or args.profile
                                             or None),
                                 out_path=args.out or "", mood=args.mood,
                                 intensity=args.intensity, seed=args.seed,
                                 effect=args.effect or None)
        else:
            out = engine.speak(args.text,
                               voice_name=(args.voice or args.profile
                                           or None),
                               out_path=args.out or "")
    if args.json:
        print(json.dumps({"path": out.get("path"),
                          "backend": out.get("backend"),
                          "script": out.get("script"),
                          "cues": out.get("cues")}, indent=2))
    else:
        print(f"said it → {out.get('path')} ({out.get('backend')})")
        if out.get("script"):
            print(f"performance: {out['script']}")
    return 0


def _cmd_voice_fetch(args: argparse.Namespace, context: Any) -> int:
    from ...voice.fetch import MODEL_REGISTRY, fetch_model

    name = (args.backend or "cosyvoice").lower()
    info = MODEL_REGISTRY.get(name, {})
    if name == "piper":
        # one voice, not the whole rhasspy/piper-voices repo
        from ...voice.fetch import fetch_piper_voice

        voice_id = (args.voice or "en_US-lessac-medium").strip()
        print(f"fetching piper voice {voice_id} (MIT) from HuggingFace…")
        try:
            path = fetch_piper_voice(voice_id, dest=args.dest or "")
        except Exception as exc:
            print(f"fetch failed: {exc}", file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps({"backend": name, "voice": voice_id,
                              "path": path,
                              "repo": "rhasspy/piper-voices"}, indent=2))
        else:
            print(f"voice ready at {path}")
            print("speak with it: "
                  f"nm voice say \"hello there\" --backend piper")
        return 0
    print(f"fetching {name} ({info.get('license', '?')} license) "
          f"from HuggingFace…")
    try:
        path = fetch_model(name, dest=args.dest or "", repo=args.repo or "")
    except Exception as exc:
        print(f"fetch failed: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps({"backend": name, "path": path,
                          "repo": args.repo or info.get("hf_repo")},
                         indent=2))
    else:
        print(f"weights ready at {path}")
        print("speak with them: "
              f"nm voice say \"hello there\" --backend {name} --perform")
    return 0


def _cmd_voice_clone(args: argparse.Namespace, context: Any) -> int:  # noqa: ARG001
    from ...voice.catalogue import default_catalogue

    cat = default_catalogue()
    try:
        voice = cat.clone(args.name, args.audio,
                          transcript=args.transcript or "",
                          backend=args.backend or "auto",
                          description=args.describe or "")
    except Exception as exc:
        print(f"clone failed: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps({"name": voice.name, "backend": voice.backend,
                          "profile": voice.profile}, indent=2))
    else:
        print(f"cloned '{voice.name}' → catalogue "
              f"(backend: {voice.backend})")
        print(f"speak with it: nm voice say \"hello there\" "
              f"--voice {voice.name} --perform")
    return 0


def _cmd_voice_catalogue_list(args: argparse.Namespace,  # noqa: ARG001
                              context: Any) -> int:
    from ...voice.catalogue import default_catalogue

    cat = default_catalogue()
    voices = cat.list()
    if args.json:
        print(json.dumps(voices, indent=2))
    elif not voices:
        print("no voices yet — clone one: "
              "nm voice clone <name> <audio>")
    else:
        for v in voices:
            mark = "●" if v["active"] else "○"
            desc = f" — {v['description']}" if v["description"] else ""
            prof = f" (profile: {v['profile']})" if v["profile"] else ""
            print(f"{mark} {v['name']} [{v['backend']}]"
                  f"{prof}{desc}")
    return 0


def _cmd_voice_use(args: argparse.Namespace, context: Any) -> int:  # noqa: ARG001
    from ...voice.catalogue import default_catalogue

    cat = default_catalogue()
    try:
        voice = cat.set_active(args.name)
    except KeyError:
        print(f"unknown voice {args.name!r} — nm voice list",
              file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps({"active": voice.name}, indent=2))
    else:
        print(f"active voice → '{voice.name}' "
              f"(per-chat switches still win in chat)")
    return 0


def _cmd_voice_current(args: argparse.Namespace,  # noqa: ARG001
                       context: Any) -> int:
    from ...voice.catalogue import default_catalogue

    cat = default_catalogue()
    voice = cat.active_for_chat()
    if args.json:
        print(json.dumps({"active": voice.name if voice else None,
                          "backend": voice.backend if voice else None},
                         indent=2))
    elif voice is None:
        print("no active voice — nm voice list")
    else:
        print(f"active voice: '{voice.name}' [{voice.backend}]")
    return 0


def _cmd_voice_rm(args: argparse.Namespace, context: Any) -> int:  # noqa: ARG001
    from ...voice.catalogue import default_catalogue

    cat = default_catalogue()
    if not cat.remove(args.name):
        print(f"unknown voice {args.name!r} — nm voice list",
              file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps({"removed": args.name}, indent=2))
    else:
        print(f"removed '{args.name}' from the catalogue")
    return 0


def _cmd_voice_describe(args: argparse.Namespace,  # noqa: ARG001
                        context: Any) -> int:
    from ...voice.catalogue import default_catalogue

    cat = default_catalogue()
    voice = cat.get(args.name)
    if voice is None:
        print(f"unknown voice {args.name!r} — nm voice list",
              file=sys.stderr)
        return 1
    voice.description = args.text.strip()
    cat._save()
    if args.json:
        print(json.dumps({"name": voice.name,
                          "description": voice.description}, indent=2))
    else:
        print(f"description set for '{voice.name}'")
    return 0


def _cmd_voice_transcript(args: argparse.Namespace,  # noqa: ARG001
                          context: Any) -> int:
    from ...voice.catalogue import default_catalogue

    cat = default_catalogue()
    profile = cat.library.get(args.name)
    if profile is None:
        print(f"unknown voice {args.name!r} — nm voice list",
              file=sys.stderr)
        return 1
    profile.prompt_text = args.text.strip()
    cat.library._save_index()
    if args.json:
        print(json.dumps({"name": args.name,
                          "prompt_chars": len(args.text)}, indent=2))
    else:
        print(f"transcript set for '{args.name}' ({len(args.text)} chars)")
    return 0


def _cmd_voice_match(args: argparse.Namespace,  # noqa: ARG001
                     context: Any) -> int:
    from ...voice.catalogue import default_catalogue

    cat = default_catalogue()
    res = cat.library.match_voice(args.audio)
    if args.json:
        print(json.dumps(res, indent=2, default=str))
        return 0 if res.get("ok") else 1
    if not res.get("ok"):
        print(f"no match: {res.get('reason')}", file=sys.stderr)
        return 1
    print(f"🎙️ '{args.audio}' sounds most like '{res['name']}' "
          f"(distance {res.get('distance')}, {res.get('verdict', '')})")
    return 0


def _cmd_voice_probe(args: argparse.Namespace,  # noqa: ARG001
                     context: Any) -> int:
    from ...voice.tts import probe_reference_audio

    info = probe_reference_audio(args.audio)
    if args.json:
        print(json.dumps(info, indent=2, default=str))
        return 0 if info.get("ok") else 1
    if not info.get("ok"):
        print(f"probe failed: {'; '.join(info.get('warnings', []))}",
              file=sys.stderr)
        return 1
    print(f"🔬 {args.audio}: {info.get('seconds')}s @ "
          f"{info.get('sample_rate')} Hz, {info.get('channels')}ch")
    vp = info.get("voice_print") or {}
    feats = vp.get("features") or {}
    if feats:
        print(f"  mean F0: {feats.get('mean_f0_hz')} Hz "
              f"(±{feats.get('f0_std_hz')})")
        print(f"  centroid: {feats.get('centroid_hz')} Hz · "
              f"loudness {feats.get('rms_db')} dBFS")
    print(f"  verdict: {vp.get('verdict', info.get('verdict', ''))}")
    for w in info.get("warnings", [])[:4]:
        print(f"  ⚠️ {w}")
    return 0


def _cmd_voice_convert(args: argparse.Namespace, context: Any) -> int:
    """Route `nm voice convert` — file, live loop, server, status, probe."""
    from ...voice import realtime as _rt

    if args.export_probe:
        try:
            result = _rt.export_probe_model(args.out or "")
        except Exception as exc:  # noqa: BLE001 — honest failure, named
            print(f"probe export failed: {exc}", file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"probe model → {result['path']}")
            print(f"receipt → {result['receipt']}")
            print("convert with it: "
                  "nm voice convert <file> --model "
                  f"{result['path']}")
        return 0
    if args.status:
        status = _rt.conversion_status()
        if args.json:
            print(json.dumps(status, indent=2))
            return 0
        print(_rt.status_text(status))
        return 0
    params = _rt.ConversionParams(
        direction=args.direction,
        buffer_size_minimum=args.buffer_size,
        pitch_shift_semitones=args.pitch_shift,
        skip_vocode=args.skip_vocode,
        discard_vocoded_audio=args.discard_vocoded,
    )
    if args.serve:
        if args.transport == "inproc":
            print("serve needs a network transport: "
                  "--transport tcp|zeromq", file=sys.stderr)
            return 2
        try:
            # skip-vocode needs no model: the converter returns the
            # resampled input without ever calling an executor.
            executor = (None if args.skip_vocode
                        else _rt.resolve_executor(
                            model_path=args.model,
                            service_ref=args.service))
        except _rt.ConversionUnavailable as exc:
            print(f"cannot serve: {exc}", file=sys.stderr)
            return 1
        server = _rt.ConversionServer(_rt.Converter(executor, params))
        transport = args.transport
        print(f"conversion sidecar serving ({transport}) — "
              "Ctrl-C to stop.\n")
        try:
            if transport == "zeromq":
                _rt.serve_zeromq(
                    server, f"tcp://*:{args.port}")
            else:
                _rt.serve_tcp(server, args.host, args.port)
        except _rt.ConversionUnavailable as exc:
            print(f"cannot serve: {exc}", file=sys.stderr)
            return 1
        except KeyboardInterrupt:
            print("\nsidecar stopped.")
        return 0
    if args.live:
        return _cmd_voice_convert_live(args, context, params)
    if not args.file:
        print("usage: nm voice convert <file> [--out PATH] "
              "[--model PATH | --service REF] | --live | --serve | "
              "--status | --export-probe", file=sys.stderr)
        return 2
    try:
        executor = (None if args.skip_vocode
                    else _rt.resolve_executor(
                        model_path=args.model,
                        service_ref=args.service))
    except _rt.ConversionUnavailable as exc:
        print(f"cannot convert: {exc}", file=sys.stderr)
        return 1
    out = args.out or ""
    if not out:
        stem = Path(args.file).stem
        out = str(Path(args.file).with_name(f"{stem}-converted.wav"))
    try:
        receipt = _rt.convert_file(args.file, out, executor=executor,
                                   params=params)
    except _rt.ConversionUnavailable as exc:
        print(f"cannot convert: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 — honest failure, named
        print(f"conversion failed: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(receipt, indent=2))
    else:
        print(f"converted → {receipt['output']} "
              f"({receipt['input_samples']} samples "
              f"@{receipt['input_rate']} Hz -> "
              f"{receipt['output_samples']} samples "
              f"@{receipt['output_rate']} Hz, "
              f"{receipt['executor']}, {args.direction})")
    return 0


def _cmd_voice_convert_live(args: argparse.Namespace, context: Any,
                            params: Any) -> int:
    """The live mic -> model -> speaker loop (consent-gated)."""
    from ...voice import realtime as _rt
    from ...voice.session import (
        ConsentStore,
        default_mic,
        default_speaker,
    )

    settings = context.settings
    data_dir = _voice_data_dir(args, settings)
    consent = ConsentStore(data_dir)
    if not consent.consented(args.device):
        try:
            ans = input("Allow microphone recording on this device? [y/N] ")
        except EOFError:
            ans = ""
        if ans.strip().lower() not in ("y", "yes"):
            print("no consent — not recording.")
            return 1
        consent.grant(args.device)
    mic = default_mic()
    if mic is None:
        print("no microphone available on this device.", file=sys.stderr)
        return 1
    speaker = default_speaker()
    if speaker is None:
        print("no speaker available on this device.", file=sys.stderr)
        return 1
    try:
        executor = (None if params.skip_vocode
                    else _rt.resolve_executor(
                        model_path=args.model,
                        service_ref=args.service))
    except _rt.ConversionUnavailable as exc:
        print(f"cannot convert: {exc}", file=sys.stderr)
        return 1
    if args.transport == "inproc":
        server = _rt.ConversionServer(_rt.Converter(executor, params))
        client = _rt.InProcessClient(server)
    else:
        try:
            client = _rt.make_client(
                args.transport, host=args.host, port=args.port)
        except (_rt.ConversionUnavailable, ValueError,
                OSError) as exc:
            print(f"cannot reach the sidecar: {exc}", file=sys.stderr)
            return 1
    loop = _rt.RealtimeConverter(
        mic=mic, speaker=speaker, client=client, params=params,
        data_dir=data_dir)
    print("Realtime conversion live — speak, and hear it back converted "
          "(Ctrl-C to stop).\n")
    try:
        report = loop.run(seconds=args.seconds)
    except _rt.ConversionUnavailable as exc:
        print(f"cannot convert: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nconversion stopped.")
        return 0
    finally:
        client.close()
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"live conversion ended ({report['end_reason']}): "
              f"{report['batches_sent']} batches sent, "
              f"{report['batches_received']} converted back, "
              f"{report['activations']} activations, "
              f"{report['transport_errors']} transport errors.")
    return 0


def _cmd_voice_listen(args: argparse.Namespace, context: Any) -> int:
    import time
    from pathlib import Path

    from ...voice.session import (ConsentStore, EnergyVAD, MicCapture,
                                default_mic, write_wav_bytes)

    data_dir = _voice_data_dir(args, context.settings)
    consent = ConsentStore(data_dir)
    if not consent.consented(args.device):
        try:
            ans = input("Allow microphone recording on this device? [y/N] ")
        except EOFError:
            ans = ""
        if ans.strip().lower() not in ("y", "yes"):
            print("no consent — not recording.")
            return 1
        consent.grant(args.device)
    mic = default_mic()
    if mic is None:
        print("no microphone available.", file=sys.stderr)
        return 1
    capture = MicCapture(mic)
    capture.start()
    vad = EnergyVAD(silence_ms=800)
    chunks: list[bytes] = []
    deadline = time.time() + max(1.0, args.secs)
    try:
        while time.time() < deadline:
            chunk = capture.poll(timeout=0.1)
            if chunk is None:
                continue
            chunks.append(chunk)
            if vad.observe(chunk) == "silence" and len(chunks) > 40:
                break
    finally:
        capture.stop()
        mic.close()
    out = args.out or str(Path(data_dir) / "tmp" /
                          f"listen-{int(time.time())}.wav")
    write_wav_bytes(out, b"".join(chunks))
    if args.json:
        print(json.dumps({"path": out, "chunks": len(chunks)}, indent=2))
    else:
        print(f"recorded → {out}")
    return 0


def _cmd_voice_transcribe(args: argparse.Namespace, context: Any) -> int:
    stt = _voice_stt_from_tools(context)
    text = stt(args.file)
    if args.json:
        print(json.dumps({"text": text}, indent=2))
    else:
        print(text)
    return 0


def _cmd_voice_stats(args: argparse.Namespace, context: Any) -> int:
    from ...voice.session import StatsStore

    summary = StatsStore(_voice_data_dir(args, context.settings)).summary()
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(f"sessions: {summary['sessions']}  turns: {summary['turns']}  "
              f"barge-ins: {summary['barge_ins']}")
        print(f"ear-to-ear p50: {summary['ear_to_ear_ms_p50']} ms  "
              f"p95: {summary['ear_to_ear_ms_p95']} ms")
    return 0


def _cmd_voice_consent(args: argparse.Namespace, context: Any) -> int:
    from ...voice.session import ConsentStore

    store = ConsentStore(_voice_data_dir(args, context.settings))
    if args.revoke:
        store.revoke(args.device)
        result = {"device": args.device, "consented": False}
    elif args.grant:
        store.grant(args.device)
        result = {"device": args.device, "consented": True}
    else:
        result = {"device": args.device,
                  "consented": store.consented(args.device)}
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        state = "granted" if result["consented"] else "not granted"
        print(f"recording consent for {args.device!r}: {state}")
    return 0


def _cmd_voice_purge(args: argparse.Namespace, context: Any) -> int:
    from ...voice.session import VoiceSession

    session = VoiceSession(
        think=lambda t: "", stt=lambda p: "", tts=lambda t, p: {},
        data_dir=_voice_data_dir(args, context.settings))
    removed = session.purge_audio()
    if args.json:
        print(json.dumps({"removed": removed}, indent=2))
    else:
        print(f"deleted {removed} retained audio file(s).")
    return 0


def _cmd_voice_decrypt(args: argparse.Namespace, context: Any) -> int:
    from pathlib import Path

    from ...voice.session import decrypt_kept_audio, write_wav_bytes

    key = _voice_read_key_file(args.key_file)
    try:
        pcm = decrypt_kept_audio(args.file, key)
    except Exception as exc:
        print(f"decrypt failed: {exc}", file=sys.stderr)
        return 1
    out = args.out or str(Path(args.file).with_suffix("").with_suffix(".wav"))
    write_wav_bytes(out, pcm)
    print(f"decrypted → {out}")
    return 0
