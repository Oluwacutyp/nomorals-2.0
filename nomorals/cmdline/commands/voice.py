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
    from ...social.chat.base import ChatMessage, ChatRef
    from ...social.chat.gateway import ChatGateway
    from ...agents.partner_runtime import PartnerRuntime

    chat = ChatRef(platform="voice", chat_id=device, kind="dm",
                   title="Voice call", peer="owner")
    gateway = ChatGateway({}, db=context.db, owner_chats={chat.key})
    gateway.register_chat(chat)  # marks is_owner in the chat registry
    runtime = PartnerRuntime(context, gateway=gateway)
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
    print(f"unknown voice action: {action}", file=sys.stderr)
    return 2


def _cmd_voice_call(args: argparse.Namespace, context: Any) -> int:
    import time
    from pathlib import Path

    from ...voice.session import VoiceSession

    settings = context.settings
    data_dir = _voice_data_dir(args, settings)
    audio_key = (_voice_read_key_file(args.audio_key_file)
                 if args.audio_key_file else None)

    tts = None
    if not args.no_tts:
        from ...voice.tts import UniversalTTS

        engine = UniversalTTS(backend=settings.audio.tts_engine or "auto")

        def _tts(text: str, profile: str) -> dict:
            out_path = str(Path(data_dir) / "tmp" /
                           f"tts-{int(time.time() * 1000)}.wav")
            return engine.speak(text, voice_name=profile or None,
                                out_path=out_path)

        tts = _tts
    else:
        def _tts(text: str, profile: str) -> dict:  # noqa: ARG001
            return {"path": ""}

    def _deliver(turn_id: str, text: str) -> None:
        print(f"\n━━━ {turn_id} ━━━\n{text}\n")

    def _ask() -> bool:
        try:
            ans = input("Allow microphone recording on this device? [y/N] ")
        except EOFError:
            return False
        return ans.strip().lower() in ("y", "yes")

    session = VoiceSession(
        think=_voice_think_runtime(context, args.device),
        stt=_voice_stt_from_tools(context),
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
