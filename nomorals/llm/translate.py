"""The translation organ: one ladder, every surface.

Owner ask (2026-11-03): Devon surfaces global news in the source's
language and the owner reads English — *"a translator for all this
strange languages from news will be really appreciated."*  This organ
is the answer, built as an IMPLEMENTATION LADDER (owner law, same
day): every rung is a full implementation, never a downscale, and the
profile runtime + plain availability pick the rung per device:

1. **local** — an NLLB-class seq2seq model on this machine, riding
   the repo's lazy-import local-inference seam (the ``integrations``
   STT pattern).  Weights are data the owner drops in; when deps or
   weights are absent the rung reports exactly that and the ladder
   moves on — it never fakes a translation.
2. **brain** — translation through the existing LLM interface
   (``brain_for``), the judgment rung: idioms, short tricky text,
   unknown source languages.
3. **providers** — CODE rungs: independent free translation endpoints
   behind the repo's source-health pattern (``agency.sourcehealth`` —
   seed list, every outcome noted, failures shed by the same
   stand-down/disable circuit breaker the news feeds use).  The seed
   list is a SEED (open-sources law): extra instances configure in
   via ``settings.translate.providers``.
4. **identity** — the marked last resort: the original text returned
   WITH a plain note that translation was unavailable.  Untranslated
   text is never passed off as a translation.

Detection is laddered too (see :mod:`nomorals.llm.languages`).

Consumers: owner-addressed deliveries translate at composition time
(the news organ's digest + push paths, the pulses' story fetch, the
briefing composer's news section — so the morning VOICE briefing is
written from already-translated text and TTS speaks the translation),
and the ``/translate`` command for on-demand work.  Group rooms keep
their own languages: nothing here translates a room unbidden — only
owner-addressed deliveries and explicit calls.

Doc 19 §4 (pass T1): every translation leaves a durable receipt
(:mod:`nomorals.llm.receipts`), the bidirectional quality gate
(:mod:`nomorals.llm.translation_quality`) is wired in for
fidelity-class requests, and :mod:`nomorals.llm.ticket` is the typed
facade callers use when the exactness class matters.

Pass T2: the ticket path adds section-wise document translation
(:mod:`nomorals.llm.sections`), an applied glossary store
(:mod:`nomorals.llm.glossary`), and this organ's cache gains its
durable layer (:mod:`nomorals.llm.cache`) under the hot dicts —
consulted before the ladder, written on success, invalidated on
expiry or local-model change.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote

from ..core.http import HttpClient
from ..core.logging_setup import get_logger
from .cache import TranslationCacheStore, cache_key_for
from .languages import (
    NLLB_CODES,
    detect_language,
    language_name,
    normalize_code,
)
from .receipts import TranslationReceiptStore, new_receipt_id
from .translation_quality import translate_with_quality

_log = get_logger(__name__)

__all__ = [
    "TranslationOutcome",
    "ItemTranslation",
    "Translator",
    "translator_for",
    "owner_language",
    "localize_news_rows",
    "DEFAULT_LOCAL_MODEL_REPO",
    "EXACTNESS_CLASSES",
    "FIDELITY_CLASSES",
]

#: Exactness classes a caller may request (doc 19 §4).  ``gist`` is
#: the everyday class; the fidelity classes are the ones whose
#: results may be labelled exact — and only when the quality gate
#: passes them (see ``Translator._gate_verdict``).
EXACTNESS_CLASSES = ("gist", "fidelity", "legal", "code")
FIDELITY_CLASSES = ("fidelity", "legal", "code")

#: The local rung's default weights (NLLB-200 distilled 600M — the
#: small-specialist end of the NLLB family, 200 languages).
DEFAULT_LOCAL_MODEL_REPO = "facebook/nllb-200-distilled-600M"

#: Provider seeds (open-sources law: seeds, never a closed list —
#: ``settings.translate.providers`` adds Lingva-compatible instances).
_MYMEMORY_URL = "https://api.mymemory.translated.net/get"
_LINGVA_SEED = "https://lingva.ml"
_LIBRE_SEED = "https://libretranslate.com"

#: Health circuit-breaker thresholds — the source-health pattern's own
#: numbers, mirrored for the in-memory (db-less) tracker.
_STAND_DOWN_AFTER = 3
_DISABLE_AFTER = 10
_STAND_DOWN_COOLDOWN_S = 600.0
_DISABLE_COOLDOWN_S = 3600.0

#: Profiles where an unconfigured local seq2seq model is the wrong
#: rung (the ladder exists so the device class decides, not habit).
_LOW_POWER_PROFILES = {"termux", "mobile", "embedded"}


@dataclass
class TranslationOutcome:
    """One ladder result. ``translated=False`` is always honest:
    ``rung`` says why (same-language, or the marked identity)."""

    text: str
    source_lang: str = ""
    target_lang: str = "en"
    rung: str = "identity"
    translated: bool = False
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text, "source_lang": self.source_lang,
            "target_lang": self.target_lang, "rung": self.rung,
            "translated": self.translated, "note": self.note,
        }


@dataclass
class ItemTranslation:
    """A headline + gist pair after the ladder."""

    title: str
    summary: str
    original_title: str = ""
    original_summary: str = ""
    source_lang: str = ""
    translated: bool = False
    rung: str = "identity"


# ── text helpers ─────────────────────────────────────────────────────

_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")


def _split_chunks(text: str, limit: int) -> list[str]:
    """Split prose into <= ``limit``-char chunks on sentence bounds.

    Falls back to hard cuts for a single oversized sentence.  The
    chunks re-join with single spaces — translation units, not a
    lossless codec, and every rung treats them that way.
    """
    text = (text or "").strip()
    if len(text) <= limit:
        return [text] if text else []
    parts: list[str] = []
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            continue
        if len(para) <= limit:
            parts.append(para)
            continue
        for sent in _SENTENCE_END.split(para):
            sent = sent.strip()
            if not sent:
                continue
            while len(sent) > limit:
                cut = sent.rfind(" ", 0, limit)
                if cut < limit // 3:
                    cut = limit
                parts.append(sent[:cut].strip())
                sent = sent[cut:].strip()
            if sent:
                parts.append(sent)
    # pack small neighbours back together up to the limit
    packed: list[str] = []
    cur = ""
    for part in parts:
        if cur and len(cur) + 1 + len(part) > limit:
            packed.append(cur)
            cur = part
        else:
            cur = f"{cur} {part}" if cur else part
    if cur:
        packed.append(cur)
    return packed


def _word_count(text: str) -> int:
    return len(re.findall(r"[^\W\d_]+", text or "", re.UNICODE))


# ── provider health (the source-health pattern) ──────────────────────

class _ProviderHealth:
    """Seed-provider health: SourceHealth over the db when there is
    one (durable, shared with the news feeds' circuit breaker), the
    same stand-down semantics in memory when there is not."""

    def __init__(self, db: Any = None) -> None:
        self._sh: Any = None
        if db is not None:
            try:
                from ..agency.sourcehealth import SourceHealth

                self._sh = SourceHealth(db)
            except Exception:  # noqa: BLE001 — health is a guard, not a gate
                _log.debug("translate: SourceHealth unavailable",
                           exc_info=True)
                self._sh = None
        self._mem: dict[str, dict[str, Any]] = {}

    def _spec(self, key: str, name: str, url: str) -> Any:
        from ..agency.sourcehealth import SourceSpec

        return SourceSpec(key=f"translate:{key}", category="translation",
                          name=name, url=url, kind="json")

    def blocked(self, key: str, name: str, url: str) -> bool:
        if self._sh is not None:
            try:
                rec = self._sh.record(f"translate:{key}")
                return bool(rec is not None and rec.status in
                            ("stood-down", "disabled"))
            except Exception:  # noqa: BLE001
                return False
        state = self._mem.get(key)
        if not state:
            return False
        consec = int(state.get("consec", 0))
        age = time.time() - float(state.get("last_fail", 0.0))
        if consec >= _DISABLE_AFTER:
            return age < _DISABLE_COOLDOWN_S
        if consec >= _STAND_DOWN_AFTER:
            return age < _STAND_DOWN_COOLDOWN_S
        return False

    def note(self, key: str, name: str, url: str, *, ok: bool,
             error: str = "") -> None:
        if self._sh is not None:
            try:
                self._sh.note_result(self._spec(key, name, url), ok=ok,
                                     error=error[:200])
                return
            except Exception:  # noqa: BLE001
                _log.debug("translate: health note failed", exc_info=True)
        state = self._mem.setdefault(key, {"consec": 0, "last_fail": 0.0})
        if ok:
            state["consec"] = 0
        else:
            state["consec"] = int(state.get("consec", 0)) + 1
            state["last_fail"] = time.time()


# ── rung: local NMT ──────────────────────────────────────────────────

class _TransformersBackend:
    """The repo's local-inference seam: lazy torch/transformers load,
    exactly the way the STT organ loads faster-whisper.  Loads on
    first use; any failure raises to the rung, which records the
    honest unavailable reason."""

    def __init__(self, model_ref: str) -> None:
        self.model_ref = model_ref
        self._tok: Any = None
        self._model: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch  # type: ignore[import-not-found]  # noqa: F401
        from transformers import (  # type: ignore[import-not-found]
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
        )

        self._tok = AutoTokenizer.from_pretrained(self.model_ref)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_ref)
        try:
            if torch.cuda.is_available():
                self._model = self._model.to("cuda")
        except Exception:  # noqa: BLE001 — CPU is a fine rung
            pass
        self._model.eval()

    def translate(self, text: str, src_nllb: str, dst_nllb: str) -> str:
        import torch

        self._load()
        self._tok.src_lang = src_nllb
        encoded = self._tok(text, return_tensors="pt",
                            truncation=True, max_length=1024)
        try:
            if next(self._model.parameters()).is_cuda:
                encoded = {k: v.to("cuda") for k, v in encoded.items()}
        except Exception:  # noqa: BLE001
            pass
        forced = self._tok.convert_tokens_to_ids(dst_nllb)
        with torch.no_grad():
            out = self._model.generate(
                **encoded, forced_bos_token_id=forced,
                max_length=1024, num_beams=4, early_stopping=True)
        return str(self._tok.batch_decode(
            out, skip_special_tokens=True)[0]).strip()


class LocalNmtRung:
    """Rung 1 — a local NLLB-class model.  Real when weights + deps
    exist; an honest unavailable reason (never a fake) when not."""

    name = "local"

    def __init__(self, *, model_ref: str = "", home: str = "",
                 profile_kind: str = "",
                 backend: Any = None) -> None:
        self._model_ref = (model_ref or "").strip()
        self._home = home or os.path.expanduser("~/.nomorals")
        self._profile_kind = profile_kind
        self._backend = backend
        self._reason = ""
        self._checked = False
        self._resolved = ""

    def _explicit(self) -> bool:
        return bool(self._model_ref)

    def _resolve_weights(self) -> str:
        """A loadable weights reference, or "" when none are present."""
        if self._resolved:
            return self._resolved
        ref = self._model_ref
        if ref:
            # an explicit path, or an explicit HF id (owner intent:
            # transformers may fetch it) — both are loadable refs.
            self._resolved = ref
            return ref
        local_dir = (Path(self._home) / "models"
                     / "nllb-200-distilled-600M")
        if (local_dir / "config.json").is_file():
            self._resolved = str(local_dir)
            return self._resolved
        hf_cache = (Path(os.path.expanduser("~/.cache/huggingface"))
                    / "hub" / ("models--" + DEFAULT_LOCAL_MODEL_REPO
                               .replace("/", "--")) / "snapshots")
        if hf_cache.is_dir() and any(hf_cache.iterdir()):
            self._resolved = DEFAULT_LOCAL_MODEL_REPO
            return self._resolved
        return ""

    def available(self) -> tuple[bool, str]:
        if self._backend is not None:
            return True, "injected backend"
        if self._checked:
            return (not self._reason), self._reason
        self._checked = True
        profile = (self._profile_kind or "").strip().lower()
        if profile in _LOW_POWER_PROFILES and not self._explicit():
            self._reason = (f"profile {profile}: local seq2seq is not "
                            "the rung for this device class")
            return False, self._reason
        try:
            import torch  # type: ignore[import-not-found]  # noqa: F401
            import transformers  # type: ignore[import-not-found]  # noqa: F401
        except Exception:  # noqa: BLE001
            self._reason = ("transformers/torch not installed on this "
                            "machine")
            return False, self._reason
        if not self._resolve_weights():
            self._reason = (f"no local NMT weights found — drop "
                            f"{DEFAULT_LOCAL_MODEL_REPO} under "
                            f"{self._home}/models or set "
                            "NM_TRANSLATE_MODEL")
            return False, self._reason
        return True, ""

    def translate(self, text: str, src: str, dst: str) -> str | None:
        src_nllb = NLLB_CODES.get(src or "", "")
        dst_nllb = NLLB_CODES.get(dst or "", "")
        if not src_nllb or not dst_nllb:
            return None  # pair outside NLLB's table: decline honestly
        ok, _ = self.available()
        if not ok:
            return None
        try:
            if self._backend is None:
                self._backend = _TransformersBackend(
                    self._resolve_weights())
            parts = []
            for chunk in _split_chunks(text, 700):
                out = self._backend.translate(chunk, src_nllb, dst_nllb)
                if not out:
                    return None
                parts.append(out)
            return " ".join(parts).strip() or None
        except Exception as exc:  # noqa: BLE001 — a dead rung sheds
            self._reason = f"local model failed: {type(exc).__name__}: {exc}"[:200]
            self._backend = None
            _log.warning("translate: local rung failed: %s", exc)
            return None


# ── rung: the brain ──────────────────────────────────────────────────

class BrainRung:
    """Rung 2 — translation through the existing LLM interface."""

    name = "brain"

    def __init__(self, context: Any = None, brain: Any = None) -> None:
        self._context = context
        self._brain = brain

    def available(self) -> tuple[bool, str]:
        if self._brain is not None:
            return True, ""
        if getattr(self._context, "router", None) is not None:
            return True, ""
        return False, "no brain is answering on this context"

    def _get_brain(self) -> Any:
        if self._brain is None:
            from .brain import brain_for

            self._brain = brain_for(self._context)
        return self._brain

    def translate(self, text: str, src: str, dst: str) -> str | None:
        ok, _ = self.available()
        if not ok:
            return None
        from ..llm.base import Message

        dst_name = language_name(dst)
        src_name = language_name(src) if src else ""
        parts: list[str] = []
        try:
            brain = self._get_brain()
            for chunk in _split_chunks(text, 3000):
                if src_name:
                    prompt = (
                        f"Translate the following text from {src_name} "
                        f"to {dst_name}. Reply with ONLY the "
                        f"translation — no quotes, no commentary, keep "
                        f"names, numbers and links exactly as they "
                        f"are.\n\n{chunk}")
                else:
                    prompt = (
                        f"Detect the language of the following text "
                        f"and translate it to {dst_name}. Reply with "
                        f"ONLY the translation — no quotes, no "
                        f"commentary, keep names, numbers and links "
                        f"exactly as they are.\n\n{chunk}")
                response = brain.chat(
                    [Message(role="user", content=prompt)],
                    task_kind="translate")
                out = (getattr(response, "text", "") or "").strip()
                if not out:
                    return None
                if (src and src != dst and out == chunk
                        and _word_count(chunk) >= 3):
                    return None  # an echo is not a translation
                parts.append(out)
        except Exception as exc:  # noqa: BLE001
            _log.warning("translate: brain rung failed: %s", exc)
            return None
        return " ".join(parts).strip() or None


# ── rung: provider endpoints (code rungs) ────────────────────────────

class _Decline(Exception):
    """A rung declining a pair (not a failure): no health note."""


class ProviderRung:
    """One free translation endpoint behind the health pattern."""

    def __init__(self, key: str, name: str, kind: str, base: str,
                 client: Any, health: _ProviderHealth) -> None:
        self.key = key
        self.name = name
        self.kind = kind
        self.base = base.rstrip("/")
        self._client = client
        self._health = health
        self.rung_label = f"provider:{key}"

    def available(self) -> tuple[bool, str]:
        if self._health.blocked(self.key, self.name, self.base):
            return False, "stood down by the health circuit breaker"
        return True, ""

    def _one(self, text: str, src: str, dst: str) -> str | None:
        if self.kind == "mymemory":
            if not src:
                # MyMemory needs a named source language: a DECLINE,
                # not a provider failure — no health note is taken.
                raise _Decline("mymemory needs a source language")
            url = (f"{_MYMEMORY_URL}?q={quote(text)}"
                   f"&langpair={src}|{dst}")
            resp = self._client.get(url)
            if not getattr(resp, "ok", False):
                return None
            data = resp.json() or {}
            out = str((data.get("responseData") or {})
                      .get("translatedText") or "").strip()
            if (not out or "MYMEMORY WARNING" in out.upper()
                    or "QUERY LENGTH LIMIT" in out.upper()
                    or "INVALID" in out.upper()[:40]):
                return None
            return out
        if self.kind == "lingva":
            url = (f"{self.base}/api/v1/{src or 'auto'}/{dst}/"
                   f"{quote(text, safe='')}")
            resp = self._client.get(url)
            if not getattr(resp, "ok", False):
                return None
            data = resp.json() or {}
            out = str(data.get("translation") or "").strip()
            return out or None
        if self.kind == "libre":
            resp = self._client.post_json(
                f"{self.base}/translate",
                {"q": text, "source": src or "auto", "target": dst,
                 "format": "text"})
            if not getattr(resp, "ok", False):
                return None
            data = resp.json() or {}
            out = str(data.get("translatedText") or "").strip()
            return out or None
        return None

    def translate(self, text: str, src: str, dst: str) -> str | None:
        ok, _ = self.available()
        if not ok:
            return None
        limit = 450 if self.kind == "mymemory" else 900
        try:
            parts = []
            for chunk in _split_chunks(text, limit):
                out = self._one(chunk, src, dst)
                if not out:
                    self._health.note(self.key, self.name, self.base,
                                      ok=False,
                                      error="empty/failed response")
                    return None
                parts.append(out)
        except _Decline:
            return None
        except Exception as exc:  # noqa: BLE001
            self._health.note(self.key, self.name, self.base, ok=False,
                              error=f"{type(exc).__name__}: {exc}"[:200])
            return None
        self._health.note(self.key, self.name, self.base, ok=True)
        return " ".join(parts).strip() or None


# ── the organ ────────────────────────────────────────────────────────

class Translator:
    """The translation organ bound to one agent context.

    Construction never raises: every rung reports its own
    availability, and a context with nothing live still gets the
    marked identity fallback rather than an exception in a delivery
    path.
    """

    def __init__(self, context: Any = None, *, db: Any = None,
                 brain: Any = None, client: Any = None,
                 local_backend: Any = None, profile_kind: str = "",
                 health: _ProviderHealth | None = None,
                 receipt_store: Any = None) -> None:
        self.context = context
        self.settings = getattr(context, "settings", None)
        self.db = db if db is not None else getattr(context, "db", None)
        self._mem_lang = ""
        # Hot cache keys carry a caller-supplied extra ("" on the
        # plain path): the ticket path folds its glossary/preserve
        # fingerprint in, so differently-glossaried translations of
        # the same text never share an entry.
        self._cache: dict[tuple[str, str, str, str],
                          TranslationOutcome] = {}
        self._gate_cache: dict[tuple[str, str, str, str],
                               dict[str, Any]] = {}
        self._last_receipt: dict[str, Any] | None = None
        if client is None:
            tools = getattr(self.settings, "tools", None)
            client = HttpClient(
                timeout=15.0,
                user_agent=getattr(tools, "user_agent",
                                   "NoMoralsCore/0.1"),
                proxy_url=getattr(tools, "proxy_url", "") or "")
        self._client = client
        self._health = health or _ProviderHealth(self.db)
        tset = getattr(self.settings, "translate", None)
        model_ref = (getattr(tset, "local_model", "")
                     or os.environ.get("NM_TRANSLATE_MODEL", ""))
        home = (getattr(self.settings, "home", "")
                or os.path.expanduser("~/.nomorals"))
        self.home = home
        self._local_model_ref = model_ref
        self.receipts = receipt_store or TranslationReceiptStore(
            self.db, home)
        # The durable cache layer under the hot dicts (doc 19 §4,
        # T2): consulted before the ladder, written on success —
        # receipts distinguish hot-hit / durable-hit / miss.
        self._durable = TranslationCacheStore(self.db)
        if not profile_kind:
            profile_kind = self._detect_profile_kind()
        self.local = LocalNmtRung(model_ref=model_ref, home=home,
                                  profile_kind=profile_kind,
                                  backend=local_backend)
        self.brain = BrainRung(context, brain=brain)
        self.providers: list[ProviderRung] = [
            ProviderRung("mymemory", "MyMemory", "mymemory",
                         _MYMEMORY_URL, client, self._health),
            ProviderRung("lingva", "Lingva (seed instance)", "lingva",
                         _LINGVA_SEED, client, self._health),
            ProviderRung("libretranslate", "LibreTranslate (public)",
                         "libre", _LIBRE_SEED, client, self._health),
        ]
        extra = getattr(tset, "providers", "") or ""
        for i, base in enumerate(
                b.strip() for b in str(extra).split(",") if b.strip()):
            self.providers.append(ProviderRung(
                f"lingva-x{i + 1}", f"Lingva instance {base}", "lingva",
                base, client, self._health))

    # ── profile ──────────────────────────────────────────────────────

    @staticmethod
    def _detect_profile_kind() -> str:
        try:
            from ..core.profile import detect_profile

            return str(detect_profile().kind or "")
        except Exception:  # noqa: BLE001 — no gate when unknown
            return ""

    # ── owner language ───────────────────────────────────────────────

    def _ensure_prefs(self) -> None:
        if self.db is None:
            return
        try:
            self.db.execute(
                "CREATE TABLE IF NOT EXISTS translate_prefs "
                "(key TEXT PRIMARY KEY, value TEXT)")
        except Exception:  # noqa: BLE001
            _log.debug("translate: prefs table failed", exc_info=True)

    def owner_language(self) -> str:
        """The owner's language: stored pref -> settings -> env -> en."""
        if self.db is not None:
            try:
                self._ensure_prefs()
                row = self.db.query_one(
                    "SELECT value FROM translate_prefs WHERE key = ?",
                    ("owner_language",))
                saved = (row or {}).get("value") if row else None
                code = normalize_code(str(saved or ""))
                if code:
                    return code
            except Exception:  # noqa: BLE001
                _log.debug("translate: pref read failed", exc_info=True)
        if self._mem_lang:
            return self._mem_lang
        tset = getattr(self.settings, "translate", None)
        code = normalize_code(
            str(getattr(tset, "owner_language", "") or "")
            or os.environ.get("NM_TRANSLATE_OWNER_LANGUAGE", ""))
        return code or "en"

    def set_owner_language(self, raw: str) -> str:
        """Persist the owner's language; returns the ISO code.

        Raises ValueError naming the problem when ``raw`` names no
        language the organ knows — a typo must not silently store.
        """
        code = normalize_code(raw)
        if not code:
            raise ValueError(
                f"unknown language {raw!r} — name one I know "
                f"(e.g. en, fr, es, de, yo) or its English name")
        self._mem_lang = code
        if self.db is not None:
            try:
                self._ensure_prefs()
                self.db.execute(
                    "INSERT OR REPLACE INTO translate_prefs "
                    "(key, value) VALUES (?, ?)",
                    ("owner_language", code))
            except Exception:  # noqa: BLE001
                _log.debug("translate: pref save failed", exc_info=True)
        return code

    # ── detection ────────────────────────────────────────────────────

    @staticmethod
    def detect(text: str) -> tuple[str, float]:
        return detect_language(text)

    # ── the ladder ───────────────────────────────────────────────────

    def _rungs(self) -> list[Any]:
        return [self.local, self.brain, *self.providers]

    def ladder_status(self) -> list[dict[str, Any]]:
        """Per-rung availability, for /translate status. Honest by
        construction: it asks each rung, it does not guess."""
        out: list[dict[str, Any]] = []
        for rung in self._rungs():
            ok, reason = rung.available()
            label = getattr(rung, "rung_label", rung.name)
            out.append({"rung": label, "available": ok,
                        "detail": reason or "live"})
        out.append({"rung": "identity", "available": True,
                    "detail": "marked fallback — original text, "
                              "plainly labelled"})
        return out

    def _ladder_outcome(self, raw: str, src: str,
                        dst: str) -> TranslationOutcome | None:
        """The first rung that honestly answers, or None.

        One ladder walk, shared by ``translate()`` and by the
        quality gate's round trip (the gate's back-translation is a
        ladder walk in reverse — it must not mint receipts or
        recurse into the gate, so it comes through here).
        """
        for rung in self._rungs():
            ok, _reason = rung.available()
            if not ok:
                continue
            out = rung.translate(raw, src, dst)
            if not out:
                continue
            if (src and src != dst and out.strip() == raw.strip()
                    and _word_count(raw) >= 3):
                continue  # an echo is not a translation, any rung
            return TranslationOutcome(
                text=out, source_lang=src, target_lang=dst,
                rung=getattr(rung, "rung_label", rung.name),
                translated=True)
        return None

    # ── quality gate + receipts (doc 19 §4, pass T1) ─────────────

    @staticmethod
    def _gate_not_run(reason: str) -> dict[str, Any]:
        """A gate verdict for a check that honestly did not run."""
        return {"status": "not_run", "f1": None, "quality": None,
                "threshold": None, "judge_score": None,
                "reason": reason}

    def _gate_verdict(self, raw: str, outcome: TranslationOutcome, *,
                      exactness: str,
                      reverse_fn: Callable[[str, str, str], str]
                      | None = None) -> dict[str, Any]:
        """The bidirectional quality gate's verdict on one produced
        translation, in ``translation_quality``'s own shape.

        The gate is a MARK, not a rung selector: its shipped
        consumer (``i18n.bridge.score_contribution``) treats
        ``passed`` as the label a translation must earn, doc 19's
        rule is refuse-to-label-exact on failure, and doc 21's
        posture for weak-signal languages is warn — so a failed
        verdict never swaps or blocks the translation; it marks it
        unverified and (via llm.ticket) refuses the exact label.
        The gate runs for fidelity-class requests; anything it
        cannot honestly check is ``not_run`` with the reason, never
        a fake pass.  ``reverse_fn`` overrides the round trip's
        back-translation leg (a local-only ticket's gate may only
        ride the local rung).
        """
        if not outcome.translated:
            return self._gate_not_run(
                "identity fallback — no translation was produced "
                "to verify")
        if exactness not in FIDELITY_CLASSES:
            return self._gate_not_run(
                "gist class — the gate runs for fidelity-class "
                "requests (fidelity/legal/code)")
        src = outcome.source_lang or ""
        dst = outcome.target_lang or ""
        if not src:
            return self._gate_not_run(
                "source language undetected — a round trip needs a "
                "named source language")
        if src == dst:
            return self._gate_not_run("already in the target language")
        forward = outcome.text
        probe = raw.strip()

        def _fn(text: str, s: str, d: str) -> str:
            # The forward leg returns the ladder's chosen candidate
            # verbatim (the i18n bridge's adapter shape); only the
            # back-translation leg does new work.
            if text == probe and s == src and d == dst:
                return forward
            if reverse_fn is not None:
                return reverse_fn(text, s, d) or ""
            back = self._ladder_outcome(text, s, d)
            return back.text if back is not None else ""

        try:
            result = translate_with_quality(
                raw, source_lang=src, target_lang=dst,
                translate_fn=_fn)
        except Exception as exc:  # noqa: BLE001 — a crashed gate verifies nothing
            _log.warning("translate: quality gate failed to run: %s",
                         exc)
            return self._gate_not_run(
                "quality gate failed to run: "
                + type(exc).__name__)
        reason = result.judge_reason or (
            "round trip kept the original's content"
            if result.passed else
            "round trip lost too much of the original's content")
        return {"status": "passed" if result.passed else "failed",
                "f1": round(result.back_translation_f1, 4),
                "quality": round(result.quality, 4),
                "threshold": result.threshold,
                "judge_score": result.judge_score,
                "reason": reason}

    @staticmethod
    def _quality_mark(outcome: TranslationOutcome,
                      gate: dict[str, Any]) -> str:
        """The one-word quality truth a receipt carries."""
        if outcome.rung == "same-language":
            return "same-language"
        if not outcome.translated:
            return "identity"
        status = gate.get("status")
        if status == "passed":
            return "verified"
        if status == "failed":
            return "unverified"
        return "not_checked"

    def _mint_receipt(self, outcome: TranslationOutcome, *,
                      chars: int, cache: str, exactness: str,
                      gate: dict[str, Any],
                      requester: str) -> dict[str, Any]:
        """Build + persist the receipt for one translation.

        Fail-open: the store never raises, and the receipt is kept
        as ``last_receipt`` for facades (llm.ticket) either way.
        """
        receipt = {
            "receipt_id": new_receipt_id(),
            "at": time.time(),
            "source_lang": outcome.source_lang,
            "target_lang": outcome.target_lang,
            "rung": outcome.rung,
            "chars": int(chars),
            "translated": bool(outcome.translated),
            "cache": cache,
            "exactness": exactness,
            "gate": gate,
            "quality_mark": self._quality_mark(outcome, gate),
            "requester": requester or "",
        }
        try:
            self.receipts.append(receipt)
        except Exception:  # noqa: BLE001 — belt + braces: receipts never gate
            _log.debug("translate: receipt mint failed", exc_info=True)
        self._last_receipt = receipt
        return receipt

    @property
    def last_receipt(self) -> dict[str, Any] | None:
        """The receipt minted by this organ's most recent
        translation (None before the first).  Facades read the
        receipt id + gate verdict from here."""
        return self._last_receipt

    def recent_receipts(self, *, limit: int = 50, source_lang: str = "",
                        target_lang: str = "",
                        rung: str = "") -> list[dict[str, Any]]:
        """Recent translation receipts, newest first — the read API
        T2's surfaces render from.  Fail-open: a broken store reads
        as empty, never as an exception in a surface."""
        try:
            return self.receipts.recent(
                limit=limit, source_lang=source_lang,
                target_lang=target_lang, rung=rung)
        except Exception:  # noqa: BLE001
            _log.debug("translate: receipt read failed", exc_info=True)
            return []

    def _durable_lookup(self, key: tuple[str, str, str, str]
                        ) -> TranslationOutcome | None:
        """Promote a durable-cache entry into the hot dicts.

        A ``local``-rung entry whose recorded model reference
        differs from the configured one is invalidated (doc 19:
        cache invalidates on model change); expiry is the store's
        own check.  Anything unparsable reads as a miss.
        """
        raw, src, dst, extra = key
        payload = self._durable.get(cache_key_for(raw, src, dst, extra))
        if not payload:
            return None
        body = payload.get("outcome") or {}
        if body.get("rung") == "local" and str(
                payload.get("model_ref", "")) != self._local_model_ref:
            return None
        try:
            outcome = TranslationOutcome(
                text=str(body["text"]),
                source_lang=str(body.get("source_lang", src)),
                target_lang=str(body.get("target_lang", dst)),
                rung=str(body.get("rung", "identity")),
                translated=bool(body.get("translated", False)),
                note=str(body.get("note", "")))
        except (KeyError, TypeError, ValueError):
            return None
        if not outcome.translated:
            return None  # successes only — a stored failure is a miss
        self._cache[key] = outcome
        gate = payload.get("gate")
        self._gate_cache[key] = (gate if isinstance(gate, dict)
                                 else self._gate_not_run(
                                     "no verdict rode the cache entry"))
        return outcome

    def translate(self, text: str, *, target: str = "",
                  source: str = "", exactness: str = "gist",
                  requester: str = "",
                  cache_key_extra: str = "") -> TranslationOutcome:
        """Translate ``text`` down the ladder.

        ``target`` defaults to the owner's language; ``source`` to
        detection.  The result ALWAYS says what happened — which rung
        answered, or the marked identity when none could.
        ``exactness`` is the requested class (doc 19 §4): fidelity
        classes run the quality gate, whose verdict lands on the
        durable receipt this call mints either way.  ``requester``
        attributes the receipt when the call path carries one
        (news, the URL gist); "" is the honest unattributed mark.
        ``cache_key_extra`` separates cache entries that differ by
        more than text + pair (the ticket path's glossary/preserve
        fingerprint); "" keeps the plain organ key.
        """
        if exactness not in EXACTNESS_CLASSES:
            exactness = "gist"
        raw = text or ""
        dst = normalize_code(target) or self.owner_language()
        if not raw.strip():
            outcome = TranslationOutcome(text=raw, target_lang=dst,
                                         rung="identity",
                                         note="empty text")
            self._mint_receipt(
                outcome, chars=0, cache="miss", exactness=exactness,
                gate=self._gate_not_run("empty text"),
                requester=requester)
            return outcome
        src = normalize_code(source)
        if not src:
            detected, _conf = detect_language(raw)
            src = "" if detected == "und" else detected
        if src and src == dst:
            outcome = TranslationOutcome(
                text=raw, source_lang=src, target_lang=dst,
                rung="same-language",
                note="already in the target language")
            self._mint_receipt(
                outcome, chars=len(raw), cache="miss",
                exactness=exactness,
                gate=self._gate_not_run(
                    "already in the target language"),
                requester=requester)
            return outcome
        key = (raw, src, dst, cache_key_extra)
        cached = self._cache.get(key)
        cache_label = "hit"
        if cached is None:
            cached = self._durable_lookup(key)
            cache_label = "durable-hit"
        if cached is not None:
            gate = self._gate_cache.get(key)
            if gate is None or (
                    gate.get("status") == "not_run"
                    and exactness in FIDELITY_CLASSES):
                # A fidelity request on a gist-era cache entry earns
                # its verdict now, against the cached candidate.
                gate = self._gate_verdict(raw, cached,
                                          exactness=exactness)
                self._gate_cache[key] = gate
            self._mint_receipt(
                cached, chars=len(raw), cache=cache_label,
                exactness=exactness, gate=gate, requester=requester)
            return cached
        outcome = self._ladder_outcome(raw, src, dst)
        if outcome is None:
            outcome = TranslationOutcome(
                text=raw, source_lang=src, target_lang=dst,
                rung="identity",
                note="translation unavailable right now — every rung "
                     "declined or failed, so this is the ORIGINAL "
                     "text, not a translation")
        gate = self._gate_verdict(raw, outcome, exactness=exactness)
        # Only successes are cached: a transient outage must not pin
        # a text to the identity rung for the life of the process.
        if outcome.translated:
            if len(self._cache) > 512:
                self._cache.clear()
                self._gate_cache.clear()
            self._cache[key] = outcome
            self._gate_cache[key] = gate
            self._durable.put(
                cache_key_for(raw, src, dst, cache_key_extra),
                source_lang=src, target_lang=dst, rung=outcome.rung,
                payload={"outcome": outcome.to_dict(), "gate": gate,
                         "model_ref": self._local_model_ref})
        self._mint_receipt(
            outcome, chars=len(raw), cache="miss",
            exactness=exactness, gate=gate, requester=requester)
        return outcome

    # ── items + rows (the delivery shapes) ───────────────────────────

    def translate_item(self, title: str, summary: str = "", *,
                       target: str = "") -> ItemTranslation:
        """Translate one headline + gist pair (news-shaped)."""
        title = title or ""
        summary = summary or ""
        probe = f"{title} {summary}".strip()
        dst = normalize_code(target) or self.owner_language()
        src, _conf = detect_language(probe) if probe else ("und", 0.0)
        if src != "und" and src == dst:
            return ItemTranslation(title=title, summary=summary,
                                   source_lang=src,
                                   rung="same-language")
        t_out = self.translate(title, target=dst,
                               source="" if src == "und" else src,
                               requester="news")
        s_out = (self.translate(summary, target=dst,
                                source=t_out.source_lang or "",
                                requester="news")
                 if summary.strip() else None)
        translated = t_out.translated or bool(s_out and s_out.translated)
        if not translated:
            return ItemTranslation(
                title=title, summary=summary,
                source_lang=t_out.source_lang or src, rung=t_out.rung)
        return ItemTranslation(
            title=t_out.text if t_out.translated else title,
            summary=(s_out.text if (s_out and s_out.translated)
                     else summary),
            original_title=title if t_out.translated else "",
            original_summary=summary if (s_out and s_out.translated)
            else "",
            source_lang=t_out.source_lang or src,
            translated=True,
            rung=t_out.rung if t_out.translated else
            (s_out.rung if s_out else t_out.rung))

    def localize_rows(self, rows: list[dict[str, Any]], *,
                      target: str = "") -> list[dict[str, Any]]:
        """Copies of news rows with title/summary in the owner's
        language; translated rows carry ``title_original`` /
        ``summary_original`` / ``lang`` so renderers can keep the
        original headline as the small secondary line.  Input rows
        are never mutated (they are ledger rows)."""
        out: list[dict[str, Any]] = []
        for row in rows:
            copy = dict(row)
            item = self.translate_item(
                str(row.get("title", "") or ""),
                str(row.get("summary", "") or ""), target=target)
            if item.translated:
                copy["title"] = item.title
                copy["summary"] = item.summary
                if item.original_title:
                    copy["title_original"] = item.original_title
                if item.original_summary:
                    copy["summary_original"] = item.original_summary
                if item.source_lang:
                    copy["lang"] = item.source_lang
            out.append(copy)
        return out

    # ── URL gist ─────────────────────────────────────────────────────

    def translate_url(self, url: str, *, target: str = "") -> dict[str, Any]:
        """Fetch a page and translate it: the full text when short,
        a gist (head of the article) when long — labelled either way."""
        dst = normalize_code(target) or self.owner_language()
        try:
            resp = self._client.get(url)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "url": url,
                    "error": f"fetch failed: {type(exc).__name__}: {exc}"}
        if not getattr(resp, "ok", False):
            return {"ok": False, "url": url,
                    "error": f"fetch failed: HTTP "
                             f"{getattr(resp, 'status', '?')}"}
        page_title, body = _html_to_text(resp.text or "")
        if not body:
            return {"ok": False, "url": url,
                    "error": "the page carried no readable text"}
        full = len(body) <= 1500
        if full:
            chunk = body
        else:
            # gist = the head of the article, cut on a sentence
            # boundary so the translation never starts mid-thought.
            cut = body.rfind(". ", 0, 1500)
            chunk = body[:cut + 1] if cut > 400 else body[:1500]
        out = self.translate(chunk, target=dst, requester="url")
        title_out = (self.translate(page_title, target=dst,
                                    requester="url")
                     if page_title else None)
        return {
            "ok": True, "url": url, "full": full,
            "title": (title_out.text if title_out else ""),
            "title_original": page_title,
            "text": out.text, "translated": out.translated,
            "source_lang": out.source_lang,
            "target_lang": dst, "rung": out.rung, "note": out.note,
            "words": _word_count(body),
        }


_TAG_RE = re.compile(r"<[^>]+>")
_SCRIPT_RE = re.compile(
    r"<(script|style|noscript)[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>",
                       re.IGNORECASE | re.DOTALL)


def _html_to_text(markup: str) -> tuple[str, str]:
    """(page title, readable text) from raw HTML — stdlib only."""
    import html as _html

    title = ""
    m = _TITLE_RE.search(markup or "")
    if m:
        title = _html.unescape(_TAG_RE.sub(" ", m.group(1))).strip()
        title = re.sub(r"\s+", " ", title)
    body = _SCRIPT_RE.sub(" ", markup or "")
    body = _TAG_RE.sub(" ", body)
    body = _html.unescape(body)
    body = re.sub(r"[ \t]+", " ", body)
    body = re.sub(r"\n\s*\n+", "\n", body)
    lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
    return title, "\n".join(lines)


# ── context binding ──────────────────────────────────────────────────

def translator_for(context: Any) -> Translator:
    """The Translator for an agent context — one per context, cached
    (the ``brain_for`` pattern).  Never raises."""
    try:
        cached = getattr(context, "_translator_for_ctx", None)
        if isinstance(cached, Translator):
            return cached
        translator = Translator(context)
        try:
            context._translator_for_ctx = translator  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 — caching is a bonus
            pass
        return translator
    except Exception:  # noqa: BLE001
        _log.debug("translator_for failed; bare Translator",
                   exc_info=True)
        return Translator(None)


def owner_language(context: Any) -> str:
    """The owner's language for a context (default English)."""
    try:
        return translator_for(context).owner_language()
    except Exception:  # noqa: BLE001
        return "en"


def localize_news_rows(context: Any,
                       rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Delivery-path helper: rows localized for the owner.

    A failure inside the organ degrades to the identity rung (the
    original rows, visibly untranslated) — a delivery never dies
    because translation broke, and the failure is logged, not hidden.
    """
    if not rows:
        return rows
    try:
        return translator_for(context).localize_rows(rows)
    except Exception as exc:  # noqa: BLE001
        _log.warning("translate: localize_news_rows failed: %s", exc)
        return rows
