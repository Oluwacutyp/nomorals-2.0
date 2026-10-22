"""Step 8 captcha flow: non-captcha first, classified, learned, capped.

This module is the *policy* around :mod:`nomorals.tools.captcha`
(detection + backends). It answers, in order:

1. **Non-captcha path first.** Before any challenge is solved, try
   the plain fetch — and an archive mirror — via
   :func:`resolve_without_captcha`. Most "walls" are just a page that
   also ships an API, an RSS feed, or an archived copy. Solving is
   what happens when the non-captcha paths fail, not the default.
2. **Classify, don't guess.** :func:`classify_challenge` separates
   *captcha* (a solvable widget) from *bot-wall* (an interactive
   interstitial with nothing to solve) from *rate-limit* (HTTP 429 —
   back off for ``Retry-After``, never "solve" a throttle).
3. **Learn.** :class:`SuccessLedger` records per ``(kind, site,
   method)`` outcomes so the next attempt picks the method that
   actually worked there.
4. **Cap.** :class:`OriginCostCeiling` enforces a per-origin,
   per-day spend ceiling for paid solving: on breach it pings the
   owner (injected callback) and refuses — it never overruns.
5. **Pipeline, cheapest first.** :class:`CaptchaFlow` runs
   passive → OCR/image-model → audio (local ASR first) → paid
   service (only if configured and within ceiling) → human ping
   (only when stuck). Every solve is audited with cost, and the audit
   never carries tokens or solved text.
6. **Audio captchas: local ASR first.**
   :func:`solve_audio_local_first` tries whisper.cpp
   (:class:`voice.stt.WhisperCppBackend`) before any paid service.

All external effects are injected (``fetcher``, ``ocr_fn``,
``service_fn``, ``ping_owner``, identity binder) so this module
performs no network, no spend, and no notification on its own — and
tests can prove each stage's ordering with fakes. The identity
binder (TrialVault-backed, one per service) is *not* imported here:
``tools`` (L4) may not import ``agents`` (L5) — callers inject it.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from ..core.logging_setup import get_logger
from .captcha import CaptchaChallenge, detect

__all__ = [
    "classify_challenge",
    "non_captcha_candidates",
    "resolve_without_captcha",
    "SuccessLedger",
    "OriginCostCeiling",
    "solve_audio_local_first",
    "CaptchaFlow",
    "solve_pipeline",
]

_log = get_logger(__name__)

#: body markers for a solvable captcha widget
_CAPTCHA_MARKERS = (
    "g-recaptcha", "h-captcha", "cf-turnstile", "data-sitekey",
    "geetest", "funcaptcha", "captcha-image", "captcha_image",
    # ByteDance/TikTok slider verify: in-house markup, no sitekey —
    # without these the fetch classifier calls it a bot-wall with
    # "nothing to solve", but the local slider solver solves it.
    "captcha_verify_container", "secsdk-captcha",
)

#: body markers for an interactive bot-wall interstitial
_BOT_WALL_MARKERS = (
    "just a moment", "cf-challenge", "__cf_chl", "verify you are human",
    "checking your browser", "attention required", "access denied",
    "bot detected", "unusual traffic", "px-captcha",
)

_DEFAULT_RETRY_AFTER_S = 60.0


# ── classification ───────────────────────────────────────────────────────────

def classify_challenge(status: int | None = None,
                       headers: dict[str, Any] | None = None,
                       body: str = "", url: str = "") -> dict[str, Any]:
    """Classify a response: captcha | bot_wall | rate_limit | ok | error.

    A 429 is a *rate limit* — the correct action is ``backoff`` for
    ``Retry-After`` seconds, never a solve attempt. A solvable widget
    is a *captcha*; an interstitial with no widget is a *bot-wall*
    (wait / change approach / ping the owner — there is nothing to
    solve). Never raises on odd input.
    """
    headers = {str(k).lower(): v for k, v in (headers or {}).items()}
    low = (body or "").lower()
    if status == 429:
        retry_after = _parse_retry_after(headers.get("retry-after"))
        return {"kind": "rate_limit", "action": "backoff",
                "retry_after_s": retry_after,
                "detail": "HTTP 429 — throttled, not challenged; "
                          "back off, do not solve"}
    if any(m in low for m in _CAPTCHA_MARKERS):
        return {"kind": "captcha", "action": "solve",
                "retry_after_s": None,
                "detail": "solvable captcha widget detected"}
    if any(m in low for m in _BOT_WALL_MARKERS):
        return {"kind": "bot_wall", "action": "wait_or_reroute",
                "retry_after_s": None,
                "detail": "interactive bot-wall interstitial — "
                          "nothing to solve, wait or reroute"}
    if status is not None and status >= 500:
        return {"kind": "error", "action": "retry_later",
                "retry_after_s": None,
                "detail": f"HTTP {status} — server error, not a challenge"}
    if status == 403:
        return {"kind": "bot_wall", "action": "wait_or_reroute",
                "retry_after_s": None,
                "detail": "HTTP 403 with no widget — treated as a wall"}
    if status is not None and 200 <= status < 300:
        return {"kind": "ok", "action": "proceed", "retry_after_s": None,
                "detail": f"HTTP {status} — no challenge"}
    return {"kind": "ok" if not low else "unknown",
            "action": "proceed", "retry_after_s": None,
            "detail": "no challenge markers found"}


def _parse_retry_after(value: Any) -> float:
    if value is None:
        return _DEFAULT_RETRY_AFTER_S
    try:
        return max(0.0, float(str(value).strip()))
    except (TypeError, ValueError):
        return _DEFAULT_RETRY_AFTER_S


# ── non-captcha path first ───────────────────────────────────────────────────

def non_captcha_candidates(url: str) -> list[dict[str, str]]:
    """Non-captcha ways to get ``url``: direct fetch, then archive."""
    url = (url or "").strip()
    if not url:
        return []
    return [
        {"method": "direct", "url": url},
        {"method": "archive",
         "url": f"https://web.archive.org/web/2id_/{url}"},
    ]


def resolve_without_captcha(url: str, *, fetcher: Callable[..., Any],
                            candidates: list[dict[str, str]] | None = None
                            ) -> dict[str, Any] | None:
    """Try the non-captcha paths. Returns the first clean fetch, else None.

    ``fetcher(url)`` returns ``(status, headers, body)`` or a dict
    with those keys. A fetch counts as clean only when
    :func:`classify_challenge` says ``ok`` — a wall page is not a
    result, however nicely it fetched. Fetcher errors fall through to
    the next candidate; they never become solves.
    """
    for cand in (candidates if candidates is not None
                 else non_captcha_candidates(url)):
        try:
            raw = fetcher(cand["url"])
        except Exception as exc:  # noqa: BLE001 - next candidate
            _log.debug("non-captcha %s failed for %s: %s",
                       cand["method"], cand["url"], exc)
            continue
        if isinstance(raw, dict):
            status, headers, body = (raw.get("status"), raw.get("headers") or {},
                                     raw.get("body") or "")
        else:
            status, headers, body = raw
        verdict = classify_challenge(status, headers, body, cand["url"])
        if verdict["kind"] == "ok":
            return {"method": cand["method"], "url": cand["url"],
                    "status": status, "body": body}
    return None


# ── learned success rates ────────────────────────────────────────────────────

def _ledger_key(kind: str, site: str, method: str) -> str:
    return f"{kind}||{site.lower()}||{method}"


class SuccessLedger:
    """Per-(kind, site, method) solve outcomes, persisted as JSON.

    ``rate`` is None until the method has at least one recorded
    attempt there — unknown is unknown, not zero.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def _load(self) -> dict[str, Any]:
        try:
            data = json.loads(self.path.read_text("utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def _save(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True), "utf-8")
        tmp.replace(self.path)

    def record(self, kind: str, site: str, method: str, *,
               ok: bool) -> None:
        data = self._load()
        key = _ledger_key(kind, site, method)
        row = data.get(key) or {"ok": 0, "fail": 0}
        row["ok" if ok else "fail"] = int(row.get("ok" if ok else "fail", 0)) + 1
        data[key] = row
        self._save(data)

    def rate(self, kind: str, site: str, method: str) -> float | None:
        row = self._load().get(_ledger_key(kind, site, method))
        if not row:
            return None
        total = int(row.get("ok", 0)) + int(row.get("fail", 0))
        if total <= 0:
            return None
        return int(row.get("ok", 0)) / total

    def best_method(self, kind: str, site: str, methods: list[str],
                    *, min_samples: int = 1) -> tuple[str, float | None]:
        """Best method for this (kind, site): highest learned rate.

        Untried methods have no rate; they lose to any tried method
        (exploit what worked), and when nothing was ever tried the
        first method wins (explore in caller order).
        """
        if not methods:
            raise ValueError("best_method needs at least one method")
        tried = [(m, self.rate(kind, site, m)) for m in methods]
        tried = [(m, r) for m, r in tried if r is not None]
        if not tried:
            return methods[0], None
        # prefer methods with enough samples, then highest rate, then order
        def _rank(item: tuple[str, float]) -> tuple[int, float, int]:
            method, rate = item
            row = self._load().get(_ledger_key(kind, site, method)) or {}
            samples = int(row.get("ok", 0)) + int(row.get("fail", 0))
            return (1 if samples >= min_samples else 0, rate,
                    -methods.index(method))
        method, rate = max(tried, key=_rank)
        return method, rate


# ── per-origin cost ceiling ──────────────────────────────────────────────────

def _origin_of(url_or_site: str) -> str:
    from urllib.parse import urlparse
    host = urlparse(url_or_site).hostname if "://" in str(url_or_site) else None
    return (host or str(url_or_site or "")).lower().strip()


class OriginCostCeiling:
    """Per-origin, per-day spend ceiling for paid solving.

    The ceiling never overruns: :meth:`can_spend` is checked before
    any paid call, and :meth:`check_or_ping` pings the owner (injected
    callback, best-effort) and refuses when the day's cap is hit.
    """

    def __init__(self, path: str | Path, *,
                 daily_cap_usd: float = 0.50) -> None:
        self.path = Path(path)
        self.daily_cap_usd = float(daily_cap_usd)

    @staticmethod
    def _today() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _load(self) -> dict[str, Any]:
        try:
            data = json.loads(self.path.read_text("utf-8"))
            if isinstance(data, dict) and data.get("day") == self._today():
                return data
        except (OSError, ValueError):
            pass
        return {"day": self._today(), "origins": {}}

    def _save(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True), "utf-8")
        tmp.replace(self.path)

    def spent_today(self, origin: str) -> float:
        row = self._load().get("origins", {}).get(_origin_of(origin)) or {}
        return float(row.get("spend", 0.0))

    def can_spend(self, origin: str, estimate_usd: float = 0.003
                  ) -> tuple[bool, str]:
        origin = _origin_of(origin)
        spent = self.spent_today(origin)
        if spent + float(estimate_usd) > self.daily_cap_usd:
            return False, (f"origin {origin}: daily cap "
                           f"${self.daily_cap_usd:.2f} would be exceeded "
                           f"(${spent:.4f} spent)")
        return True, ""

    def record_spend(self, origin: str, cost_usd: float,
                     kind: str = "") -> None:
        data = self._load()
        origins = data.setdefault("origins", {})
        row = origins.setdefault(_origin_of(origin), {"spend": 0.0, "solves": 0})
        row["spend"] = round(float(row.get("spend", 0.0)) + float(cost_usd), 6)
        row["solves"] = int(row.get("solves", 0)) + 1
        if kind:
            row["last_kind"] = kind
        self._save(data)

    def check_or_ping(self, origin: str, estimate_usd: float = 0.003, *,
                      ping_owner: Callable[[dict[str, Any]], Any] | None = None
                      ) -> dict[str, Any]:
        """Gate a paid solve: allowed, or refused + owner pinged."""
        ok, reason = self.can_spend(origin, estimate_usd)
        if ok:
            return {"allowed": True, "needs_owner_ping": False,
                    "spent_today": self.spent_today(origin),
                    "cap": self.daily_cap_usd}
        payload = {"origin": _origin_of(origin),
                   "spent_today": self.spent_today(origin),
                   "cap": self.daily_cap_usd,
                   "estimate_usd": float(estimate_usd),
                   "reason": reason}
        if ping_owner is not None:
            try:
                ping_owner(payload)
            except Exception as exc:  # noqa: BLE001 - ping never sinks the gate
                _log.warning("origin ceiling ping failed: %s", exc)
        return {"allowed": False, "needs_owner_ping": True, **payload}


# ── audio captchas: local ASR first ──────────────────────────────────────────

def _local_asr(audio: Any, language: str = "en") -> str:
    """Local whisper.cpp transcription. "" when unavailable/empty."""
    try:
        from ..voice.stt import WhisperCppBackend  # lazy: same layer
        backend = WhisperCppBackend()
    except Exception as exc:  # noqa: BLE001 - unavailable is honest ""
        _log.debug("whisper.cpp unavailable for audio captcha: %s", exc)
        return ""
    path = audio
    tmp: str | None = None
    try:
        if isinstance(audio, (bytes, bytearray)):
            fd, tmp = tempfile.mkstemp(suffix=".wav")
            with os.fdopen(fd, "wb") as fh:
                fh.write(bytes(audio))
            path = tmp
        result = backend.transcribe(str(path), language=language)
        if isinstance(result, dict):
            return str(result.get("text") or "").strip()
        return str(result or "").strip()
    except Exception as exc:  # noqa: BLE001 - local ASR failure → fall through
        _log.debug("local audio-captcha ASR failed: %s", exc)
        return ""
    finally:
        if tmp:
            try:
                os.unlink(tmp)
            except OSError:
                pass


def solve_audio_local_first(audio: Any, *,
                            asr_fn: Callable[..., Any] | None = None,
                            service_fn: Callable[..., Any] | None = None,
                            language: str = "en") -> dict[str, Any]:
    """Solve an audio captcha: local ASR first, paid service only after.

    Returns ``{"ok", "backend", "text", "local"}``. ``asr_fn``/``service_fn``
    are injectable (tests + callers with their own ASR); the default
    ``asr_fn`` is the local whisper.cpp path above.
    """
    if asr_fn is not None:
        text = str(asr_fn(audio, language=language) or "").strip()
    else:
        text = _local_asr(audio, language=language)
    if text:
        return {"ok": True, "backend": "whisper.cpp", "text": text,
                "local": True}
    if service_fn is not None:
        res = service_fn(audio) or {}
        text = str(res.get("text") or res.get("token") or "").strip()
        return {"ok": bool(text), "backend": "service", "text": text,
                "local": False,
                "cost_usd": float(res.get("cost_usd") or 0.0)}
    return {"ok": False, "backend": "none", "text": "", "local": True}


# ── staged pipeline ──────────────────────────────────────────────────────────

class CaptchaFlow:
    """Passive → OCR → audio → paid service → human ping (only if stuck).

    Every attempt is audited (kind, site, stage, ok, cost — never
    tokens or solved text). Rate-limits short-circuit to backoff
    before any solve stage runs.
    """

    def __init__(self, *, ledger: SuccessLedger | None = None,
                 ceiling: OriginCostCeiling | None = None,
                 audit_path: str | Path | None = None,
                 ocr_fn: Callable[..., Any] | None = None,
                 audio_fn: Callable[..., Any] | None = None,
                 slider_fn: Callable[..., Any] | None = None,
                 service_fn: Callable[..., Any] | None = None,
                 ping_owner: Callable[[dict[str, Any]], Any] | None = None,
                 fetcher: Callable[..., Any] | None = None,
                 estimate_usd: float = 0.003) -> None:
        self.ledger = ledger
        self.ceiling = ceiling
        self.audit_path = Path(audit_path) if audit_path else None
        self.ocr_fn = ocr_fn
        self.audio_fn = audio_fn
        self.slider_fn = slider_fn
        self.service_fn = service_fn
        self.ping_owner = ping_owner
        self.fetcher = fetcher
        self.estimate_usd = float(estimate_usd)

    # -- audit ---------------------------------------------------------------
    def _audit(self, entry: dict[str, Any]) -> None:
        """Append one audit row. Never carries tokens or solved text."""
        safe = {k: v for k, v in entry.items()
                if k not in ("token", "text", "image_bytes", "audio_bytes")}
        safe.setdefault("ts", time.time())
        if self.audit_path is None:
            return
        try:
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.audit_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(safe) + "\n")
        except OSError:
            _log.warning("captcha-flow audit write failed")

    def _record(self, kind: str, site: str, stage: str, *,
                ok: bool, cost: float = 0.0) -> None:
        if self.ledger is not None:
            try:
                self.ledger.record(kind, site, stage, ok=ok)
            except Exception:  # noqa: BLE001 - ledger never sinks a solve
                _log.debug("captcha-flow ledger write failed", exc_info=True)
        self._audit({"kind": kind, "site": site, "stage": stage,
                     "ok": ok, "cost_usd": round(float(cost), 6)})

    # -- local kind-solver stage ---------------------------------------------
    def _local_stage(self, ch: dict[str, Any], *, kind: str, site: str,
                     tried: list[str]) -> dict[str, Any] | None:
        """Dispatch to the per-kind local solver registry.

        Returns the result dict on a local clear, ``None`` to fall
        through to the later stages (no solver for the kind, no live
        session, or the local attempt failed — the failure is
        recorded either way).
        """
        fn = self.slider_fn
        if fn is None:
            try:
                from ..browser import slider_solver as _ss  # noqa: F401
                from ..browser.kind_solvers import kind_solver_for
            except Exception:  # noqa: BLE001 - spine import is additive
                return None
            if kind_solver_for(kind) is None:
                return None
            metadata = ch.get("metadata") or {}
            session = metadata.get("session") or metadata.get("tab")
            if session is None:
                self._record(kind, site, "local_no_session", ok=False)
                return None

            def fn(challenge_dict: dict[str, Any],
                   _session: Any = session) -> dict[str, Any]:
                from ..browser.kind_solvers import kind_solver_for as kf
                solver_fn = kf(str(challenge_dict.get("kind") or ""))
                if solver_fn is None:
                    return {"ok": False,
                            "detail": "kind solver vanished"}
                return solver_fn(_session, challenge_dict)

        tried.append("local")
        try:
            res = fn(ch) or {}
        except Exception as exc:  # noqa: BLE001 - later stages remain
            res = {"ok": False, "error": str(exc)}
        ok = bool(res.get("ok"))
        self._record(kind, site, f"local_{kind}", ok=ok)
        if ok:
            return {"ok": True, "stage": f"local_{kind}", "kind": kind,
                    "local": True,
                    "detail": str(res.get("detail") or "")}
        return None

    # -- solve ---------------------------------------------------------------
    def solve(self, challenge: Any, *,
              url: str | None = None) -> dict[str, Any]:
        """Run the pipeline for one challenge. Dicts and CaptchaChallenge both accepted."""
        if isinstance(challenge, CaptchaChallenge):
            ch: dict[str, Any] = {"kind": challenge.kind,
                                  "sitekey": challenge.sitekey,
                                  "page_url": challenge.page_url,
                                  "image_url": challenge.image_url,
                                  "image_bytes": challenge.image_bytes,
                                  "action": challenge.action,
                                  "metadata": dict(challenge.metadata)}
        elif isinstance(challenge, dict):
            ch = dict(challenge)
        else:
            raise ValueError("solve needs a CaptchaChallenge or a dict")
        kind = str(ch.get("kind", "") or "unknown")
        page_url = url or str(ch.get("page_url") or "")
        site = _origin_of(page_url) if page_url else ""

        # stage 0 — non-captcha path first (direct + archive mirror)
        if page_url and self.fetcher is not None:
            clean = resolve_without_captcha(page_url, fetcher=self.fetcher)
            if clean is not None:
                self._record(kind, site, "non_captcha", ok=True)
                return {"ok": True, "stage": "non_captcha",
                        "method": clean["method"], "url": clean["url"],
                        "kind": kind}

        # stage 0b — classification: a 429 is backoff, never a solve
        http_status = ch.get("http_status")
        if http_status is not None or ch.get("headers") is not None or ch.get("body"):
            verdict = classify_challenge(http_status, ch.get("headers") or {},
                                         ch.get("body") or "", page_url)
            if verdict["kind"] == "rate_limit":
                self._record(kind, site, "rate_limit", ok=False)
                return {"ok": False, "stage": "rate_limit", "kind": "rate_limit",
                        "action": "backoff",
                        "retry_after_s": verdict["retry_after_s"],
                        "detail": verdict["detail"]}
            if verdict["kind"] == "bot_wall" and kind in ("", "unknown"):
                self._record(kind, site, "bot_wall", ok=False)
                return {"ok": False, "stage": "bot_wall", "kind": "bot_wall",
                        "action": "wait_or_reroute",
                        "detail": verdict["detail"]}

        # stage 1 — passive: detect (report only). Nothing detected and
        # no explicit kind → there is no challenge to solve.
        html = str(ch.get("html") or ch.get("body") or "")
        if html and kind in ("", "unknown"):
            found = detect(html, page_url)
            if found:
                kind = found[0].kind
            else:
                # slider markup is invisible to the token detector —
                # the browser layer's slider detection names it.
                try:
                    from ..browser.slider_solver import (
                        detect_slider_challenge)
                    if detect_slider_challenge(html, page_url) is not None:
                        kind = "slider"
                except Exception:  # noqa: BLE001 - detection is additive
                    pass
            if kind in ("", "unknown"):
                self._record(kind, site, "passive", ok=True)
                return {"ok": True, "stage": "passive",
                        "kind": "no_challenge",
                        "detail": "passive detection found no challenge"}

        tried: list[str] = ["passive"]
        self._record(kind, site, "passive", ok=True)

        # stage 1b — local kind solver (in-session; the slider solver
        # is the first registrant). Runs BEFORE any paid stage whenever
        # the kind has a registered local solver AND the challenge
        # carries a live session (or a caller injected a solver fn).
        # No session → recorded skip, flow continues unchanged.
        local_out = self._local_stage(ch, kind=kind, site=site,
                                      tried=tried)
        if local_out is not None:
            return local_out

        # stage 2 — OCR / image model (image captchas only)
        if kind == "image_captcha" and self.ocr_fn is not None:
            tried.append("ocr")
            try:
                res = self.ocr_fn(ch) or {}
            except Exception as exc:  # noqa: BLE001 - next stage
                res = {"ok": False, "error": str(exc)}
            text = str(res.get("text") or "").strip()
            self._record(kind, site, "ocr", ok=bool(text))
            if text:
                return {"ok": True, "stage": "ocr", "kind": kind, "text": text}

        # stage 3 — audio (audio captchas; local ASR before any service)
        if kind == "audio_captcha":
            tried.append("audio")
            audio = ch.get("image_bytes") or ch.get("audio_bytes") or \
                ch.get("audio_url") or ch.get("image_url") or b""
            if self.audio_fn is not None:
                res = self.audio_fn(audio) or {}
            else:
                res = solve_audio_local_first(
                    audio, service_fn=self.service_fn)
            text = str(res.get("text") or "").strip()
            cost = float(res.get("cost_usd") or 0.0)
            self._record(kind, site, "audio", ok=bool(text), cost=cost)
            if text:
                return {"ok": True, "stage": "audio", "kind": kind,
                        "text": text, "local": bool(res.get("local", True))}
            if res.get("backend") == "service" and cost and self.ceiling is not None:
                self.ceiling.record_spend(site or kind, cost, kind)

        # stage 4 — paid service, only if configured and within ceiling
        if self.service_fn is not None:
            tried.append("service")
            gate = {"allowed": True, "needs_owner_ping": False}
            if self.ceiling is not None:
                gate = self.ceiling.check_or_ping(
                    site or kind, self.estimate_usd,
                    ping_owner=self.ping_owner)
            if gate["allowed"]:
                try:
                    res = self.service_fn(ch) or {}
                except Exception as exc:  # noqa: BLE001 - human stage next
                    res = {"ok": False, "error": str(exc)}
                cost = float(res.get("cost_usd") or 0.0)
                ok = bool(res.get("ok") and
                          (res.get("token") or res.get("text")))
                self._record(kind, site, "service", ok=ok, cost=cost)
                if cost and self.ceiling is not None:
                    self.ceiling.record_spend(site or kind, cost, kind)
                if ok:
                    out: dict[str, Any] = {"ok": True, "stage": "service",
                                           "kind": kind, "cost_usd": cost}
                    if res.get("token"):
                        out["token"] = res["token"]
                    if res.get("text"):
                        out["text"] = res["text"]
                    return out
            else:
                self._record(kind, site, "service_blocked_by_ceiling", ok=False)
                return {"ok": False, "stage": "ceiling", "kind": kind,
                        "takeover": False, "needs_owner_ping": True,
                        "detail": "paid solving blocked by the per-origin "
                                  "daily ceiling — owner pinged"}

        # stage 5 — human ping, only when stuck (every cheaper stage failed)
        tried.append("human")
        if self.ping_owner is not None:
            try:
                self.ping_owner({"kind": kind, "site": site,
                                 "page_url": page_url, "stages_tried": tried,
                                 "detail": "all automatic stages failed — "
                                           "owner solve needed"})
            except Exception as exc:  # noqa: BLE001 - takeover stands
                _log.warning("captcha human ping failed: %s", exc)
        self._record(kind, site, "human", ok=False)
        return {"ok": False, "stage": "human", "kind": kind,
                "takeover": True, "stages_tried": tried,
                "detail": "all automatic stages failed — owner pinged"}


def solve_pipeline(challenge: Any, *, url: str | None = None,
                   **flow_kwargs: Any) -> dict[str, Any]:
    """One-call pipeline: build a :class:`CaptchaFlow` and solve."""
    return CaptchaFlow(**flow_kwargs).solve(challenge, url=url)
