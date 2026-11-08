"""Webshot organ (DESIGN §18 + §20 expansion): render a web
page to a real image (or print PDF) and hand it back.

§18's build rule was mine-first: Devon already owns a browser
service whose screenshot path reloads a URL through headless
Chromium — but that path carries no variant space (no device
profiles, no dark mode, no element clip, no PDF-for-a-URL).
This organ is the full class over the same engine:

- **Rung 1 — the navigation pipeline** (``webnav``): the URL
  opens in Devon's real browser session (a persistent
  profile beside the book, so cookies and gate clearances
  survive across captures), settles, runs the interstitial
  pass (cookie banners, consent walls, age gates) and the
  bot-check pass (managed wait-out, checkbox click, owner
  takeover for hard puzzle walls) — and only then shoots:
  full-page / viewport / single-element clip, and
  ``page.pdf`` for the print rung.  A gate or wall that
  refuses to clear is FINAL: no seed may shoot over a
  refused page, and the refusal is the answer.  Fails fast
  and honestly when Playwright/Chromium is absent — never
  a fake or empty image (the browser service's own law).
  ``/webrec`` records through the same pipeline (the
  warm-up pass clears first; the recording opens on the
  real page).
- **Rung 2 — provider seeds** behind the agency SourceHealth
  breaker (the translate build's pattern): microlink and
  thum.io, both key-less seeds.  The open-sources law applies:
  these are SEEDS, and a provider is only attempted for the
  option subset it can actually honor — a dark-mode or clip
  request never silently degrades into a light viewport shot;
  when no rung can honor the request, the failure says so.
- **Page-watch** (§20b): a watch re-shots its URL on the
  message cycle (no daemon — the channel drive's discipline),
  measures the change against the last shot (pixel-diff
  ratio), and alerts only when the page actually changed.
- **Journal**: every request — success or failure — is
  recorded (URL, rung, size, requester, scope).  An unreadable
  journal raises after quarantining, like every other store in
  this system; it is never silently reset.

URL discipline: http(s) only, plus self-contained
``data:text/html`` pages.  Private / loopback / link-local
targets are refused unless the caller explicitly allows them
(operator DM use of his own machine) — a group member cannot
aim Devon's browser at the LAN she sits on.
"""

from __future__ import annotations

import base64
import hashlib
import ipaddress
import json
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

__all__ = [
    "ShotOptions", "CaptureResult", "ShotUsageError",
    "WebshotUnavailable", "WebshotStoreError",
    "DEVICE_PROFILES", "parse_shot_tail", "validate_url",
    "capture", "playwright_renderer", "nav_renderer",
    "provider_specs",
    "WebshotBook", "change_ratio", "image_dimensions",
    "CHANGE_THRESHOLD", "WATCH_MIN_INTERVAL_S",
    "RecOptions", "parse_rec_tail", "record_video",
]

#: Device profiles: name → Playwright context facts.  The
#: phone profile renders at scale 2 (a real phone shot is a
#: retina shot); width/height are the CSS viewport.
DEVICE_PROFILES: dict[str, dict[str, Any]] = {
    "desktop": {"width": 1440, "height": 900, "scale": 1,
                "mobile": False},
    "tablet": {"width": 768, "height": 1024, "scale": 2,
               "mobile": True},
    "phone": {"width": 390, "height": 844, "scale": 2,
              "mobile": True},
}

#: A watch fires at most this often, however it was written.
WATCH_MIN_INTERVAL_S = 900
WATCH_DEFAULT_INTERVAL_S = 3600

#: Pixel-diff mean (0..1) above which a re-shot counts as a
#: real change — below it is rendering noise (clock digits,
#: caret blink, ad rotation shimmer).
CHANGE_THRESHOLD = 0.02

_JOURNAL_RING = 300
_FETCH_CAP = 12_000_000
_UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")


class ShotUsageError(ValueError):
    """The request itself is malformed — the reply says why."""


class WebshotUnavailable(RuntimeError):
    """Every rung failed or refused; carries the per-rung truth."""

    def __init__(self, url: str, reasons: list[str]) -> None:
        self.url = url
        self.reasons = reasons
        super().__init__(
            f"no rung could shoot {url}: " + "; ".join(reasons))


class WebshotStoreError(RuntimeError):
    """The journal/watch store exists but cannot be read."""


@dataclass
class ShotOptions:
    """The §20 variant space of one shot."""

    full_page: bool = False
    device: str = "desktop"
    #: True only when the request NAMED the device.  An
    #: unchosen device is not a desktop request — the
    #: navigation pipeline walks its default fingerprint
    #: ladder (mobile first) for those.
    device_chosen: bool = False
    scheme: str = "light"          # light | dark
    wait: str = "load"             # load | idle
    clip: str = ""                 # CSS selector, "" = no clip
    pdf: bool = False
    annotate: bool = False         # §20c: the vision organ
                                   # reads the shot too

    def provider_compatible(self) -> dict[str, bool]:
        """Which provider seeds can honor these options."""
        base = not self.pdf and not self.clip
        return {
            # microlink: viewport PNG, light only.
            "microlink": base and not self.full_page
            and self.scheme == "light",
            # thum.io: width + full-page PNG, light only.
            "thumio": base and self.scheme == "light",
        }


@dataclass
class CaptureResult:
    data: bytes
    title: str = ""
    final_url: str = ""
    rung: str = ""
    pdf: bool = False
    width: int = 0
    height: int = 0
    #: Interstitial gates the navigation pass cleared on the
    #: way in (kinds, in order) — the reply names them.
    gates_cleared: list[str] = field(default_factory=list)
    #: How a bot check cleared ("managed_wait" /
    #: "checkbox_click" / "owner_takeover"), "" when the
    #: page was simply open.
    wall_via: str = ""
    #: The fingerprint that cleared, and every fingerprint
    #: tried before it (the ladder journal).
    presentation: str = ""
    presentations_tried: list[str] = field(
        default_factory=list)


# ── request grammar ──────────────────────────────────────────

_MODIFIERS = {
    "full": ("full_page", True),
    "fullpage": ("full_page", True),
    "viewport": ("full_page", False),
    "dark": ("scheme", "dark"),
    "light": ("scheme", "light"),
    "pdf": ("pdf", True),
    "phone": ("device", "phone"),
    "mobile": ("device", "phone"),
    "tablet": ("device", "tablet"),
    "desktop": ("device", "desktop"),
    "idle": ("wait", "idle"),
    "annotate": ("annotate", True),
    "read": ("annotate", True),
}


def _looks_like_url(token: str) -> bool:
    low = token.lower()
    return low.startswith(("http://", "https://", "data:")) \
        or ("." in token and " " not in token
            and not low.startswith("/"))


def parse_shot_tail(tail: str, *, reply_url: str = ""
                    ) -> tuple[str, ShotOptions]:
    """Parse ``/webshot``'s tail: modifiers in any order, then
    the URL — or no URL, when the caller supplies the replied
    message's link (``reply_url``).  ``clip`` takes the NEXT
    token as its selector.  Raises :class:`ShotUsageError`
    with the exact problem."""
    opts = ShotOptions()
    tokens = str(tail or "").split()
    url = ""
    i = 0
    while i < len(tokens):
        token = tokens[i]
        low = token.lower()
        if low == "clip":
            if i + 1 >= len(tokens):
                raise ShotUsageError(
                    "clip needs a CSS selector after it")
            opts.clip = tokens[i + 1]
            i += 2
            continue
        if low in _MODIFIERS:
            field_name, value = _MODIFIERS[low]
            setattr(opts, field_name, value)
            if field_name == "device":
                opts.device_chosen = True
            i += 1
            continue
        if _looks_like_url(token):
            if url:
                raise ShotUsageError(
                    "one URL per shot — drop the extra link")
            url = token
            i += 1
            continue
        raise ShotUsageError(
            f"I don't know the webshot option {token!r} — "
            "options: full, phone|tablet|desktop, dark, idle, "
            "pdf, annotate, clip <selector>, then the URL")
    if not url:
        url = str(reply_url or "").strip()
    if not url:
        raise ShotUsageError(
            "give me a URL, or reply to a message that has one: "
            "/webshot [full] [phone|tablet|desktop] [dark] "
            "[pdf] [clip <selector>] <url>")
    if not url.lower().startswith(("http://", "https://",
                                   "data:")):
        url = "https://" + url
    return url, opts


_URL_RE = None  # compiled lazily in extract_url


def extract_url(text: str) -> str:
    """The first URL in a block of text (the replied message's
    link, for reply-targeted shots).  Data extraction, not
    intent routing."""
    import re

    global _URL_RE
    if _URL_RE is None:
        _URL_RE = re.compile(r"https?://[^\s<>'\")\]]+")
    match = _URL_RE.search(str(text or ""))
    return match.group(0).rstrip(".,") if match else ""


# ── URL discipline ───────────────────────────────────────────

def _host_is_private(host: str) -> bool:
    """True when the host resolves only to non-public space.
    An unresolvable host is NOT called private here — the
    renderer will fail on it honestly, with its own error."""
    try:
        infos = socket.getaddrinfo(host, None)
    except (OSError, UnicodeError):
        return False
    for info in infos:
        try:
            ip = ipaddress.ip_address(info[4][0])
        except ValueError:
            continue
        if ip.is_private or ip.is_loopback or ip.is_link_local \
                or ip.is_reserved or ip.is_multicast:
            return True
    return False


def validate_url(url: str, *, allow_private: bool = False) -> str:
    """Normalize + judge a target URL.  Returns the URL to
    shoot; raises :class:`ShotUsageError` on refusal."""
    raw = str(url or "").strip()
    if not raw:
        raise ShotUsageError("the URL is empty")
    if raw.lower().startswith("data:"):
        if raw.lower().startswith("data:text/html"):
            return raw
        raise ShotUsageError(
            "only data:text/html pages can be shot — other "
            "data: payloads are not pages")
    parts = urllib.parse.urlsplit(raw)
    if parts.scheme not in ("http", "https"):
        raise ShotUsageError(
            f"I can only shoot http(s) pages, not "
            f"{parts.scheme or 'a bare path'}")
    host = parts.hostname or ""
    if not host:
        raise ShotUsageError("that URL has no host")
    if not allow_private and _host_is_private(host):
        raise ShotUsageError(
            f"{host} resolves to a private/loopback address — "
            "I don't aim the browser at the local network from "
            "a group request")
    return raw


# ── image facts ──────────────────────────────────────────────

def image_dimensions(data: bytes) -> tuple[int, int]:
    """(width, height) of PNG/JPEG/WebP bytes; (0, 0) when
    the bytes are not a readable image."""
    try:
        import io

        from PIL import Image

        with Image.open(io.BytesIO(data)) as img:
            return int(img.width), int(img.height)
    except Exception:  # noqa: BLE001 - unreadable is a fact
        return (0, 0)


def change_ratio(old: bytes, new: bytes) -> float:
    """Mean per-pixel difference (0..1) between two shots,
    compared on a common 64x64 grayscale grid so size changes
    alone don't read as content changes — a resize to the same
    grid keeps identical content at ~0."""
    import io

    from PIL import Image, ImageChops, ImageStat

    def _grid(data: bytes):
        with Image.open(io.BytesIO(data)) as img:
            return img.convert("L").resize((64, 64))

    diff = ImageChops.difference(_grid(old), _grid(new))
    return float(ImageStat.Stat(diff).mean[0]) / 255.0


# ── rung 1: local headless Chromium ──────────────────────────

def playwright_renderer(url: str, opts: ShotOptions
                        ) -> CaptureResult:
    """One shot through the repo's browser engine.  Raises
    RuntimeError with the concrete reason on any failure —
    the ladder records it and moves on; a page that never
    idles is still shot (idle is best-effort discipline, the
    load event is the contract)."""
    import importlib.util

    if importlib.util.find_spec("playwright") is None:
        raise RuntimeError(
            "playwright is not installed on this host "
            "(pip install playwright && playwright install "
            "chromium)")
    from playwright.sync_api import sync_playwright

    profile = DEVICE_PROFILES[opts.device]
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            try:
                context = browser.new_context(
                    viewport={"width": profile["width"],
                              "height": profile["height"]},
                    device_scale_factor=profile["scale"],
                    is_mobile=bool(profile["mobile"]),
                    has_touch=bool(profile["mobile"]),
                    color_scheme=opts.scheme)
                page = context.new_page()
                page.goto(url, timeout=30_000,
                          wait_until="load")
                if opts.wait == "idle":
                    try:
                        page.wait_for_load_state(
                            "networkidle", timeout=5_000)
                    except Exception:  # noqa: BLE001 - shot
                        pass           # anyway, per docstring
                title = ""
                try:
                    title = page.title() or ""
                except Exception:  # noqa: BLE001
                    title = ""
                if opts.pdf:
                    data = page.pdf(
                        width=f"{profile['width']}px",
                        height=f"{profile['height']}px",
                        print_background=True,
                        prefer_css_page_size=False)
                    return CaptureResult(
                        data=data, title=title,
                        final_url=page.url, pdf=True)
                if opts.clip:
                    loc = page.locator(opts.clip).first
                    try:
                        data = loc.screenshot(timeout=10_000)
                    except Exception as exc:  # noqa: BLE001
                        raise RuntimeError(
                            f"no element matched {opts.clip!r} "
                            f"on the page ({exc})") from exc
                else:
                    data = page.screenshot(
                        full_page=opts.full_page)
                return CaptureResult(
                    data=data, title=title, final_url=page.url)
            finally:
                browser.close()
    except RuntimeError:
        raise
    except Exception as exc:  # noqa: BLE001 - rung failure is
        raise RuntimeError(          # data for the ladder
            f"chromium shot failed: {exc}") from exc


def nav_renderer(url: str, opts: ShotOptions, *,
                 takeover_hook: Any = None,
                 profile_dir: Path | None = None,
                 on_step: Any = None) -> CaptureResult:
    """Rung 1: one shot through the navigation pipeline
    (``webnav``) — the page BEHIND its gates and bot
    checks, in Devon's persistent browser profile.
    GateBlocked / WallBlocked propagate typed: the ladder
    treats a refusal as final.  NavUnavailable (no engine)
    is a plain rung failure — the seeds may still try."""
    from . import webnav

    ladder = webnav.presentation_ladder(
        opts.device if opts.device_chosen else None)
    session, nav = webnav.navigate(
        url, ladder=ladder, scheme=opts.scheme,
        profile_dir=profile_dir, takeover_hook=takeover_hook,
        on_step=on_step)
    try:
        if opts.pdf:
            data = session.capture_still(pdf=True)
        elif opts.clip:
            page = session.page
            loc = page.locator(opts.clip).first
            try:
                data = bytes(loc.screenshot(timeout=10_000))
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    f"no element matched {opts.clip!r} on "
                    f"the page ({exc})") from exc
        else:
            data = session.capture_still(
                full_page=opts.full_page)
    finally:
        session.__exit__()
    return CaptureResult(
        data=data, title=nav.title, final_url=nav.final_url,
        rung="playwright", pdf=opts.pdf,
        gates_cleared=[g.kind for g in nav.gates],
        wall_via=nav.wall or "",
        presentation=nav.presentation,
        presentations_tried=list(nav.presentations_tried))


# ── rung 2: provider seeds (SourceHealth-gated) ──────────────

def _fetch(url: str, *, cap: int = _FETCH_CAP,
           timeout: float = 30.0) -> tuple[int, bytes, str]:
    """urllib GET following the sourcehealth fetcher's
    discipline (browser UA, never raises): (status, body,
    error).  Status 0 = transport failure."""
    req = urllib.request.Request(url, headers={
        "User-Agent": _UA, "Accept": "*/*"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return int(resp.status), resp.read(cap), ""
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read()[:200_000]
        except Exception:  # noqa: BLE001
            body = b""
        return int(exc.code), body, f"HTTP {exc.code}"
    except Exception as exc:  # noqa: BLE001 - transport failure
        return 0, b"", str(exc)[:300]


def _microlink(url: str, opts: ShotOptions, fetcher) -> bytes:
    api = ("https://api.microlink.io?url="
           + urllib.parse.quote(url, safe="")
           + "&screenshot=true&meta=false")
    status, body, err = fetcher(api)
    if status != 200 or not body:
        raise RuntimeError(f"microlink: {err or 'HTTP ' + str(status)}")
    try:
        payload = json.loads(body.decode("utf-8", "replace"))
        shot_url = payload["data"]["screenshot"]["url"]
    except (ValueError, KeyError, TypeError) as exc:
        raise RuntimeError(
            "microlink answered without a screenshot") from exc
    status, body, err = fetcher(str(shot_url))
    if status != 200 or not body:
        raise RuntimeError(
            f"microlink image fetch: {err or 'HTTP ' + str(status)}")
    return body


def _thumio(url: str, opts: ShotOptions, fetcher) -> bytes:
    profile = DEVICE_PROFILES[opts.device]
    width = int(profile["width"])
    api = f"https://image.thum.io/get/width/{width}"
    if opts.full_page:
        api += "/fullpage"
    api += "/" + url
    status, body, err = fetcher(api)
    if status != 200 or not body:
        raise RuntimeError(f"thum.io: {err or 'HTTP ' + str(status)}")
    return body


#: The provider seeds: name → fetch fn(url, opts, fetcher).
PROVIDERS: dict[str, Callable[..., bytes]] = {
    "microlink": _microlink,
    "thumio": _thumio,
}


def provider_specs() -> list[Any]:
    """SourceSpec seeds for the screenshot providers (the
    open-sources law: seeds, health-gated, never a closed
    list baked into the ladder's logic)."""
    from ...agency.sourcehealth import SourceSpec

    return [
        SourceSpec(key="webshot:microlink", category="webshot",
                   name="microlink", url="https://microlink.io",
                   kind="api",
                   notes="key-less screenshot seed"),
        SourceSpec(key="webshot:thumio", category="webshot",
                   name="thum.io", url="https://www.thum.io",
                   kind="api",
                   notes="key-less screenshot seed"),
    ]


def _provider_healthy(health: Any, spec: Any) -> bool:
    """Breaker gate: stood-down / disabled sources are skipped
    until a success re-activates them (the SourceHealth
    contract); unknown sources get their chance."""
    if health is None:
        return True
    try:
        rec = health.record(spec.key)
    except Exception:  # noqa: BLE001 - health never breaks
        return True
    if rec is None:
        return True
    return rec.status not in ("stood-down", "disabled")


def _note(health: Any, spec: Any, *, ok: bool,
          error: str = "") -> None:
    if health is None:
        return
    try:
        health.note_result(spec, ok=ok, error=error,
                           items=1 if ok else 0,
                           http=200 if ok else 0)
    except Exception:  # noqa: BLE001 - health never breaks
        pass


# ── the ladder ───────────────────────────────────────────────

Renderer = Callable[[str, ShotOptions], CaptureResult]


def capture(url: str, opts: ShotOptions, *,
            db: Any = None,
            local_renderer: Renderer | None = None,
            fetcher: Callable[..., tuple[int, bytes, str]] | None
            = None,
            takeover_hook: Any = None,
            profile_dir: Path | None = None,
            on_step: Any = None) -> CaptureResult:
    """Shoot ``url`` down the ladder.  ``local_renderer`` /
    ``fetcher`` are injectable (tests, alternate engines);
    production callers pass only ``db`` (for the breaker).
    Raises :class:`WebshotUnavailable` with every rung's
    reason when nothing could honor the request.

    Ladder discipline: rung 1 is the navigation pipeline.
    A gate/wall REFUSAL from it is final — the seeds are
    never allowed to shoot over a page that refused Devon
    (their shot would be the gate itself, or a cached
    rendering that pretends the wall isn't there).  An
    unavailable engine or an ordinary rung failure still
    falls through to the seeds, as before."""
    from . import webnav

    reasons: list[str] = []
    try:
        if local_renderer is not None:
            result = local_renderer(url, opts)
        else:
            result = nav_renderer(
                url, opts, takeover_hook=takeover_hook,
                profile_dir=profile_dir, on_step=on_step)
        result.rung = result.rung or "playwright"
        if not result.data:
            raise RuntimeError("the renderer returned no bytes")
        if not result.pdf:
            result.width, result.height = image_dimensions(
                result.data)
            if not result.width:
                raise RuntimeError(
                    "the renderer returned bytes that are not "
                    "a readable image")
        return result
    except (webnav.GateBlocked, webnav.WallBlocked) as exc:
        raise WebshotUnavailable(url, [str(exc)]) from exc
    except Exception as exc:  # noqa: BLE001 - next rung
        reasons.append(f"local chromium: {exc}")

    compatible = opts.provider_compatible()
    health = None
    specs: dict[str, Any] = {}
    if db is not None:
        try:
            from ...agency.sourcehealth import SourceHealth

            health = SourceHealth(db)
            specs = {s.name: s for s in provider_specs()}
        except Exception:  # noqa: BLE001 - breaker optional
            health = None
    fetch = fetcher or _fetch
    for name, fn in PROVIDERS.items():
        if not compatible.get(name):
            reasons.append(
                f"{name}: cannot honor these options "
                "(dark / clip / pdf / full-page beyond its "
                "space)")
            continue
        spec = specs.get(name)
        if spec is not None and not _provider_healthy(health,
                                                      spec):
            reasons.append(f"{name}: stood down by the "
                           "health breaker")
            continue
        try:
            data = fn(url, opts, fetch)
            width, height = image_dimensions(data)
            if not width:
                raise RuntimeError(
                    "the provider returned bytes that are not "
                    "a readable image")
            _note(health, spec, ok=True)
            return CaptureResult(data=data, rung=name,
                                 final_url=url, width=width,
                                 height=height)
        except Exception as exc:  # noqa: BLE001 - next seed
            _note(health, spec, ok=False, error=str(exc))
            reasons.append(f"{name}: {exc}")
    raise WebshotUnavailable(url, reasons)


# ── /webrec: the recording grammar + entry ───────────────────

#: Recording bounds: 5s is the floor the owner set; 60s is
#: the ceiling a chat video can stay watchable (and the
#: file shippable) at.  The default sits between.
REC_MIN_S = 5.0
REC_MAX_S = 60.0
REC_DEFAULT_S = 10.0

_REC_SECONDS = None  # compiled lazily in parse_rec_tail


@dataclass
class RecOptions:
    """One /webrec request."""

    seconds: float = REC_DEFAULT_S
    device: str = "desktop"
    #: True only when the request NAMED the device (see
    #: ShotOptions.device_chosen).
    device_chosen: bool = False
    scheme: str = "light"
    #: "min" / "max" when the requested duration was
    #: clamped to the bounds — the reply reports it.
    clamped: str = ""


def parse_rec_tail(tail: str, *, reply_url: str = ""
                   ) -> tuple[str, RecOptions]:
    """Parse ``/webrec``'s tail: ``[12s|12] [phone|tablet|
    desktop] [dark|light] <url>`` in any order, or no URL
    when the replied message supplies one.  Duration is
    clamped to [5s, 60s] and the clamp is RECORDED on the
    result, never silent."""
    global _REC_SECONDS
    if _REC_SECONDS is None:
        import re

        _REC_SECONDS = re.compile(r"^(\d+(?:\.\d+)?)s?$")
    opts = RecOptions()
    tokens = str(tail or "").split()
    url = ""
    for token in tokens:
        low = token.lower()
        match = _REC_SECONDS.match(low)
        if match and not _looks_like_url(token):
            opts.seconds = float(match.group(1))
            continue
        if low in ("phone", "mobile"):
            opts.device = "phone"
            opts.device_chosen = True
            continue
        if low in ("tablet", "desktop"):
            opts.device = low
            opts.device_chosen = True
            continue
        if low in ("dark", "light"):
            opts.scheme = low
            continue
        if _looks_like_url(token):
            if url:
                raise ShotUsageError(
                    "one URL per recording — drop the extra "
                    "link")
            url = token
            continue
        raise ShotUsageError(
            f"I don't know the webrec option {token!r} — "
            "options: [seconds, 5–60] [phone|tablet|desktop] "
            "[dark|light], then the URL")
    if not url:
        url = str(reply_url or "").strip()
    if not url:
        raise ShotUsageError(
            "give me a URL, or reply to a message that has "
            "one: /webrec [12s] [phone] [dark] <url>")
    if not url.lower().startswith(("http://", "https://",
                                   "data:")):
        url = "https://" + url
    if opts.seconds < REC_MIN_S:
        opts.seconds = REC_MIN_S
        opts.clamped = "min"
    elif opts.seconds > REC_MAX_S:
        opts.seconds = REC_MAX_S
        opts.clamped = "max"
    return url, opts


def record_video(url: str, opts: RecOptions, *,
                 out_dir: Path,
                 profile_dir: Path | None = None,
                 takeover_hook: Any = None,
                 on_step: Any = None) -> Any:
    """Record ``url`` through the navigation pipeline
    (``webnav.record_page``): the interstitial + bot-check
    passes run BEFORE the recording starts, so the video
    opens on the real page and scrolls it, eased, for the
    requested duration.  Returns a ``webnav.RecordOutcome``;
    raises ``webnav.NavError`` (typed) on honest failure."""
    from . import webnav

    ladder = webnav.presentation_ladder(
        opts.device if opts.device_chosen else None)
    return webnav.record_page(
        url, seconds=opts.seconds, out_dir=out_dir,
        ladder=ladder, scheme=opts.scheme,
        profile_dir=profile_dir, takeover_hook=takeover_hook,
        on_step=on_step)


# ── the book: journal + watches ──────────────────────────────

@dataclass
class WebshotBook:
    """Durable webshot state under the social data dir: the
    request journal (bounded ring) and the page-watch list
    with its baselines on disk beside it."""

    root: Path
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def path(self) -> Path:
        return self.root / "webshots.json"

    def load(self) -> "WebshotBook":
        if not self.path.exists():
            self.data = {"journal": [], "watches": [],
                         "watch_seq": 0}
            return self
        try:
            raw = json.loads(self.path.read_text(
                encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("webshot store root is not "
                                 "an object")
        except (OSError, ValueError) as exc:
            quarantine = self.path.with_suffix(
                f".corrupt-{int(time.time())}")
            try:
                self.path.replace(quarantine)
            except OSError:
                pass
            raise WebshotStoreError(
                f"webshot store at {self.path} is unreadable "
                f"({type(exc).__name__}: {exc}); quarantined "
                f"to {quarantine}") from exc
        raw.setdefault("journal", [])
        raw.setdefault("watches", [])
        raw.setdefault("watch_seq", 0)
        self.data = raw
        return self

    def save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1),
                       encoding="utf-8")
        tmp.replace(self.path)

    # ── journal ──
    def record(self, *, url: str, ok: bool, rung: str = "",
               error: str = "", size: int = 0, width: int = 0,
               height: int = 0, pdf: bool = False,
               scope: str = "", requester: str = "",
               options: str = "", ts: float | None = None
               ) -> dict[str, Any]:
        entry = {
            "ts": ts if ts is not None else time.time(),
            "url": url, "ok": bool(ok), "rung": rung,
            "error": str(error)[:300], "bytes": int(size),
            "width": int(width), "height": int(height),
            "pdf": bool(pdf), "scope": scope,
            "requester": requester, "options": options,
        }
        journal = self.data.setdefault("journal", [])
        journal.append(entry)
        del journal[:-_JOURNAL_RING]
        self.save()
        return entry

    def journal(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.data.get("journal") or []
        return list(rows[-limit:])

    # ── watches ──
    def add_watch(self, *, url: str, options: ShotOptions,
                  scope: str, requester: str,
                  interval_s: float = WATCH_DEFAULT_INTERVAL_S,
                  ts: float | None = None) -> dict[str, Any]:
        now = ts if ts is not None else time.time()
        seq = int(self.data.get("watch_seq") or 0) + 1
        self.data["watch_seq"] = seq
        watch = {
            "id": f"w{seq}", "url": url,
            "options": asdict(options), "scope": scope,
            "requester": requester,
            "interval_s": max(float(interval_s),
                              WATCH_MIN_INTERVAL_S),
            "created": now, "next_due": now,
            "last_shot": 0.0, "last_change": 0.0,
            "last_ratio": 0.0,
        }
        self.data.setdefault("watches", []).append(watch)
        self.save()
        return watch

    def remove_watch(self, watch_id: str) -> bool:
        rows = self.data.get("watches") or []
        kept = [w for w in rows if w.get("id") != watch_id]
        if len(kept) == len(rows):
            return False
        self.data["watches"] = kept
        baseline = self._baseline_path(watch_id)
        try:
            baseline.unlink()
        except OSError:
            pass
        self.save()
        return True

    def watches(self) -> list[dict[str, Any]]:
        return list(self.data.get("watches") or [])

    def _baseline_path(self, watch_id: str) -> Path:
        digest = hashlib.sha256(
            watch_id.encode("utf-8")).hexdigest()[:16]
        return self.root / "webshot_watches" / f"{digest}.png"

    def poll_watches(self, *, now: float | None = None,
                     db: Any = None,
                     local_renderer: Renderer | None = None,
                     fetcher: Callable[..., tuple[int, bytes,
                                                  str]] | None
                     = None) -> list[dict[str, Any]]:
        """One cycle pass over due watches: re-shoot, measure,
        and return the alerts (changed pages only).  A first
        shot sets the baseline silently; a failed re-shot is
        journaled and the watch retries next interval — a
        watch never alerts on an error.  Baselines persist
        across restarts (files beside the store)."""
        now = now if now is not None else time.time()
        alerts: list[dict[str, Any]] = []
        dirty = False
        for watch in self.data.get("watches") or []:
            if float(watch.get("next_due") or 0) > now:
                continue
            dirty = True
            watch["next_due"] = now + float(
                watch.get("interval_s")
                or WATCH_DEFAULT_INTERVAL_S)
            opts = ShotOptions(**{
                k: v for k, v in (watch.get("options") or {})
                .items()
                if k in ShotOptions.__dataclass_fields__})
            try:
                result = capture(
                    str(watch["url"]), opts, db=db,
                    local_renderer=local_renderer,
                    fetcher=fetcher)
            except WebshotUnavailable as exc:
                self.record(
                    url=str(watch["url"]), ok=False,
                    error=f"watch {watch['id']}: {exc}",
                    scope=str(watch.get("scope") or ""),
                    requester=str(watch.get("requester") or ""),
                    options="watch", ts=now)
                continue
            baseline = self._baseline_path(str(watch["id"]))
            old = baseline.read_bytes() \
                if baseline.exists() else b""
            watch["last_shot"] = now
            if old:
                ratio = change_ratio(old, result.data)
                watch["last_ratio"] = ratio
                if ratio > CHANGE_THRESHOLD:
                    watch["last_change"] = now
                    alerts.append({
                        "watch": watch, "ratio": ratio,
                        "result": result})
            baseline.parent.mkdir(parents=True, exist_ok=True)
            baseline.write_bytes(result.data)
            self.record(
                url=str(watch["url"]), ok=True,
                rung=result.rung, size=len(result.data),
                width=result.width, height=result.height,
                scope=str(watch.get("scope") or ""),
                requester=str(watch.get("requester") or ""),
                options="watch", ts=now)
        if dirty:
            self.save()
        return alerts


def default_book(workspace_dir: str | Path) -> WebshotBook:
    """The runtime's book: beside the group store."""
    root = Path(workspace_dir) / "data" / "social"
    return WebshotBook(root).load()
