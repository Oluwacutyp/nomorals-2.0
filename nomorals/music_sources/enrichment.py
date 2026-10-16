"""Enrichment chain: fingerprint -> AcoustID -> MusicBrainz -> one hop.

Every lookup is an injected callable. The chain never imports the
audio fingerprint module and never reaches for a network client on
its own. Callers wire real callables in; tests wire fakes in.

A missing step never raises: the step is recorded in ``steps`` with
status "missing" (or "error" when the callable raised) and its name
is added to ``missing`` on the result.

Layer 0: stdlib only. No imports from other packages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

__all__ = ["EnrichmentResult", "EnrichmentChain"]

STEP_NAMES = (
    "fingerprint",
    "acoustid",
    "musicbrainz",
    "discogs",
    "audiodb",
    "cover_art",
    "acousticbrainz",
)


@dataclass
class EnrichmentResult:
    """Outcome of one resolve() call."""

    fingerprint: Any = None
    acoustid_id: Optional[str] = None
    musicbrainz_id: Optional[str] = None
    discogs: Any = None
    audiodb: Any = None
    cover_art: Any = None
    acousticbrainz: Any = None
    steps: list = field(default_factory=list)
    missing: list = field(default_factory=list)


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (str, list, dict, tuple, set)) and len(value) == 0:
        return True
    return False


def _pick_id(value: Any, keys: tuple) -> Optional[str]:
    """Pull an id string out of a plain string or a mapping."""
    if isinstance(value, str) and value != "":
        return value
    if isinstance(value, dict):
        for key in keys:
            found = value.get(key)
            if isinstance(found, str) and found != "":
                return found
    return None


class EnrichmentChain:
    """Resolve audio into enriched identity via injected callables.

    All callables are optional. Each takes one argument (the best
    key available at that step) and returns whatever its service
    yields: a string id, a mapping, or a richer payload.
    """

    def __init__(
        self,
        fingerprint_fn: Optional[Callable[[Any], Any]] = None,
        acoustid_fn: Optional[Callable[[Any], Any]] = None,
        musicbrainz_fn: Optional[Callable[[Any], Any]] = None,
        discogs_fn: Optional[Callable[[Any], Any]] = None,
        audiodb_fn: Optional[Callable[[Any], Any]] = None,
        coverart_fn: Optional[Callable[[Any], Any]] = None,
        acousticbrainz_fn: Optional[Callable[[Any], Any]] = None,
    ) -> None:
        self.fingerprint_fn = fingerprint_fn
        self.acoustid_fn = acoustid_fn
        self.musicbrainz_fn = musicbrainz_fn
        self.discogs_fn = discogs_fn
        self.audiodb_fn = audiodb_fn
        self.coverart_fn = coverart_fn
        self.acousticbrainz_fn = acousticbrainz_fn

    def _run_step(self, name: str, fn: Optional[Callable[[Any], Any]],
                  arg: Any, steps: List[Dict[str, Any]],
                  missing: List[str]) -> Any:
        """Run one step, recording success / missing / error."""
        if fn is None:
            steps.append({"step": name, "status": "missing",
                          "detail": "no callable wired"})
            missing.append(name)
            return None
        try:
            value = fn(arg)
        except Exception as exc:  # noqa: BLE001 - never raise on a step
            steps.append({"step": name, "status": "error",
                          "detail": "%s: %s" % (type(exc).__name__, exc)})
            missing.append(name)
            return None
        if _is_empty(value):
            steps.append({"step": name, "status": "missing",
                          "detail": "callable returned empty"})
            missing.append(name)
            return None
        steps.append({"step": name, "status": "ok", "detail": "resolved"})
        return value

    def resolve(self, fingerprint_hash_or_path: Any) -> EnrichmentResult:
        """Run the chain for one input. Never raises on a missing step."""
        steps: List[Dict[str, Any]] = []
        missing: List[str] = []

        # Step 1: native fingerprint -> candidates.
        fp_raw = self._run_step("fingerprint", self.fingerprint_fn,
                                fingerprint_hash_or_path, steps, missing)
        fingerprint = fp_raw
        chain_key: Any = fingerprint_hash_or_path
        if isinstance(fp_raw, dict):
            fingerprint = (
                fp_raw.get("fingerprint") or fp_raw.get("hash") or fp_raw
            )
            chain_key = fingerprint
        elif isinstance(fp_raw, (list, tuple)) and len(fp_raw) > 0:
            chain_key = fp_raw[0]
        elif fp_raw is not None:
            chain_key = fp_raw

        # Step 2: AcoustID -> canonical MusicBrainz ID.
        ac_raw = self._run_step("acoustid", self.acoustid_fn, chain_key,
                                steps, missing)
        acoustid_id = _pick_id(ac_raw, ("acoustid_id", "id"))
        mb_from_acoustid = _pick_id(
            ac_raw, ("musicbrainz_id", "mbid", "musicbrainzid")
        )

        # Step 3: MusicBrainz -> identity.
        mb_key = mb_from_acoustid or acoustid_id or chain_key
        mb_raw = self._run_step("musicbrainz", self.musicbrainz_fn, mb_key,
                                steps, missing)
        musicbrainz_id = _pick_id(
            mb_raw, ("musicbrainz_id", "id", "mbid", "musicbrainzid")
        ) or mb_from_acoustid

        # Step 4: Discogs / TheAudioDB / Cover Art / AcousticBrainz,
        # one hop, each independent: one failing does not stop others.
        hop_key = musicbrainz_id or acoustid_id or chain_key
        discogs = self._run_step("discogs", self.discogs_fn, hop_key,
                                 steps, missing)
        audiodb = self._run_step("audiodb", self.audiodb_fn, hop_key,
                                 steps, missing)
        cover_art = self._run_step("cover_art", self.coverart_fn, hop_key,
                                   steps, missing)
        acousticbrainz = self._run_step(
            "acousticbrainz", self.acousticbrainz_fn, hop_key,
            steps, missing)

        return EnrichmentResult(
            fingerprint=fingerprint,
            acoustid_id=acoustid_id,
            musicbrainz_id=musicbrainz_id,
            discogs=discogs,
            audiodb=audiodb,
            cover_art=cover_art,
            acousticbrainz=acousticbrainz,
            steps=steps,
            missing=missing,
        )

    def resolve_batch(self, items: List[Any]) -> List[EnrichmentResult]:
        """Map resolve() over a list of inputs."""
        return [self.resolve(item) for item in items]
