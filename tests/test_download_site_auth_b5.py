"""Downloads B5 pins — the authenticated rung.

A site whose descriptor declares auth (or whose connector reports a
classified ``login`` block) resolves its session at run time from
the owner's own account — the unified cookie estate first, then the
accounts vault — and downloads through the normal chain.  A missing
session fails as found-but-blocked, class ``login``, with the exact
owner action; the chain never runs anonymously for a site that
declared it needs an account.

Pins:
* the resolver: estate session wins without a vault touch; a
  non-session cookie (an anonymous clearance) does not count as a
  login; a vault session resolves and lands in the estate, and the
  next run reuses the estate without touching the vault; missing /
  expired / locked-vault each report their own status with the
  owner action, never a value;
* the up-front gate: an auth-declared descriptor with no session
  fails LOGIN with no anonymous stage attempted; an authenticated
  download succeeds through the normal chain;
* the login-block retry (SoundCloud proof): a connector ``login``
  block retries the direct stage once, authenticated through the
  estate's cookies file, and lands; without a session the chain
  stops there — proxy / rendered / browser never run; a non-login
  block still falls through exactly as before (B4 behaviour);
* the secret value appears in no log line, no report text, no
  journal entry;
* descriptor validation: auth without auth_domains fails at LOAD
  naming site + field; the new fields round-trip through dicts.

Everything is faked at the seams — no network, no browser engine,
no real vault.  The one real credential in this file is a
clearly-marked test-only value (``TEST_SESSION_VALUE``) that no
production path ever sees.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest

from nomorals.connectors.soundcloud import SoundCloudError
from nomorals.media.downloader import (
    FailureKind,
    MediaDownloader,
    StageLog,
)
from nomorals.media.site_auth import (
    SiteAuthResolver,
    store_site_session,
    vault_key,
)
from nomorals.media.site_descriptors import (
    SOUNDCLOUD_DESCRIPTOR,
    SiteDescriptor,
    SiteDescriptorError,
    SiteDescriptorRegistry,
)
from nomorals.storage.cookie_estate import CookieEstate
from nomorals.storage.db import open_database

#: Test-only session value.  It stands in for a real credential in
#: every flow below; the "never in logs" pins assert it appears in
#: no log line, no report text, and no journal entry.
TEST_SESSION_VALUE = "TEST-SESSION-ONLY-9f3a7d1e"

SC_URL = "https://soundcloud.com/artist/gated-track"


# ── shared fakes ────────────────────────────────────────────────────


class _DictStore:
    """A dict-backed kit-style get/set store (the vault seam)."""

    def __init__(self, entries: dict[str, Any] | None = None, *,
                 raise_on_get: bool = False) -> None:
        self.entries: dict[str, Any] = dict(entries or {})
        self.gets = 0
        self.raise_on_get = raise_on_get
        self.unavailable_reason = ""

    def get(self, key: str) -> Any | None:
        self.gets += 1
        if self.raise_on_get:
            raise RuntimeError("vault get exploded (test)")
        return self.entries.get(key)

    def set(self, key: str, entry: dict[str, Any]) -> None:
        self.entries[key] = entry


class _LockedStore:
    """A vault that cannot open: reads return None and the reason
    is carried (the real VaultCookieStore's locked posture)."""

    def __init__(self) -> None:
        self.gets = 0
        self.unavailable_reason = "vault locked (test)"

    def get(self, key: str) -> Any | None:
        self.gets += 1
        return None

    def set(self, key: str, entry: dict[str, Any]) -> None:
        raise RuntimeError("vault locked (test)")


class _FakeKit:
    def __init__(self, payload: bytes = b"connector-bytes") -> None:
        self.payload = payload
        self.calls: list[str] = []

    def download(self, url, dest, progress=None):
        self.calls.append(url)
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes(self.payload)
        return SimpleNamespace(
            ok=True, path=str(dest), size_bytes=len(self.payload),
            error="", resumed=False, restarted=False)


def _no_descriptor_extractors():
    return SimpleNamespace(best_for=lambda url: None)


def _fail_direct(*a, **k):
    raise RuntimeError("connection reset by peer")


def _estate(tmp_path: Path) -> CookieEstate:
    """A fresh estate over a tmp DB, all legacy roots tmp."""
    base = {
        "profile_dir": tmp_path / "profile",
        "service_data_dir": tmp_path / "service",
        "sessions_dir": tmp_path / "sessions",
        "solved_dir": tmp_path / "solved",
    }
    for path in base.values():
        Path(path).mkdir(parents=True, exist_ok=True)
    db, _, _ = open_database(str(tmp_path / "estate.sqlite3"))
    estate = CookieEstate(db, legacy_roots=base)
    estate.ensure()
    return estate


def _session_cookie(domain: str, name: str = "auth_token",
                    *, value: str = TEST_SESSION_VALUE,
                    expires: float = 0) -> dict[str, Any]:
    return {"domain": domain, "name": name, "path": "/",
            "value": value, "expires": expires, "secure": True,
            "httpOnly": True, "sameSite": "Lax"}


def _toy_descriptor(**overrides: Any) -> SiteDescriptor:
    base: dict[str, Any] = {
        "site": "exampletunes",
        "url_patterns": (r"https?://([a-z0-9-]+\.)*exampletunes\.test/",),
        "chain": ("direct",),
        "auth": "vault",
        "auth_domains": ("exampletunes.test",),
        "auth_cookie_names": ("sess",),
    }
    base.update(overrides)
    return SiteDescriptor(**base)


def _downloader(tmp_path: Path, *, site_descriptors=None,
                site_auth=None, connector=None):
    return MediaDownloader(
        None, direct_retries=1, soundcloud_connector=connector,
        kit_downloader=_FakeKit(),
        extractors=_no_descriptor_extractors(),
        site_descriptors=site_descriptors, site_auth=site_auth)


def _resolver(tmp_path: Path, store: Any, estate: Any = None
              ) -> SiteAuthResolver:
    return SiteAuthResolver(
        vault_store=store,
        cookie_estate=estate if estate is not None
        else _estate(tmp_path))


# ── the resolver itself ─────────────────────────────────────────────


class TestResolver:
    def test_auth_none_never_touches_vault_or_estate(
            self, tmp_path):
        store = _DictStore()
        auth = _resolver(tmp_path, store).resolve(
            SOUNDCLOUD_DESCRIPTOR)
        assert auth.ok and auth.source == ""
        assert store.gets == 0

    def test_estate_session_wins_without_vault_touch(
            self, tmp_path):
        store = _DictStore()
        estate = _estate(tmp_path)
        estate.upsert([_session_cookie(
            "exampletunes.test", name="sess")], source="test")
        auth = SiteAuthResolver(
            vault_store=store, cookie_estate=estate).resolve(
                _toy_descriptor())
        assert auth.ok and auth.source == "estate"
        assert len(auth.cookies) == 1
        assert store.gets == 0

    def test_anonymous_clearance_is_not_a_login(
            self, tmp_path):
        store = _DictStore()
        estate = _estate(tmp_path)
        estate.upsert([_session_cookie(
            "exampletunes.test", name="cf_clearance")], source="test")
        auth = SiteAuthResolver(
            vault_store=store, cookie_estate=estate).resolve(
                _toy_descriptor())
        assert auth.status == "missing"
        assert "/cookies import" in auth.owner_action

    def test_vault_session_resolves_and_lands_in_estate(
            self, tmp_path):
        store = _DictStore()
        assert store_site_session(
            store, "exampletunes",
            [_session_cookie("exampletunes.test", name="sess")])
        estate = _estate(tmp_path)
        resolver = SiteAuthResolver(
            vault_store=store, cookie_estate=estate)
        first = resolver.resolve(_toy_descriptor())
        assert first.ok and first.source == "vault"
        assert len(first.cookies) == 1
        # The session landed in the estate: the next run reuses it
        # without touching the vault at all (a store that explodes
        # on get proves the vault is never consulted).
        second = SiteAuthResolver(
            vault_store=_DictStore(raise_on_get=True),
            cookie_estate=estate).resolve(_toy_descriptor())
        assert second.ok and second.source == "estate"

    def test_require_resolves_session_for_public_site(
            self, tmp_path):
        # SoundCloud declares auth "none" (public tracks need no
        # account), but a login block says a session IS needed:
        # require=True resolves a real session instead of
        # answering ok empty — and reports missing when none
        # exists.  An anonymous retry after a login block is
        # never acceptable.
        store = _DictStore()
        estate = _estate(tmp_path)
        resolver = SiteAuthResolver(
            vault_store=store, cookie_estate=estate)
        assert resolver.resolve(
            SOUNDCLOUD_DESCRIPTOR, require=True).status == "missing"
        estate.upsert([_session_cookie("soundcloud.com")],
                      source="test")
        gets_before = store.gets
        auth = resolver.resolve(SOUNDCLOUD_DESCRIPTOR, require=True)
        assert auth.ok and auth.source == "estate"
        # The estate hit resolves without a vault touch (the one
        # earlier get belongs to the missing resolve above).
        assert store.gets == gets_before

    def test_missing_session_names_owner_action(self, tmp_path):
        auth = _resolver(tmp_path, _DictStore()).resolve(
            _toy_descriptor())
        assert auth.status == "missing"
        assert "exampletunes" in auth.owner_action
        assert "/cookies import" in auth.owner_action
        assert TEST_SESSION_VALUE not in auth.owner_action

    def test_expired_vault_session_reports_expired(
            self, tmp_path):
        store = _DictStore()
        assert store_site_session(
            store, "exampletunes",
            [_session_cookie("exampletunes.test", name="sess")],
            ttl_s=-60)
        auth = _resolver(tmp_path, store).resolve(_toy_descriptor())
        assert auth.status == "expired"
        assert "expired" in auth.owner_action
        assert "/cookies import" in auth.owner_action

    def test_locked_vault_reports_unavailable(self, tmp_path):
        auth = _resolver(tmp_path, _LockedStore()).resolve(
            _toy_descriptor())
        assert auth.status == "unavailable"
        assert "vault locked (test)" in auth.reason
        assert "vault" in auth.owner_action

    def test_store_site_session_rejects_empty_and_unusable(
            self, tmp_path):
        store = _DictStore()
        assert store_site_session(store, "x", []) is False
        assert store_site_session(
            store, "x", [{"domain": "", "name": ""}]) is False
        assert vault_key("x") not in store.entries
        assert store_site_session(
            store, "x", [_session_cookie("example.com")]) is True
        assert vault_key("x") in store.entries


# ── the up-front gate ───────────────────────────────────────────────


class TestUpfrontGate:
    def test_missing_credential_fails_login_without_anonymous_attempt(
            self, tmp_path, caplog):
        desc = _toy_descriptor()
        resolver = _resolver(tmp_path, _DictStore())
        dl = _downloader(
            tmp_path,
            site_descriptors=SiteDescriptorRegistry([desc]),
            site_auth=resolver)
        with mock.patch("nomorals.tools.media.download",
                        side_effect=AssertionError(
                            "anonymous chain ran")):
            report = dl.download(
                "https://exampletunes.test/track/1",
                dest_dir=tmp_path)
        assert report.ok is False
        assert report.block_reason == "login"
        assert report.failure is FailureKind.SITE_ERROR
        assert report.tried() == ["auth"]
        assert "signed-in account" in report.likely_cause
        assert "exampletunes" in report.likely_cause
        assert "/cookies import" in report.hint
        assert TEST_SESSION_VALUE not in report.summary()
        assert TEST_SESSION_VALUE not in caplog.text

    def test_expired_credential_fails_login(self, tmp_path):
        store = _DictStore()
        assert store_site_session(
            store, "exampletunes",
            [_session_cookie("exampletunes.test", name="sess")],
            ttl_s=-60)
        dl = _downloader(
            tmp_path,
            site_descriptors=SiteDescriptorRegistry(
                [_toy_descriptor()]),
            site_auth=_resolver(tmp_path, store))
        report = dl.download("https://exampletunes.test/track/1",
                             dest_dir=tmp_path)
        assert report.ok is False
        assert report.block_reason == "login"
        assert "expired" in report.hint

    def test_vault_unavailable_fails_login(self, tmp_path):
        dl = _downloader(
            tmp_path,
            site_descriptors=SiteDescriptorRegistry(
                [_toy_descriptor()]),
            site_auth=_resolver(tmp_path, _LockedStore()))
        report = dl.download("https://exampletunes.test/track/1",
                             dest_dir=tmp_path)
        assert report.ok is False
        assert report.block_reason == "login"
        assert "vault" in report.hint

    def test_authenticated_download_succeeds_and_persists_session(
            self, tmp_path, caplog):
        store = _DictStore()
        assert store_site_session(
            store, "exampletunes",
            [_session_cookie("exampletunes.test", name="sess")])
        estate = _estate(tmp_path)
        resolver = SiteAuthResolver(
            vault_store=store, cookie_estate=estate)
        dl = _downloader(
            tmp_path,
            site_descriptors=SiteDescriptorRegistry(
                [_toy_descriptor()]),
            site_auth=resolver)

        def fake_direct(url, dest_dir, **kw):
            target = Path(dest_dir) / "track.mp3"
            target.write_bytes(b"authed-bytes")
            return {"path": str(target), "bytes": len(b"authed-bytes"),
                    "title": "Authed Track", "extractor": "direct"}

        with caplog.at_level(logging.INFO), mock.patch(
                "nomorals.tools.media.download",
                side_effect=fake_direct):
            report = dl.download(
                "https://exampletunes.test/track/1",
                dest_dir=tmp_path)
        assert report.ok is True
        assert report.tried() == ["direct"]
        assert Path(report.path).read_bytes() == b"authed-bytes"
        # The vault session landed in the estate on resolve.
        held = estate.cookies_for("exampletunes.test")
        assert [c["name"] for c in held] == ["sess"]
        assert TEST_SESSION_VALUE not in caplog.text
        assert TEST_SESSION_VALUE not in report.summary()


# ── the login-block retry (SoundCloud proof) ────────────────────────


class TestLoginBlockRetry:
    def _login_connector(self):
        def blocked(url):
            raise SoundCloudError(
                "no playable stream: SoundCloud wants a signed-in "
                "account for this track", reason="login")

        return SimpleNamespace(stream_url=blocked)

    def test_login_block_retries_authenticated_and_lands(
            self, tmp_path, monkeypatch):
        monkeypatch.setenv("NM_HOME", str(tmp_path / "home"))
        store = _DictStore()
        assert store_site_session(
            store, "soundcloud",
            [_session_cookie("soundcloud.com")])
        resolver = SiteAuthResolver(vault_store=store)
        dl = _downloader(
            tmp_path,
            site_descriptors=SiteDescriptorRegistry(
                [SOUNDCLOUD_DESCRIPTOR]),
            site_auth=resolver, connector=self._login_connector())
        calls: list[dict[str, Any]] = []

        def fake_direct(url, dest_dir, **kw):
            calls.append(dict(kw))
            if len(calls) == 1:
                raise RuntimeError("connection reset by peer")
            target = Path(dest_dir) / "gated.mp3"
            target.write_bytes(b"gated-bytes")
            return {"path": str(target), "bytes": len(b"gated-bytes"),
                    "title": "Gated Track", "extractor": "direct"}

        with mock.patch("nomorals.tools.media.download",
                        side_effect=fake_direct):
            report = dl.download(SC_URL, dest_dir=tmp_path,
                                 audio_only=True)
        assert report.ok is True
        assert report.tried() == ["direct", "connector"]
        assert Path(report.path).read_bytes() == b"gated-bytes"
        # The retry ran authenticated: the direct stage received
        # the estate's cookies file, and the file carries the
        # session cookie's NAME for soundcloud.com.
        assert len(calls) == 2
        cookies_file = calls[1].get("cookies_file") or ""
        assert cookies_file
        text = Path(cookies_file).read_text(encoding="utf-8")
        assert "auth_token" in text
        assert "soundcloud.com" in text

    def test_login_block_without_session_defers_verdict(
            self, tmp_path, monkeypatch):
        # A login block with no session does NOT stop the chain:
        # a later rung (the rendered tab runs with the profile's
        # own cookies) may still serve the download — the B1–B3
        # rendered-capture behaviour survives.  Only when the
        # chain exhausts does the LOGIN verdict stand, with the
        # owner action.
        monkeypatch.setenv("NM_HOME", str(tmp_path / "home"))
        dl = _downloader(
            tmp_path,
            site_descriptors=SiteDescriptorRegistry(
                [SOUNDCLOUD_DESCRIPTOR]),
            site_auth=SiteAuthResolver(vault_store=_DictStore()),
            connector=self._login_connector())
        order: list[str] = []

        def _record(stage):
            def run(report, *a, **k):
                order.append(stage)
                report.stages.append(StageLog(stage=stage,
                                              note="mocked"))
                return False
            return run

        with mock.patch("nomorals.tools.media.download",
                        side_effect=_fail_direct), mock.patch.object(
                            dl, "_stage_browser",
                            side_effect=_record("browser")), \
                mock.patch.object(
                    dl, "_stage_rendered",
                    side_effect=_record("rendered")), \
                mock.patch.object(
                    dl, "_stage_proxy",
                    side_effect=_record("proxy")):
            report = dl.download(SC_URL, dest_dir=tmp_path,
                                 audio_only=True)
        assert report.ok is False
        assert report.block_reason == "login"
        assert report.tried() == [
            "direct", "connector", "auth", "proxy", "rendered",
            "browser"]
        assert order == ["proxy", "rendered", "browser"]
        assert "signed-in account" in report.likely_cause
        assert "/cookies import" in report.hint

    def test_non_login_block_still_falls_through(
            self, tmp_path):
        def blocked(url):
            raise SoundCloudError(
                "no playable stream: SoundCloud blocks this track "
                "in this region", reason="region")

        dl = _downloader(
            tmp_path,
            site_descriptors=SiteDescriptorRegistry(
                [SOUNDCLOUD_DESCRIPTOR]),
            site_auth=SiteAuthResolver(vault_store=_DictStore()),
            connector=SimpleNamespace(stream_url=blocked))
        order: list[str] = []

        def browser(report, *a, **k):
            order.append("browser")
            report.stages.append(StageLog(stage="browser",
                                          note="mocked"))
            return False

        with mock.patch("nomorals.tools.media.download",
                        side_effect=_fail_direct), mock.patch.object(
                            dl, "_stage_browser",
                            side_effect=browser):
            report = dl.download(SC_URL, dest_dir=tmp_path,
                                 audio_only=True)
        assert report.ok is False
        assert report.block_reason == ""
        assert order == ["browser"]
        assert report.tried()[-1] == "browser"


# ── descriptor auth fields ──────────────────────────────────────────


class TestDescriptorAuthFields:
    def test_auth_without_domains_fails_at_load(self):
        with pytest.raises(SiteDescriptorError) as exc:
            SiteDescriptor(
                site="x", url_patterns=(r"example\.com",),
                chain=("direct",), auth="vault")
        assert "auth_domains" in str(exc.value)
        assert "'x'" in str(exc.value)

    def test_empty_cookie_name_fails_at_load(self):
        with pytest.raises(SiteDescriptorError) as exc:
            SiteDescriptor(
                site="x", url_patterns=(r"example\.com",),
                chain=("direct",), auth="vault",
                auth_domains=("example.com",),
                auth_cookie_names=("",))
        assert "auth_cookie_names" in str(exc.value)

    def test_new_fields_round_trip(self):
        desc = _toy_descriptor()
        assert SiteDescriptor.from_dict(desc.to_dict()) == desc

    def test_soundcloud_descriptor_carries_session_shape(self):
        assert SOUNDCLOUD_DESCRIPTOR.auth == "none"
        assert SOUNDCLOUD_DESCRIPTOR.auth_domains == (
            "soundcloud.com",)
        assert SOUNDCLOUD_DESCRIPTOR.auth_cookie_names == (
            "auth_token",)
