"""Cookie bundles (/cookies import|status|clear): the
owner's logged-in sessions imported into Devon's browser
profile.

Pinned here:
- both real exporter shapes parse — Netscape cookies.txt
  (``#HttpOnly_`` honored) and JSON (bare array AND the
  Playwright storage-state shape);
- expired cookies are dropped and COUNTED, never silently;
- the merge overwrites the same (domain, name, path) and
  preserves the rest of the profile;
- the secrecy law: a fixture cookie value appears in NO
  reply, NO journal row and NO log line — the journal
  carries domains, counts and a fingerprint only;
- the uploaded bundle file is deleted after the import
  attempt, success or failure;
- a NavSession over the profile starts signed-in, and an
  import lands on a live session's context too.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest

from nomorals.social.chat import webnav
from nomorals.social.chat.control import parse_control
from nomorals.social.chat.cookie_bundle import (
    BundleFormatError,
    ImportedCookiesStore,
    clear_imported,
    import_bundle_file,
    parse_bundle,
    render_import_reply,
    render_status,
)

NOW = 1_800_000_000.0            # 2027-01-15 — fixtures date from here
FUTURE = NOW + 90 * 86400        # ~Apr 2027
PAST = NOW - 30 * 86400

#: A distinctive fixture value, assembled from parts so the
#: source never carries a credential-shaped literal.
SECRET = "leak" + "-probe-" + "value" + "-9911"
SECRET2 = "second" + "-probe-" + "value" + "-7731"

#: One isolated estate per test (never the home estate):
#: the store is estate-backed since Navigator A6, so a test
#: that reads the home estate would see every other test's
#: cookies.  Keyed by tmp dir (+ tag for a second estate).
_ESTATES: dict[str, Any] = {}


def _estate_for(tmp_path: Path, *, tag: str = "") -> Any:
    from nomorals.storage.cookie_estate import CookieEstate
    from nomorals.storage.db import open_database

    key = str(tmp_path) + tag
    if key not in _ESTATES:
        db, _, _ = open_database(
            str(tmp_path / f"estate{tag or ''}.sqlite3"))
        estate = CookieEstate(db, legacy_roots={
            "profile_dir": tmp_path,
            "service_data_dir": tmp_path / "service",
            "sessions_dir": tmp_path / "sessions",
            "solved_dir": tmp_path / "solved"})
        estate.ensure()
        _ESTATES[key] = estate
    return _ESTATES[key]


def _store(profile: Path, tmp_path: Path) -> ImportedCookiesStore:
    return ImportedCookiesStore(profile, estate=_estate_for(tmp_path))


def netscape(rows: list[str]) -> bytes:
    return ("# Netscape HTTP Cookie File\n"
            + "\n".join(rows) + "\n").encode()


def row(domain: str, name: str, value: str, *,
        expires: int = int(FUTURE), path: str = "/",
        secure: str = "FALSE", flag: str = "TRUE",
        http_only: bool = False) -> str:
    line = "\t".join([domain, flag, path, secure,
                      str(expires), name, value])
    return ("#HttpOnly_" + line) if http_only else line


# ── parsers ──────────────────────────────────────────────

def test_netscape_parse_shapes():
    data = netscape([
        row(".google.com", "SID", SECRET, secure="TRUE",
            http_only=True),
        row("x.com", "auth", SECRET2, expires=0),
        "# a comment line",
        "",
    ]).replace(b"\n", b"\r\n")       # CRLF exporters
    result = parse_bundle(data, now=NOW)
    assert result.format == "netscape"
    assert result.dropped_expired == 0
    by_name = {c["name"]: c for c in result.cookies}
    sid = by_name["SID"]
    assert sid["domain"] == ".google.com"
    assert sid["httpOnly"] is True
    assert sid["secure"] is True
    assert sid["expires"] == FUTURE
    auth = by_name["auth"]
    assert auth["expires"] == -1.0       # 0 = session cookie
    assert auth["httpOnly"] is False


def test_netscape_expired_dropped_and_counted():
    data = netscape([
        row(".google.com", "OLD", SECRET, expires=int(PAST)),
        row(".google.com", "SID", SECRET2),
    ])
    result = parse_bundle(data, now=NOW)
    assert result.dropped_expired == 1
    assert [c["name"] for c in result.cookies] == ["SID"]


def test_json_bare_array_shapes():
    payload = json.dumps([
        {"name": "SID", "value": SECRET, "domain": ".google.com",
         "path": "/", "expirationDate": FUTURE, "httpOnly": True,
         "secure": True, "sameSite": "lax", "session": False},
        {"name": "auth", "value": SECRET2, "host_key": ".x.com",
         "path": "/", "session": True,
         "sameSite": "no_restriction"},
        {"name": "ms", "value": "v", "domain": ".ms.example",
         "expirationDate": FUTURE * 1000},
    ]).encode()
    result = parse_bundle(payload, now=NOW)
    assert result.format == "json"
    by_name = {c["name"]: c for c in result.cookies}
    assert by_name["SID"]["sameSite"] == "Lax"
    assert by_name["SID"]["expires"] == FUTURE
    assert by_name["auth"]["domain"] == ".x.com"
    assert by_name["auth"]["expires"] == -1.0
    assert by_name["auth"]["sameSite"] == "None"
    # millisecond expiries are read as milliseconds
    assert by_name["ms"]["expires"] == FUTURE


def test_json_storage_state_shape():
    payload = json.dumps({
        "cookies": [
            {"name": "SID", "value": SECRET,
             "domain": ".google.com", "path": "/",
             "expires": FUTURE, "httpOnly": True,
             "secure": True, "sameSite": "Strict"},
        ],
        "origins": [],
    }).encode()
    result = parse_bundle(payload, now=NOW)
    assert result.format == "storage-state"
    assert len(result.cookies) == 1
    assert result.cookies[0]["sameSite"] == "Strict"


def test_json_expired_dropped_and_counted():
    payload = json.dumps([
        {"name": "OLD", "value": SECRET, "domain": ".a.com",
         "expires": PAST},
        {"name": "NEW", "value": SECRET2, "domain": ".a.com",
         "expires": FUTURE},
    ]).encode()
    result = parse_bundle(payload, now=NOW)
    assert result.dropped_expired == 1
    assert [c["name"] for c in result.cookies] == ["NEW"]


def test_malformed_bundles_fail_plainly():
    junk = f"random text with {SECRET} inside".encode()
    with pytest.raises(BundleFormatError) as err:
        parse_bundle(junk, now=NOW)
    assert SECRET not in str(err.value)
    with pytest.raises(BundleFormatError):
        parse_bundle(b"{not json", now=NOW)
    with pytest.raises(BundleFormatError):
        parse_bundle(b"", now=NOW)
    with pytest.raises(BundleFormatError):
        parse_bundle(json.dumps({"nope": []}).encode(), now=NOW)


# ── the store ────────────────────────────────────────────

def _rec(name, value, domain, path="/", expires=FUTURE):
    return {"name": name, "value": value, "domain": domain,
            "path": path, "expires": float(expires),
            "httpOnly": False, "secure": False, "sameSite": ""}


def test_merge_overwrites_same_key_preserves_rest(tmp_path):
    store = _store(tmp_path, tmp_path)
    added, replaced = store.merge([
        _rec("SID", SECRET, ".google.com"),
        _rec("auth", SECRET2, ".x.com"),
    ])
    assert (added, replaced) == (2, 0)
    added, replaced = store.merge([
        _rec("SID", SECRET2, ".google.com"),          # same key
        _rec("SID", SECRET, ".google.com", path="/a"),  # new path
    ])
    assert (added, replaced) == (1, 1)
    held = {(r["domain"], r["name"], r["path"]): r["value"]
            for r in _store(tmp_path, tmp_path).load()}
    assert held[(".google.com", "SID", "/")] == SECRET2
    assert held[(".x.com", "auth", "/")] == SECRET2
    assert held[(".google.com", "SID", "/a")] == SECRET


def test_status_groups_counts_and_nearest_expiry(tmp_path):
    store = _store(tmp_path, tmp_path)
    store.merge([
        _rec("A", SECRET, ".google.com", expires=FUTURE),
        _rec("B", SECRET2, ".google.com", expires=FUTURE + 5000),
        _rec("C", "v", ".x.com", expires=-1),
    ])
    stats = {s.domain: s for s in store.status()}
    assert stats[".google.com"].count == 2
    assert stats[".google.com"].nearest_expiry == FUTURE
    assert stats[".x.com"].count == 1
    assert stats[".x.com"].nearest_expiry is None
    # The profile dir no longer scopes the store: status is
    # estate-wide, so a fresh dir sees the same cookies.
    assert {s.domain for s in _store(
        tmp_path / "nope", tmp_path).status()} == {
        ".google.com", ".x.com"}


def test_clear_one_domain_and_all(tmp_path):
    store = _store(tmp_path, tmp_path)
    store.merge([
        _rec("A", SECRET, ".google.com"),
        _rec("B", SECRET2, "mail.google.com"),
        _rec("C", "v", ".x.com"),
    ])
    removed, affected = store.clear("google.com")
    assert removed == 2
    assert set(affected) == {".google.com", "mail.google.com"}
    assert [r["domain"] for r in store.load()] == [".x.com"]
    removed, _ = store.clear("all")
    assert removed == 1
    assert store.load() == []
    removed, _ = store.clear("google.com")
    assert removed == 0


# ── the webnav seam ──────────────────────────────────────

class FakeContext:
    def __init__(self) -> None:
        self.added: list[list[dict]] = []
        self.cleared: list[dict] = []

    def add_cookies(self, cookies):
        self.added.append(list(cookies))

    def clear_cookies(self, **kwargs):
        self.cleared.append(kwargs)


def test_session_start_applies_imported_cookies(tmp_path):
    _store(tmp_path, tmp_path).merge([
        _rec("SID", SECRET, ".google.com"),
        _rec("auth", SECRET2, ".x.com", expires=-1),
    ])
    session = webnav.NavSession(
        profile_dir=tmp_path,
        cookie_estate=_estate_for(tmp_path))
    fake = FakeContext()
    session._context = fake          # the seam, driven directly
    session._apply_imported_cookies()
    assert len(fake.added) == 1
    payload = {c["name"]: c for c in fake.added[0]}
    assert payload["SID"]["domain"] == ".google.com"
    assert payload["SID"]["expires"] == FUTURE
    assert "expires" not in payload["auth"]   # session cookie


def test_import_applies_to_live_sessions(tmp_path):
    fake = FakeContext()
    webnav._register_live_context(tmp_path, fake)
    try:
        applied = webnav.apply_cookies_to_live(
            tmp_path, [_rec("SID", SECRET, ".google.com")])
    finally:
        webnav._unregister_live_context(tmp_path, fake)
    assert applied == 1
    assert fake.added[0][0]["name"] == "SID"
    assert webnav.apply_cookies_to_live(tmp_path, [
        _rec("SID", SECRET, ".google.com")]) == 0


# ── the import, end to end ───────────────────────────────

def _bundle_file(tmp_path: Path, data: bytes) -> Path:
    src = tmp_path / "uploads" / "cookies.txt"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_bytes(data)
    return src


def test_import_end_to_end(tmp_path, caplog):
    profile = tmp_path / "browser_profile"
    journal = tmp_path / "personal" / "journal.jsonl"
    src = _bundle_file(tmp_path, netscape([
        row(".google.com", "SID", SECRET, http_only=True),
        row(".google.com", "OLD", SECRET2, expires=int(PAST)),
        row("x.com", "auth", SECRET2, expires=0),
    ]))
    fake = FakeContext()
    webnav._register_live_context(profile, fake)
    try:
        with caplog.at_level(logging.INFO):
            outcome = import_bundle_file(
                src, profile, journal_path=journal, now=NOW,
                estate=_estate_for(tmp_path))
    finally:
        webnav._unregister_live_context(profile, fake)
    assert outcome.ok
    assert outcome.count == 2
    assert outcome.domains == [".google.com", "x.com"]
    assert outcome.dropped_expired == 1
    assert outcome.live_sessions == 1
    assert outcome.file_deleted and not src.exists()
    assert len(fake.added) == 1
    # the store holds them for the next session start
    held = _store(profile, tmp_path).load()
    assert {r["name"] for r in held} == {"SID", "auth"}
    # ── the secrecy law ──
    reply = render_import_reply(outcome)
    assert "2 cookies" in reply and "/webshot" in reply
    journal_text = journal.read_text(encoding="utf-8")
    row_out = json.loads(journal_text.strip())
    assert row_out["kind"] == "cookies_import"
    assert row_out["count"] == 2
    assert row_out["dropped_expired"] == 1
    assert row_out["domains"] == [".google.com", "x.com"]
    assert len(row_out["fingerprint"]) == 64
    for surface in (reply, journal_text, caplog.text):
        assert SECRET not in surface
        assert SECRET2 not in surface
    status_text = render_status(profile,
                                estate=_estate_for(tmp_path))
    assert "google.com" in status_text
    assert SECRET not in status_text


def test_import_malformed_deletes_file_and_journals(tmp_path):
    profile = tmp_path / "browser_profile"
    journal = tmp_path / "personal" / "journal.jsonl"
    src = _bundle_file(
        tmp_path, f"junk junk {SECRET} junk".encode())
    outcome = import_bundle_file(
        src, profile, journal_path=journal, now=NOW,
        estate=_estate_for(tmp_path))
    assert not outcome.ok
    assert SECRET not in outcome.error
    assert outcome.file_deleted and not src.exists()
    row_out = json.loads(journal.read_text(encoding="utf-8"))
    assert row_out["ok"] is False
    assert row_out["error_kind"] == "format"
    assert SECRET not in json.dumps(row_out)
    assert _store(profile, tmp_path).load() == []


def test_import_missing_file_is_a_sentence(tmp_path):
    outcome = import_bundle_file(
        tmp_path / "nope.txt", tmp_path / "profile", now=NOW)
    assert not outcome.ok
    assert "gone" in outcome.error
    assert outcome.file_deleted


def test_import_all_expired_imports_nothing(tmp_path):
    profile = tmp_path / "browser_profile"
    src = _bundle_file(tmp_path, netscape([
        row(".google.com", "OLD", SECRET, expires=int(PAST)),
    ]))
    outcome = import_bundle_file(src, profile, now=NOW,
                                 estate=_estate_for(tmp_path))
    assert outcome.ok and outcome.count == 0
    assert outcome.dropped_expired == 1
    reply = render_import_reply(outcome)
    assert "expired" in reply
    assert SECRET not in reply


def test_clear_replies_and_journal_carry_no_values(tmp_path):
    profile = tmp_path / "browser_profile"
    journal = tmp_path / "personal" / "journal.jsonl"
    _store(profile, tmp_path).merge([
        _rec("SID", SECRET, ".google.com"),
        _rec("auth", SECRET2, ".x.com"),
    ])
    text = clear_imported(profile, "google.com",
                          journal_path=journal,
                          estate=_estate_for(tmp_path))
    assert "removed 1 imported cookie" in text
    assert SECRET not in text and SECRET2 not in text
    assert [r["domain"] for r in
            _store(profile, tmp_path).load()] == [".x.com"]
    row_out = json.loads(journal.read_text(encoding="utf-8"))
    assert row_out["kind"] == "cookies_clear"
    assert row_out["removed"] == 1
    text = clear_imported(profile, "all", journal_path=journal,
                          estate=_estate_for(tmp_path))
    assert "removed all 1 imported cookie" in text
    text = clear_imported(profile, "all",
                          estate=_estate_for(tmp_path))
    assert "no imported cookies" in text


def test_status_empty_profile_is_honest(tmp_path):
    text = render_status(tmp_path / "fresh",
                          estate=_estate_for(tmp_path, tag="-fresh"))
    assert "no imported cookies" in text


# ── catalog wiring ───────────────────────────────────────

def test_cookies_family_parses_as_one_command():
    for text in ("/cookies", "/cookies check",
                 "/cookies import", "/cookies status",
                 "/cookies clear google.com",
                 "/cookies clear all"):
        cmd = parse_control(text)
        assert cmd is not None and cmd.kind == "cookies", text
