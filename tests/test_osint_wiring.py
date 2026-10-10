"""Tests for OSINT/temp-mail/security wiring (offline-safe)."""

import io
import tempfile
import os


def test_shape_routing():
    from nomorals.search.osint import (
        _looks_like_username, _looks_like_email,
        _looks_like_domain, _looks_like_ip)
    assert _looks_like_username("@testuser123") == "testuser123"
    assert _looks_like_username("hello world") is None
    assert _looks_like_email("a@b.com") == "a@b.com"
    assert _looks_like_email("not-an-email") is None
    assert _looks_like_domain("https://example.com/x") == "example.com"
    assert _looks_like_ip("8.8.8.8") == "8.8.8.8"
    assert _looks_like_ip("999.1.1.1") is None


def test_osint_specs_registered():
    from nomorals.search.sources import SOURCE_SPECS
    names = [s[0] for s in SOURCE_SPECS]
    for expected in ("osint_username", "osint_email", "osint_domain", "osint_ip"):
        assert expected in names, f"{expected} not in federated sources"


def test_temp_mail_providers_registered():
    from nomorals.accounts.temp_mail import PROVIDERS, CASCADE_PROVIDERS
    assert "1secmail" in PROVIDERS
    assert "guerrillamail" in PROVIDERS
    assert CASCADE_PROVIDERS[0] == "1secmail"


def test_mail_message_code_extraction():
    from nomorals.accounts.temp_mail import MailMessage
    m = MailMessage(subject="Your code", body_text="Your verification code is 482910. Expires soon.")
    assert m.code == "482910"
    m2 = MailMessage(subject="hi", body_text="no code here at all")
    assert m2.code == ""


def test_exif_strip_roundtrip():
    from PIL import Image
    from nomorals.security.exif import strip_exif_to_bytes, exif_summary
    img = Image.new("RGB", (64, 64), color="blue")
    exif = img.getexif()
    exif[271] = "TestCamera"
    exif[34853] = {0: b"\x00"}  # fake GPS tag
    tmp = tempfile.mktemp(suffix=".jpg")
    try:
        img.save(tmp, exif=exif)
        before = exif_summary(tmp)
        assert before["has_exif"] is True
        clean = strip_exif_to_bytes(open(tmp, "rb").read())
        open(tmp, "wb").write(clean)
        after = exif_summary(tmp)
        assert after["has_exif"] is False
        assert after["has_gps"] is False
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def test_dnsleak_report_shape():
    from nomorals.security.dnsleak import DnsLeakReport
    r = DnsLeakReport()
    d = r.to_dict()
    assert "leak" in d and "resolvers" in d and "method" in d


def test_tool_modules_import():
    import nomorals.tools.accounts
    import nomorals.tools.security
    assert hasattr(nomorals.tools.accounts, "register")
    assert hasattr(nomorals.tools.security, "register")


def test_creator_temp_email_methods():
    from nomorals.accounts.creator import AccountCreator
    assert hasattr(AccountCreator, "get_temp_email")
    assert hasattr(AccountCreator, "get_temp_email_cascade")
    assert hasattr(AccountCreator, "poll_email_code")
