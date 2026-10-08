"""Tests for build-map #93: Nigerian rental scam-detection engine.

All offline.  No network, no browser, no Seer — injectable seams only.
"""

import os
import tempfile

import pytest

from nomorals.property.scam import (
    AREA_NORMS,
    ILLEGAL_FEE_TERMS,
    ScamStore,
    check_listing,
    control_scamcheck,
)


@pytest.fixture()
def store():
    path = os.path.join(tempfile.mkdtemp(), "scam.db")
    return ScamStore(db_path=path)


# ── price anomaly ─────────────────────────────────────────────────

def test_price_way_below_norm_flags():
    text = ("2 bedroom flat for rent in Lekki Phase 1, ₦1,200,000 per annum. "
            "Address: 12 Admiralty Way, Lekki.")
    r = check_listing(text)
    assert r.area == "lekki"
    codes = [f.code for f in r.flags]
    assert "price_anomaly" in codes
    assert r.score <= 70
    assert r.verdict != "looks clean"


def test_price_at_norm_is_clean():
    text = ("2 bedroom flat for rent in Yaba, ₦2,200,000 per annum. "
            "Address: 45 Herbert Macaulay Way, Yaba.")
    r = check_listing(text)
    assert not any(f.code == "price_anomaly" for f in r.flags)


def test_price_slightly_below_is_info_only():
    # 12% below norm → info, not danger
    text = ("2 bedroom flat in Yaba, ₦1,950,000 per annum. "
            "Address: 45 Herbert Macaulay Way, Yaba.")
    r = check_listing(text)
    flags = [f for f in r.flags if f.code == "price_anomaly"]
    assert flags and flags[0].severity == "info"


# ── duplicate detection ───────────────────────────────────────────

def test_duplicate_photos_flagged(store):
    photo = b"fake-photo-bytes-1"
    text = "2 bedroom flat in Ikeja, ₦3,000,000. Address: 7 Allen Avenue, Ikeja."
    r1 = check_listing(text, store=store, photo_bytes=photo)
    assert not any(f.code == "duplicate" for f in r1.flags)
    r2 = check_listing(text, store=store, photo_bytes=photo)
    assert any(f.code == "duplicate" for f in r2.flags)


def test_duplicate_same_address_different_photos(store):
    t1 = "2 bedroom flat in Ikeja, ₦3,000,000. Address: 7 Allen Avenue, Ikeja."
    t2 = "2 bedroom flat in Ikeja, ₦2,900,000. Address: 7 Allen Avenue, Ikeja."
    check_listing(t1, store=store, photo_bytes=b"photo-a")
    r2 = check_listing(t2, store=store, photo_bytes=b"photo-b")
    assert any(f.code == "duplicate" for f in r2.flags)


# ── reverse image ─────────────────────────────────────────────────

def test_reverse_image_hook_flags():
    analyzer = lambda b: "this looks like a stock photo from another listing"
    r = check_listing("flat in Lekki, ₦4,500,000",
                      photo_bytes=b"img", photo_analyzer=analyzer)
    assert any(f.code == "reverse_image" for f in r.flags)


def test_reverse_image_clean_passes():
    analyzer = lambda b: "a real photo of a bedroom, looks genuine"
    r = check_listing("flat in Lekki, ₦4,500,000",
                      photo_bytes=b"img", photo_analyzer=analyzer)
    assert not any(f.code == "reverse_image" for f in r.flags)


def test_reverse_image_no_analyzer_no_flag():
    r = check_listing("flat in Lekki, ₦4,500,000", photo_bytes=b"img")
    assert not any(f.code == "reverse_image" for f in r.flags)


# ── fee language ──────────────────────────────────────────────────

@pytest.mark.parametrize("term", ILLEGAL_FEE_TERMS[:4])
def test_illegal_fee_terms_flagged(term):
    r = check_listing(f"2 bedroom in Yaba, ₦2,200,000. {term} of ₦50,000 applies.")
    flags = [f for f in r.flags if f.code == "illegal_fee"]
    assert flags, term
    assert "LASRERA" in flags[0].message


def test_fee_flag_language_is_factual():
    r = check_listing("flat in Yaba, ₦2,200,000. inspection fee ₦30,000 required.")
    f = next(x for x in r.flags if x.code == "illegal_fee")
    # factual, never moralizing
    assert "LASRERA declared this illegal" in f.message
    for bad in ("scammer", "fraudster", "criminal", "thief", "evil"):
        assert bad not in f.message.lower()


# ── address test ──────────────────────────────────────────────────

def test_landmark_only_address_flagged():
    r = check_listing("2 bedroom flat near Shoprite, opposite the market. "
                      "₦2,200,000 in Yaba.")
    assert any(f.code == "vague_address" for f in r.flags)


def test_full_address_passes():
    r = check_listing("2 bedroom flat, 45 Herbert Macaulay Way, Yaba. ₦2,200,000.")
    assert not any(f.code == "vague_address" for f in r.flags)


# ── payment channel ───────────────────────────────────────────────

def test_personal_account_plus_urgency_is_hard_stop():
    r = check_listing("2 bedroom in Lekki ₦4,500,000. Pay to my personal "
                      "account now, urgent, first come first serve!")
    flags = [f for f in r.flags if f.code == "payment_channel"]
    assert flags and flags[0].severity == "hard_stop"
    assert r.score == 0
    assert "do not pay" in r.verdict.lower()


def test_personal_account_alone_is_warn():
    r = check_listing("2 bedroom in Lekki ₦4,500,000. Pay to my personal account.")
    flags = [f for f in r.flags if f.code == "payment_channel"]
    assert flags and flags[0].severity == "warn"


# ── score / verdict ───────────────────────────────────────────────

def test_clean_listing_scores_high():
    r = check_listing("2 bedroom flat, 45 Herbert Macaulay Way, Yaba. "
                      "₦2,200,000 per annum. Agency: XYZ Realty Ltd.")
    assert r.score >= 70
    assert r.verdict == "looks clean"


def test_score_never_negative():
    r = check_listing("URGENT! 3 bedroom in Ikoyi ₦2,000,000! Pay to my personal "
                      "account ASAP! inspection fee ₦20,000. near the club.")
    assert r.score == 0


def test_empty_input():
    r = check_listing("")
    assert r.verdict == "nothing to check"


def test_never_raises_on_garbage():
    for bad in (None, 123, "x" * 5, "₦₦₦", "\x00" * 10):
        r = check_listing(bad)  # type: ignore[arg-type]
        assert isinstance(r.score, int)


# ── report format ─────────────────────────────────────────────────

def test_report_format_has_verdict_and_score():
    r = check_listing("2 bedroom flat, 45 Herbert Macaulay Way, Yaba. ₦2,200,000.")
    text = r.format()
    assert str(r.score) in text
    assert r.verdict in text
    assert "inspect in person" in text


# ── chat ──────────────────────────────────────────────────────────

def test_chat_empty_shows_usage():
    out = control_scamcheck("")
    assert "/scamcheck" in out


def test_chat_text_verdict():
    out = control_scamcheck("2 bedroom flat, 45 Herbert Macaulay Way, Yaba. ₦2,200,000.")
    assert "Scam check" in out


def test_chat_url_with_mock_fetcher():
    # control_scamcheck with a URL and no fetcher still works (checks the URL text)
    out = control_scamcheck("https://jiji.ng/lagos/flat-12345")
    assert "Scam check" in out


def test_chat_missing_photo_file():
    out = control_scamcheck("flat in Lekki photo:/no/such/file.jpg")
    assert "couldn't read photo" in out


def test_area_norms_seeded():
    for area in ("lekki", "yaba", "ikeja"):
        assert area in AREA_NORMS
        assert "2br" in AREA_NORMS[area]
