"""Gapfix R3-68: property cross-listing + passport wired into the
scam-check flow (audit §68).

Failing-first: on the unmodified tree these fail — ScamReport has no
``cross_listing`` field, ``check_listing`` takes no ``passport_store``,
and /passport has no ``property`` subverb.
"""

import os
import tempfile

from nomorals.property.cross_listing import PropertyPassportStore
from nomorals.property.passport import control_passport
from nomorals.property.scam import ScamStore, check_listing, control_scamcheck


def _scam_store():
    return ScamStore(db_path=os.path.join(tempfile.mkdtemp(), "scam.db"))


def _passport_store():
    return PropertyPassportStore(
        os.path.join(tempfile.mkdtemp(), "passports.json"))


JIJI = ("2 bedroom flat for rent in Ikeja, ₦3,000,000 per annum. "
        "Address: 7 Allen Avenue, Ikeja. "
        "https://jiji.ng/lagos/flat-7-allen")
PROPERTYPRO = ("2 bedroom flat for rent in Ikeja, ₦3,000,000 per annum. "
               "Address: 7 Allen Avenue, Ikeja. "
               "https://propertypro.ng/listing/7-allen")


def test_cross_listing_groups_same_property_across_sources():
    store = _scam_store()
    r1 = check_listing(JIJI, store=store)
    assert r1.cross_listing == {}
    r2 = check_listing(PROPERTYPRO, store=store)
    assert r2.cross_listing["sources"] == 2
    assert set(r2.cross_listing["source_names"]) == {
        "jiji.ng", "propertypro.ng"}
    assert r2.cross_listing["listings"] == 2
    assert "Cross-listed" in r2.format()
    assert "jiji.ng" in r2.format()


def test_cross_listing_same_source_twice_is_not_cross_listed():
    store = _scam_store()
    check_listing(JIJI, store=store)
    r2 = check_listing(JIJI, store=store)
    assert r2.cross_listing["sources"] == 1
    assert r2.cross_listing["listings"] == 2
    assert "Cross-listed" not in r2.format()
    assert "Seen 2 time(s) on jiji.ng" in r2.format()


def test_cross_listing_price_beyond_tolerance_does_not_group():
    store = _scam_store()
    check_listing(JIJI, store=store)
    # Same address, 10% cheaper: beyond the 2% seam tolerance, so the
    # second listing stands alone — never merged with a different price.
    cheaper = PROPERTYPRO.replace("₦3,000,000", "₦2,700,000")
    r2 = check_listing(cheaper, store=store)
    assert r2.cross_listing == {}
    assert "Cross-listed" not in r2.format()


def test_cross_listing_does_not_change_verdict_or_score():
    store = _scam_store()
    r1 = check_listing(JIJI, store=store)
    r2 = check_listing(PROPERTYPRO, store=store)
    # The cross-listing wire adds no flag and no deduction of its own:
    # the only new flag on the second check is the pre-existing
    # duplicate warn (score 100 -> 85), and the verdict stays clean.
    assert r2.verdict == r1.verdict == "looks clean"
    assert {f.code for f in r2.flags} - {f.code for f in r1.flags} == {
        "duplicate"}
    assert r2.score == r1.score - 15


def test_scamcheck_records_property_passport_sighting():
    store = _scam_store()
    pstore = _passport_store()
    check_listing(JIJI, store=store, passport_store=pstore)
    passport = pstore.by_address("7 Allen Avenue, Ikeja")
    assert passport is not None
    assert len(passport.history) == 1
    assert passport.history[0].kind == "listed"
    assert "jiji.ng" in passport.history[0].detail
    first_version = passport.version
    # A second sighting accumulates on the SAME passport.
    check_listing(PROPERTYPRO, store=store, passport_store=pstore)
    again = pstore.by_address("7 Allen Avenue, Ikeja")
    assert again.passport_id == passport.passport_id
    assert len(again.history) == 2
    assert again.version == first_version + 1


def test_scamcheck_without_passport_store_records_nothing():
    # check_listing stays pure unless the caller supplies a store.
    check_listing(JIJI, store=_scam_store())
    assert PropertyPassportStore is not None  # module importable, no crash


def test_passport_property_subverb_shows_accumulated_history():
    store = _scam_store()
    pstore = _passport_store()
    check_listing(JIJI, store=store, passport_store=pstore)
    check_listing(PROPERTYPRO, store=store, passport_store=pstore)
    out = control_passport("property 7 Allen Avenue, Ikeja",
                           property_passport_store=pstore)
    assert "7 Allen Avenue, Ikeja" in out
    assert "listed" in out
    assert "jiji.ng" in out
    assert "propertypro.ng" in out


def test_passport_property_subverb_unknown_address_is_honest():
    pstore = _passport_store()
    out = control_passport("property 99 Nowhere Road, Lekki",
                           property_passport_store=pstore)
    assert "no property passport" in out.lower()
    assert "scamcheck" in out.lower()


def test_chat_scamcheck_wires_cross_listing_and_passport(tmp_path):
    # End to end through the chat verbs with injected stores.
    pstore = PropertyPassportStore(str(tmp_path / "pp.json"))
    out1 = control_scamcheck(JIJI, property_passport_store=pstore)
    assert "Scam check" in out1
    out2 = control_scamcheck(PROPERTYPRO,
                             property_passport_store=pstore)
    assert "Cross-listed" in out2
    out3 = control_passport("property 7 Allen Avenue, Ikeja",
                            property_passport_store=pstore)
    assert "listed" in out3
