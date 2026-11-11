"""Mock screens Phase 5 (FakeDetail port, census §29):
bulk test-data generation — the seeded generator of
obviously synthetic personal / identity test data, the
bulk screen builders, the text / JSON / CSV / XML
exports, and the /mock demo bulk-* + /mock data forms.

Network-free: renders are local Pillow work, the
delivery seam is a recording stub — the phase-2b
suite's idiom.
"""
from __future__ import annotations

import csv
import io
import json
import re
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from nomorals.accounts.identity_bank import (
    DISPOSABLE_SURNAMES,
)
from nomorals.agents.partner.runtime_mock_screens import (
    RuntimeMockScreensMixin,
)
from nomorals.social.chat.mock_screens import (
    BulkGenerator,
    DemoDataError,
    MockScreensError,
    MockScreenService,
)
from nomorals.social.chat.mock_screens.authoring import (
    MAX_DATA_TEXT,
    demo_list_screen,
    demo_screen,
    parse_mock_tail,
)
from nomorals.social.chat.mock_screens.demo_data import (
    COUNTRIES,
    CREATIVE_FAMILIES,
    MAIL_PROVIDERS,
    MAX_BULK_DATA,
    MAX_BULK_SCREEN_ROWS,
    REFERENCE_DATE,
    bulk_call_history,
    bulk_chat_screen,
    bulk_list_screen,
    bulk_status_views,
    export_records,
)
from nomorals.social.chat.mock_screens.render import (
    render_screen,
)
from nomorals.social.chat.mock_screens.render_screens import (
    render_call_history,
    render_list_screen,
    render_status_views,
)


def _png_bytes(color=(30, 120, 200), size=(64, 64)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


class _Send:
    def __init__(self, ok=True):
        self.calls = []
        self.ok = ok

    def __call__(self, ref, caption=""):
        self.calls.append((ref, caption))
        return SimpleNamespace(ok=self.ok, error="")


# ── census accounting (census §29 build list) ────────────

#: every Phase 5 surface the census names, mapped to
#: the organ piece that carries it
CENSUS_RENDERED = {
    "names": "BulkGenerator.name",
    "usernames": "BulkGenerator.username",
    "passwords": "BulkGenerator.password",
    "multi-country phones": "BulkGenerator.phone",
    "multi-country addresses": "BulkGenerator.address",
    "email-shaped strings": "BulkGenerator.email",
    "company names": "BulkGenerator.company",
    "IMEI-shaped test values": "BulkGenerator.imei",
    "creative-name families": "BulkGenerator.creative_name",
    "user faces/avatars": "BulkGenerator.face (honest "
                          "unavailable — initials fallback)",
    "bulk chat list": "bulk_list_screen",
    "bulk call history": "bulk_call_history",
    "bulk chat": "bulk_chat_screen",
    "bulk status views": "bulk_status_views",
    "text export": "export_records(text)",
    "JSON export": "export_records(json)",
    "CSV export": "export_records(csv)",
    "XML export": "export_records(xml)",
}

#: census-named scope held back, with its census reason —
#: never filled by guessing
CENSUS_REMAINDERS = {
    "Generated user faces":
        "census §17.5: no face-generation backend exists "
        "and none is invented here — face() reports "
        "unavailable and the initials avatar stands in",
    "FakeDetail's exact creative-name word lists":
        "census §8: no per-generator controls were "
        "established — the pools are Devon's own, named "
        "as such, never claimed to be FakeDetail's",
    "100+ phone country formats":
        "census §7.4 claims 100+ formats; this build "
        "ships six with stated fictional blocks — the "
        "rest are an extension, not a guess",
    "Receipts family":
        "owner-dropped 2026-11-11 — never built in any "
        "form, and no demo-data kind feeds it",
}


def test_census_phase5_catalogue_is_accounted():
    assert set(CENSUS_RENDERED) == {
        "names", "usernames", "passwords",
        "multi-country phones", "multi-country addresses",
        "email-shaped strings", "company names",
        "IMEI-shaped test values", "creative-name families",
        "user faces/avatars", "bulk chat list",
        "bulk call history", "bulk chat",
        "bulk status views", "text export", "JSON export",
        "CSV export", "XML export"}
    assert CENSUS_REMAINDERS["Receipts family"].startswith(
        "owner-dropped")


# ── seeded reproducibility ───────────────────────────────

def test_same_seed_reproduces_identical_records():
    first = BulkGenerator(7).identities(10)
    second = BulkGenerator(7).identities(10)
    assert first == second
    assert BulkGenerator(8).identities(10) != first


def test_same_seed_reproduces_identical_screens():
    one = bulk_list_screen(count=25, seed=7)
    two = bulk_list_screen(count=25, seed=7)
    assert one.to_dict() == two.to_dict()
    assert bulk_list_screen(count=25, seed=8).to_dict() \
        != one.to_dict()


def test_bulk_renders_are_deterministic():
    screen = bulk_list_screen(count=25, seed=7)
    one, _ = render_list_screen(screen)
    two, _ = render_list_screen(screen)
    assert one == two


# ── synthetic markers ────────────────────────────────────

def test_every_person_is_obviously_synthetic():
    gen = BulkGenerator(3)
    for record in gen.identities(20):
        assert record["synthetic"] is True
        assert record["kind"] == "demo-test-data"
        assert "service" not in record
        assert type(record) is dict
        surname = record["name"].split(" ", 1)[1]
        assert surname.split(" ")[0] in DISPOSABLE_SURNAMES
        assert record["first_name"] in record["name"]
        assert record["receives_mail"] is False


def test_bulk_data_never_mints_a_trial_identity():
    gen = BulkGenerator(3)
    assert not hasattr(gen, "mint")
    assert not hasattr(gen, "store")
    record = gen.identity()
    assert "service" not in record
    assert "disposable" not in record


# ── field shapes ─────────────────────────────────────────

def test_phone_formats_are_fictional_by_range():
    gen = BulkGenerator(11)
    us = gen.phone("us")["number"]
    assert re.fullmatch(r"\+1 \(555\) 01\d\d", us), us
    gb = gen.phone("gb")["number"]
    assert re.fullmatch(r"\+44 7700 900\d{3}", gb), gb
    ng = gen.phone("ng")["number"]
    assert re.fullmatch(r"\+234 \d{3} 000 \d{4}", ng), ng
    for country in COUNTRIES:
        made = gen.phone(country)
        assert made["fictional"] is True
        assert "not attached" in made["note"]
    with pytest.raises(DemoDataError):
        gen.phone("xx")


def test_email_strings_cannot_receive_mail():
    gen = BulkGenerator(5)
    for provider in MAIL_PROVIDERS:
        made = gen.email(provider)
        assert made["email"].endswith(
            "@" + provider + ".example")
        assert made["receives_mail"] is False
        assert "cannot receive mail" in made["note"]
    with pytest.raises(DemoDataError):
        gen.email("realmail")


def test_imei_is_luhn_valid_and_test_tac():
    gen = BulkGenerator(9)
    for _ in range(20):
        made = gen.imei()
        value = made["imei"]
        assert re.fullmatch(r"\d{15}", value)
        assert value.startswith("35000000")
        assert made["format_valid"] is True
        assert made["assigned"] is False
        total = 0
        for pos, char in enumerate(reversed(value)):
            digit = int(char)
            if pos % 2 == 1:
                digit *= 2
                if digit > 9:
                    digit -= 9
            total += digit
        assert total % 10 == 0


def test_dob_is_an_adult_age_against_the_fixed_date():
    gen = BulkGenerator(4)
    for _ in range(20):
        born = date.fromisoformat(gen.dob())
        age_days = (REFERENCE_DATE - born).days
        assert 21 * 365 <= age_days <= 45 * 365


def test_password_and_username_follow_the_creator_shape():
    gen = BulkGenerator(2)
    made = gen.password()
    assert len(made) == 16
    assert re.fullmatch(r"[A-Za-z0-9!@#$%^&*()\-_=+]{16}",
                        made)
    assert len(gen.password(24, symbols=False)) == 24
    assert re.fullmatch(r"demo_[a-z0-9]{8}",
                        gen.username())
    with pytest.raises(DemoDataError):
        gen.password(2)


def test_addresses_are_synthetic_streets_real_shapes():
    gen = BulkGenerator(6)
    for country in COUNTRIES:
        made = gen.address(country)
        assert made["synthetic"] is True
        assert made["country"] == country
        assert any(word in made["street"] for word in (
            "Demo", "Test", "Sample", "Mock", "Trial",
            "Placeholder", "Example", "Sandbox"))
        assert made["city"] and made["state"]
        assert made["line"].endswith(
            ", " + country.upper())
    with pytest.raises(DemoDataError):
        gen.address("xx")


def test_company_and_bio_are_demo_shaped():
    gen = BulkGenerator(6)
    made = gen.company("us")
    assert made.split(" ")[-1] in (
        "LLC", "Inc", "Co", "Studios")
    assert "Demo" in gen.bio() or "Test" in gen.bio() \
        or "Mock" in gen.bio() or "Placeholder" in gen.bio() \
        or "Sample" in gen.bio()


def test_face_is_honestly_unavailable():
    gen = BulkGenerator(1)
    face = gen.face()
    assert face["available"] is False
    assert "no face-generation backend" in face["reason"]
    assert face["fallback"] == "initials avatar"
    person = gen.person()
    assert person["face"]["available"] is False
    assert person["face"]["fallback"] == "initials avatar"


def test_creative_families_all_produce_names():
    gen = BulkGenerator(12)
    for family in CREATIVE_FAMILIES:
        made = gen.creative_name(family)
        assert isinstance(made, str) and made.strip()
    with pytest.raises(DemoDataError):
        gen.creative_name("not-a-family")


def test_unknown_locale_is_refused():
    with pytest.raises(DemoDataError):
        BulkGenerator(0, locale="xx")


# ── bulk screens ─────────────────────────────────────────

def test_bulk_chat_list_fifty_rows_render():
    screen = bulk_list_screen(count=50, seed=7)
    assert len(screen.rows) == 50
    assert len(screen.participants) == 51
    ids = [p.id for p in screen.participants]
    assert len(set(ids)) == len(ids)
    data, drawn = render_list_screen(screen)
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    # exactly one drawn element per row, in order
    assert [e["id"] for e in drawn["elements"]] == [
        "r" + str(i) for i in range(50)]


def test_bulk_list_times_descend_newest_first():
    screen = bulk_list_screen(count=50, seed=7)
    times = [row.time for row in screen.rows]
    assert times[0] == "now"
    assert times[1].endswith("m")
    assert times[9] == "Yesterday"
    assert any(t in ("Monday", "Tuesday", "Wednesday",
                     "Thursday", "Friday", "Saturday",
                     "Sunday") for t in times[10:16])


def test_bulk_list_flavors_and_platforms():
    calls = bulk_list_screen(count=10, seed=1,
                             flavor="calls")
    assert calls.flavor == "calls"
    assert all(row.preview_kind.startswith("call_")
               for row in calls.rows)
    status = bulk_list_screen(count=10, seed=1,
                              flavor="status")
    assert status.flavor == "status"
    assert all(row.section for row in status.rows)
    telegram = bulk_list_screen(platform="telegram",
                                count=10, seed=1)
    assert telegram.tabs
    assert telegram.tabs[0].label == "All Chats"
    instagram = bulk_list_screen(platform="instagram",
                                 count=10, seed=1)
    assert instagram.stories
    with pytest.raises(DemoDataError):
        bulk_list_screen(platform="myspace", count=5)
    with pytest.raises(DemoDataError):
        bulk_list_screen(count=5, flavor="nope")


def test_bulk_call_history_fifty_entries_render():
    screen = bulk_call_history(count=50, seed=7)
    assert len(screen.entries) == 50
    for entry in screen.entries:
        assert entry.type in ("incoming", "outgoing",
                              "missed")
        assert entry.count >= 1
        assert entry.label
    data, drawn = render_call_history(screen)
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    # exactly one drawn element per entry, in order
    assert [e["id"] for e in drawn["elements"]] == [
        "h" + str(i) for i in range(50)]


def test_bulk_chat_messages_build_over_a_clock():
    screen = bulk_chat_screen(count=30, seed=7)
    assert len(screen.messages) == 31  # + day divider
    assert screen.messages[0].kind == "divider"
    senders = {m.sender for m in screen.messages[1:]}
    assert senders <= {p.id for p in screen.participants}
    assert "me" in senders
    times = [m.time for m in screen.messages[1:]]
    assert times == sorted(times, key=_clock_minutes)
    data, _ = render_screen(screen)
    assert data[:8] == b"\x89PNG\r\n\x1a\n"


def _clock_minutes(text: str) -> int:
    parts = text.split(" ")
    hh, mm = parts[0].split(":")
    total = int(hh) * 60 + int(mm)
    if parts[1] == "PM" and int(hh) != 12:
        total += 12 * 60
    return total


def test_bulk_status_views_count_rows():
    screen = bulk_status_views(count=25, seed=7)
    assert len(screen.viewers) == 25
    data, drawn = render_status_views(screen)
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(drawn["elements"]) > 25


def test_bulk_screen_caps_name_the_reason():
    with pytest.raises(DemoDataError) as exc:
        bulk_list_screen(count=0)
    assert "at least 1" in str(exc.value)
    with pytest.raises(DemoDataError) as exc:
        bulk_list_screen(count=MAX_BULK_SCREEN_ROWS + 1)
    assert str(MAX_BULK_SCREEN_ROWS) in str(exc.value)
    assert "canvas" in str(exc.value)
    with pytest.raises(DemoDataError):
        BulkGenerator(0).identities(MAX_BULK_DATA + 1)


# ── exports round-trip ───────────────────────────────────

def test_exports_round_trip_through_standard_parsers():
    records = BulkGenerator(7).identities(10)
    parsed = json.loads(export_records(records, "json"))
    assert parsed == records
    rows = list(csv.DictReader(
        io.StringIO(export_records(records, "csv"))))
    assert len(rows) == 10
    assert rows[0]["name"] == records[0]["name"]
    root = ET.fromstring(export_records(records, "xml"))
    assert root.tag == "demo-data"
    assert len(root.findall("record")) == 10
    assert root.findall("record")[0].find(
        "name").text == records[0]["name"]
    text = export_records(records, "text")
    assert len(text.splitlines()) == 10
    assert records[0]["name"] in text.splitlines()[0]
    with pytest.raises(DemoDataError):
        export_records(records, "yaml")


# ── authoring + runtime ──────────────────────────────────

def test_parse_dispatch_modes_bulk_and_data():
    cases = {
        "demo bulk-list whatsapp 25 seed 7": "demo-list",
        "demo bulk-calls 10 seed 1": "demo-list",
        "demo bulk-status-list 10": "demo-list",
        "demo bulk-history 10 seed 3": "demo-history",
        "demo bulk-chat telegram 30 seed 1": "demo",
        "demo bulk-views 12 seed 2": "demo-views",
        "data identity 3 seed 1 json": "data",
        "data creative band 2 seed 4 text": "data",
    }
    for tail, mode in cases.items():
        _obj, got = parse_mock_tail(tail)
        assert got == mode, tail
    screen, _mode = parse_mock_tail(
        "demo bulk-list whatsapp 25 seed 7")
    assert len(screen.rows) == 25
    screen, _mode = parse_mock_tail("demo bulk-history 10")
    assert len(screen.entries) == 10
    text, _mode = parse_mock_tail("data identity 3 json")
    assert len(json.loads(text)) == 3
    text, _mode = parse_mock_tail("data phone 2 us csv")
    assert len(list(csv.DictReader(io.StringIO(text)))) == 2
    with pytest.raises(MockScreensError):
        parse_mock_tail("demo bulk-list whatsapp 201")
    with pytest.raises(MockScreensError):
        parse_mock_tail(
            "data identity " + str(MAX_DATA_TEXT + 1)
            + " json")
    with pytest.raises(MockScreensError):
        parse_mock_tail("data nosuchkind 3")
    with pytest.raises(MockScreensError):
        parse_mock_tail("data creative 3")
    with pytest.raises(MockScreensError):
        parse_mock_tail("demo bulk-list whatsapp nope")


class _Runtime(RuntimeMockScreensMixin):
    def __init__(self, tmp_path, operator=True):
        self._tmp = tmp_path
        self._operator = operator
        self.send = _Send()

    def _is_operator(self, message):
        return self._operator

    def _expression_root(self):
        return self._tmp

    def _makers_audience(self, message):
        return "personal"

    def _expression_send(self, message, ref, caption=""):
        return self.send(ref, caption)


def _message():
    return SimpleNamespace(
        chat=SimpleNamespace(kind="dm"), sender="owner")


@pytest.mark.parametrize("tail", [
    "demo bulk-list whatsapp 25 seed 7",
    "demo bulk-calls 10 seed 1",
    "demo bulk-history 10 seed 3",
    "demo bulk-chat telegram 30 seed 1",
    "demo bulk-views 12 seed 2",
])
def test_command_delivers_each_bulk_demo(tmp_path, tail):
    runtime = _Runtime(tmp_path)
    reply = runtime._control_mock("mock", tail, _message())
    assert reply == ""
    assert len(runtime.send.calls) == 1
    ref, caption = runtime.send.calls[0]
    assert ref.kind == "image"
    assert Path(ref.path).exists()
    assert "mock screen" in caption


@pytest.mark.parametrize("tail, check", [
    ("data identity 3 seed 1 json",
     lambda reply: len(json.loads(reply)) == 3),
    ("data phone 2 us csv",
     lambda reply: len(list(csv.DictReader(
         io.StringIO(reply)))) == 2),
    ("data creative band 2 seed 4 text",
     lambda reply: "band" in reply),
])
def test_command_data_answers_in_text(tmp_path, tail, check):
    runtime = _Runtime(tmp_path)
    reply = runtime._control_mock("mock", tail, _message())
    assert reply
    assert runtime.send.calls == []
    assert check(reply)


def test_command_bulk_stays_owner_gated(tmp_path):
    runtime = _Runtime(tmp_path, operator=False)
    reply = runtime._control_mock(
        "mock", "demo bulk-list whatsapp 25", _message())
    assert reply == "that one's just for the owner."
    reply = runtime._control_mock(
        "mock", "data identity 3", _message())
    assert reply == "that one's just for the owner."
    assert runtime.send.calls == []


def test_service_delivers_a_bulk_list(tmp_path):
    service = MockScreenService(media_dir=tmp_path / "m")
    send = _Send()
    out = service.render_list(
        bulk_list_screen(count=50, seed=7),
        audience="personal", send=send)
    assert out["sent"] is True
    ref, caption = send.calls[0]
    assert ref.kind == "image" and ref.mime == "image/png"
    assert Path(ref.path).exists()
    assert "mock screen" in caption


# ── regression: existing demos byte-identical ────────────

def test_existing_demos_still_render_deterministically():
    one, _ = render_screen(demo_screen("whatsapp"))
    two, _ = render_screen(demo_screen("whatsapp"))
    assert one == two
    one, _ = render_list_screen(
        demo_list_screen("whatsapp-list", "android"))
    two, _ = render_list_screen(
        demo_list_screen("whatsapp-list", "android"))
    assert one == two
