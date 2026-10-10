"""Wiring tests: idle connectors → user-facing surfaces.

P1: /image backends — leonardo, stability_ai, nano_banana dispatch through
    get_backend() and translate onto the real connector methods.
P2: /connectors chat command — list / status / connect; never raises,
    never echoes secrets.
P3: /notion, /gcal, /trello chat commands — reachable from dispatch,
    honest failures, owner-explicit writes.
P4: /exness, /stripe chat commands — trading + payments, same contract
    (reachable from dispatch, honest failures, explicit writes, money
    moves confirmation-gated at the connector).

No network anywhere: connectors are faked, HTTP is mocked.
"""

from __future__ import annotations

import base64
import io
import os
import sys
from types import SimpleNamespace
from unittest import mock

import pytest

from nomorals.accounts.vault import CredentialVault
from nomorals.agents.partner.runtime import PartnerRuntime
from nomorals.agents.partner.runtime_system import RuntimeSystemMixin
from nomorals.media_edit.generate import (
    PAID_IMAGE_BACKENDS,
    GenerativeEditError,
    LeonardoBackend,
    NanoBananaBackend,
    StabilityAIBackend,
    _PaidAPIBackend,
    backend_status,
    get_backend,
)
from nomorals.social.chat.control import (
    COMMAND_DETAILS,
    CONTROL_COMMANDS,
    help_text,
    parse_control,
)
from nomorals.storage.db import Database

PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwAD"
    "hgGAWjR9awAAAABJRU5ErkJggg=="
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _vault() -> CredentialVault:
    return CredentialVault(Database(":memory:"), master_passphrase="test")


def _pil_bytes(color=(255, 0, 0), size=(8, 8)) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


class FakeCtx:
    def __init__(self, db):
        self.db = db
        self.settings = SimpleNamespace()


class FakeSelf(RuntimeSystemMixin):
    """Minimal stand-in with what RuntimeSystemMixin methods need.

    Inherits the mixin so helper calls (self._chat_connector, ...) resolve;
    only `context` is faked.
    """

    def __init__(self, db=None):
        self.context = FakeCtx(db)


@pytest.fixture()
def chat_env(monkeypatch):
    """Chat-side vault env: passphrase set, no connector keys."""
    monkeypatch.setenv("NM_VAULT_PASSPHRASE", "test")
    for var in ("LEONARDO_API_KEY", "STABILITY_API_KEY", "NANO_BANANA_API_KEY",
                "GEMINI_API_KEY", "NOTION_TOKEN", "TRELLO_API_KEY",
                "TRELLO_API_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    return _vault()


def _mixin(name, fake_self, *args):
    fn = getattr(RuntimeSystemMixin, name)
    return fn.__get__(fake_self)(*args)


# ---------------------------------------------------------------------------
# P1: /image backend dispatch
# ---------------------------------------------------------------------------

class TestPaidBackendDispatch:
    def test_all_three_dispatch(self):
        assert isinstance(get_backend("leonardo", vault=object()),
                          LeonardoBackend)
        assert isinstance(get_backend("stability_ai", vault=object()),
                          StabilityAIBackend)
        assert isinstance(get_backend("nano_banana", vault=object()),
                          NanoBananaBackend)

    def test_env_selection(self, monkeypatch):
        monkeypatch.setenv("MEDIA_GEN_BACKEND", "stability_ai")
        assert isinstance(get_backend(vault=object()), StabilityAIBackend)

    def test_auto_never_selects_paid(self, monkeypatch):
        monkeypatch.setenv("MEDIA_GEN_BACKEND", "auto")
        with mock.patch(
            "nomorals.media_edit.generate.DiffusersBackend.available",
                return_value=False), \
            mock.patch.dict(sys.modules, {"huggingface_hub": None}):
            with pytest.raises(GenerativeEditError) as excinfo:
                get_backend(vault=object())
        # the error names the paid backends as opt-in options, but auto
        # itself must never return one
        assert "leonardo" in str(excinfo.value)

    def test_unknown_backend_still_errors(self):
        with pytest.raises(GenerativeEditError) as excinfo:
            get_backend("nope", vault=object())
        assert "unknown MEDIA_GEN_BACKEND" in str(excinfo.value)

    def test_paid_constant(self):
        assert set(PAID_IMAGE_BACKENDS) == {
            "leonardo", "stability_ai", "nano_banana"}

    def test_vault_locked_is_clear_not_bare(self, monkeypatch):
        monkeypatch.delenv("NM_VAULT_PASSPHRASE", raising=False)
        with pytest.raises(GenerativeEditError) as excinfo:
            get_backend("leonardo")
        assert "NM_VAULT_PASSPHRASE" in str(excinfo.value)

    def test_backend_status_never_raises_and_lists_paid(self, monkeypatch):
        monkeypatch.delenv("NM_VAULT_PASSPHRASE", raising=False)
        st = backend_status()  # must not raise
        assert set(st["paid"]) == set(PAID_IMAGE_BACKENDS)
        assert st["paid"]["leonardo"]["ready"] is False
        assert "NM_VAULT_PASSPHRASE" in st["paid"]["leonardo"]["reason"]

    def test_describe_names_cost(self):
        for cls in (LeonardoBackend, StabilityAIBackend, NanoBananaBackend):
            d = cls.__new__(cls).describe()
            assert "paid" in d.lower()


class TestAspectSelection:
    def test_closest_aspect(self):
        assert _PaidAPIBackend._closest_aspect(
            1344, 768, NanoBananaBackend.ASPECTS) == "16:9"
        assert _PaidAPIBackend._closest_aspect(
            768, 1344, NanoBananaBackend.ASPECTS) == "9:16"
        assert _PaidAPIBackend._closest_aspect(
            1024, 1024, NanoBananaBackend.ASPECTS) == "1:1"
        assert _PaidAPIBackend._closest_aspect(
            None, None, StabilityAIBackend.ASPECTS) == "1:1"
        assert _PaidAPIBackend._closest_aspect(
            1600, 900, StabilityAIBackend.ASPECTS) == "16:9"


class _FakeLeonardo:
    def __init__(self):
        self.calls = []

    def generate_image(self, prompt, **kw):
        self.calls.append(("generate_image", prompt, kw))
        return [{"id": "gen1", "url": "https://x/y.png"},
                {"id": "gen2", "url": "https://x/z.png"}][:kw.get(
                    "num_images", 1)]

    def edit_image(self, image, prompt, **kw):
        self.calls.append(("edit_image", prompt, kw))
        return [{"id": "e1", "url": "https://x/e.png"}]


class _FakeStability:
    def __init__(self):
        self.calls = []

    def generate_image(self, prompt, **kw):
        self.calls.append(("generate_image", prompt, kw))
        return {"image_bytes": _pil_bytes(), "mime_type": "image/png",
                "seed": 1, "finish_reason": "SUCCESS"}

    def edit_image(self, image, prompt, **kw):
        self.calls.append(("edit_image", prompt, kw))
        return {"image_bytes": _pil_bytes((0, 0, 255)),
                "mime_type": "image/png", "seed": 2,
                "finish_reason": "SUCCESS"}


class _FakeNanoBanana:
    def __init__(self):
        self.calls = []

    def generate_image(self, prompt, **kw):
        self.calls.append(("generate_image", prompt, kw))
        return [{"image_bytes": _pil_bytes((0, 255, 0)),
                 "mime_type": "image/png", "text": ""}]

    def edit_image(self, image, prompt, **kw):
        self.calls.append(("edit_image", prompt, kw))
        return [{"image_bytes": _pil_bytes((255, 255, 0)),
                 "mime_type": "image/png", "text": ""}]


def _patch_connector(fake):
    return mock.patch("nomorals.connectors.registry.create_connector",
                      return_value=fake)


class TestPaidBackendCalls:
    def test_leonardo_generate(self):
        fake = _FakeLeonardo()
        be = LeonardoBackend(vault=object())
        with _patch_connector(fake), \
                mock.patch.object(LeonardoBackend, "_download",
                                  return_value=_pil_bytes()) as dl:
            images = be.generate("a cat", width=1024, height=1024, n=2)
        assert len(images) == 2
        assert dl.call_count == 2
        _name, prompt, kw = fake.calls[0]
        assert prompt == "a cat"
        assert kw["confirmed"] is True  # owner typed the prompt
        assert kw["num_images"] == 2
        assert kw["width"] == 1024 and kw["height"] == 1024

    def test_leonardo_strength_mapping(self):
        """Our strength = regeneration; Leonardo init_strength = preservation."""
        fake = _FakeLeonardo()
        be = LeonardoBackend(vault=object())
        from PIL import Image
        with _patch_connector(fake), \
                mock.patch.object(LeonardoBackend, "_download",
                                  return_value=_pil_bytes()):
            be.img2img(Image.new("RGB", (64, 64)), "make it sunset",
                       strength=0.75)
        _name, _prompt, kw = fake.calls[0]
        assert kw["init_strength"] == pytest.approx(0.25)

    def test_leonardo_edit_applies_mask(self):
        fake = _FakeLeonardo()
        be = LeonardoBackend(vault=object())
        from PIL import Image
        red = Image.new("RGB", (32, 32), (255, 0, 0))
        with _patch_connector(fake), \
                mock.patch.object(LeonardoBackend, "_download",
                                  return_value=_pil_bytes((0, 0, 255),
                                                          (32, 32))):
            out = be.edit(red, "blue", mask=(0, 0, 8, 8), strength=0.5)
        # unmasked corner stays red (original preserved through the mask)
        assert out.getpixel((30, 30)) == (255, 0, 0)

    def test_stability_generate(self):
        fake = _FakeStability()
        be = StabilityAIBackend(vault=object())
        with _patch_connector(fake):
            images = be.generate("a dog", width=1344, height=768,
                                 seed=7, n=2)
        assert len(images) == 2
        assert all(hasattr(i, "save") for i in images)
        _name, prompt, kw = fake.calls[0]
        assert kw["confirmed"] is True
        assert kw["aspect_ratio"] == "16:9"
        assert kw["model"] == "sd3.5-large"
        assert fake.calls[1][2]["seed"] == 8  # seed varies across n

    def test_stability_img2img_strength_passthrough(self):
        fake = _FakeStability()
        be = StabilityAIBackend(vault=object())
        from PIL import Image
        with _patch_connector(fake):
            be.img2img(Image.new("RGB", (32, 32)), "p", strength=0.6)
        assert fake.calls[0][2]["strength"] == pytest.approx(0.6)
        assert fake.calls[0][2]["confirmed"] is True

    def test_nanobanana_generate(self):
        fake = _FakeNanoBanana()
        be = NanoBananaBackend(vault=object())
        with _patch_connector(fake):
            images = be.generate("a bird", width=768, height=1344)
        assert len(images) == 1
        _name, prompt, kw = fake.calls[0]
        assert kw["confirmed"] is True
        assert kw["aspect_ratio"] == "9:16"

    def test_nanobanana_model_env(self, monkeypatch):
        monkeypatch.setenv("NANO_BANANA_MODEL", "gemini-3-pro-image-preview")
        be = NanoBananaBackend(vault=object())
        assert be.model == "gemini-3-pro-image-preview"

    def test_generate_needs_prompt(self):
        be = StabilityAIBackend(vault=object())
        with pytest.raises(GenerativeEditError):
            be.generate("  ")
        with pytest.raises(GenerativeEditError):
            be.generate("x", n=0)

    def test_connector_failure_becomes_clear_error(self):
        class Boom:
            def generate_image(self, *a, **k):
                raise RuntimeError("api exploded")

        be = LeonardoBackend(vault=object())
        with _patch_connector(Boom()):
            with pytest.raises(GenerativeEditError) as excinfo:
                be.generate("x")
        assert "leonardo generation failed" in str(excinfo.value)


# ---------------------------------------------------------------------------
# P2: /connectors chat command
# ---------------------------------------------------------------------------

class TestConnectorsCommand:
    def test_registered(self):
        for cmd in ("connectors", "notion", "gcal", "trello"):
            assert cmd in CONTROL_COMMANDS, cmd
            assert cmd in COMMAND_DETAILS, cmd
            assert f"/{cmd}" in help_text(), cmd

    def test_parse_shapes(self):
        c = parse_control("/connectors status leonardo")
        assert (c.kind, c.tail) == ("connectors", "status leonardo")
        c = parse_control("/connectors connect wise")
        assert c.kind == "connectors"
        assert parse_control("/notion dbs").kind == "notion"
        assert parse_control("/gcal agenda 3").kind == "gcal"
        assert parse_control("/trello boards").kind == "trello"
        assert parse_control("/exness balance").kind == "exness"
        assert parse_control("/stripe balance").kind == "stripe"

    def test_dispatch_targets_exist(self):
        for m in ("_control_connectors", "_control_notion",
                  "_control_gcal", "_control_trello",
                  "_control_exness", "_control_stripe"):
            assert callable(getattr(PartnerRuntime, m, None)), m

    def test_list_table(self, chat_env, monkeypatch):
        monkeypatch.setenv("NM_VAULT_PASSPHRASE", "test")
        out = _mixin("_control_connectors", FakeSelf(chat_env.db), "")
        assert isinstance(out, str)
        from nomorals.connectors import list_connectors as _lc
        n = len(_lc())
        assert f"connectors ({n}):" in out.splitlines()[0]
        for cid in ("leonardo", "notion", "gcalendar", "trello", "wise",
                    "stability_ai", "nano_banana"):
            assert cid in out, cid

    def test_unknown_name_suggests(self, chat_env, monkeypatch):
        monkeypatch.setenv("NM_VAULT_PASSPHRASE", "test")
        out = _mixin("_control_connectors", FakeSelf(chat_env.db),
                     "status leonrdo")
        assert "unknown connector" in out
        assert "leonardo" in out  # did-you-mean

    def test_connect_guided_names_env_var_no_echo(self, chat_env,
                                                  monkeypatch):
        """No key in env → guided message naming the variable; the key is
        never echoed because it never travels through chat."""
        monkeypatch.setenv("NM_VAULT_PASSPHRASE", "test")
        out = _mixin("_control_connectors", FakeSelf(chat_env.db),
                     "connect leonardo")
        assert "LEONARDO_API_KEY" in out
        assert "never" in out and "chat" in out

    def test_connect_already_connected(self, chat_env, monkeypatch):
        monkeypatch.setenv("NM_VAULT_PASSPHRASE", "test")
        chat_env.store(service="connector:wise", username="w-user",
                       password="sekret", credential_type="api_key")
        out = _mixin("_control_connectors", FakeSelf(chat_env.db),
                     "connect wise")
        assert "already connected" in out
        assert "w-user" in out
        assert "sekret" not in out  # never echoed

    def test_status_not_connected(self, chat_env, monkeypatch):
        monkeypatch.setenv("NM_VAULT_PASSPHRASE", "test")
        out = _mixin("_control_connectors", FakeSelf(chat_env.db),
                     "status trello")
        assert "not connected" in out

    def test_vault_locked_message(self, monkeypatch):
        monkeypatch.delenv("NM_VAULT_PASSPHRASE", raising=False)
        out = _mixin("_control_connectors", FakeSelf(Database(":memory:")),
                     "list")
        assert "vault is locked" in out
        assert "NM_VAULT_PASSPHRASE" in out

    def test_never_raises(self, chat_env, monkeypatch):
        monkeypatch.setenv("NM_VAULT_PASSPHRASE", "test")
        for tail in ["", "bogus", "status", "connect", "connect !!!",
                     "status " + "x" * 200, "list extra words here"]:
            out = _mixin("_control_connectors", FakeSelf(chat_env.db), tail)
            assert isinstance(out, str) and out, tail

    def test_chat_connector_not_connected_hint(self, chat_env, monkeypatch):
        monkeypatch.setenv("NM_VAULT_PASSPHRASE", "test")
        fake = FakeSelf(chat_env.db)
        with pytest.raises(Exception) as excinfo:
            RuntimeSystemMixin._chat_connector.__get__(fake)("notion")
        assert "/connectors connect notion" in str(excinfo.value)


# ---------------------------------------------------------------------------
# P3: /notion, /gcal, /trello
# ---------------------------------------------------------------------------

class _FakeNotion:
    def __init__(self):
        self.calls = []
        self.dbs = [
            {"id": "db-1",
             "title": [{"plain_text": "Tasks"}]},
        ]

    def list_databases(self, query="", page_size=50):
        self.calls.append(("list_databases", query))
        if query:
            return [d for d in self.dbs
                    if query.lower() in d["title"][0]["plain_text"].lower()]
        return self.dbs

    def query_database(self, database_id, page_size=50, **kw):
        self.calls.append(("query_database", database_id))
        return {"results": [
            {"id": "p1", "properties": {
                "Name": {"type": "title",
                         "title": [{"plain_text": "Buy milk"}]}}},
            {"id": "p2", "properties": {
                "Name": {"type": "title",
                         "title": [{"plain_text": "Read book"}]}}},
        ], "has_more": False, "next_cursor": None}

    def create_page(self, **kw):
        self.calls.append(("create_page", kw))
        return {"id": "new-1", "url": "https://notion.so/new-1"}


class _FakeGCal:
    def __init__(self):
        self.calls = []

    def list_events(self, **kw):
        self.calls.append(("list_events", kw))
        return {"events": [
            {"summary": "Dentist",
             "start": {"dateTime": "2026-10-10T14:00:00+01:00"},
             "location": "Lagos"},
            {"summary": "Gym", "start": {"date": "2026-10-11"}},
        ], "next_page_token": "", "time_zone": "Africa/Lagos"}

    def create_event(self, summary, start, end, **kw):
        self.calls.append(("create_event", summary, start, end, kw))
        return {"id": "ev1", "summary": summary, "start": start,
                "htmlLink": "https://cal/ev1"}


class _FakeTrello:
    def __init__(self):
        self.calls = []
        self.boards = [{"id": "b1", "name": "Personal"}]
        self.lists = {"b1": [{"id": "l1", "name": "Todo"},
                             {"id": "l2", "name": "Done"}]}

    def list_boards(self):
        self.calls.append(("list_boards",))
        return self.boards

    def list_lists(self, board_id):
        self.calls.append(("list_lists", board_id))
        return self.lists.get(board_id, [])

    def list_cards(self, list_id):
        self.calls.append(("list_cards", list_id))
        return [{"id": "c1", "name": "Buy milk", "due": "2026-10-12T00:00:00Z",
                 "url": "https://trello/c1"}]

    def create_card(self, list_id, name, **kw):
        self.calls.append(("create_card", list_id, name, kw))
        return {"id": "c9", "name": name, "url": "https://trello/c9"}


def _with_fake_connector(fake):
    return mock.patch.object(RuntimeSystemMixin, "_chat_connector",
                             return_value=fake)


class TestNotionCommand:
    def test_dbs(self):
        fake = _FakeNotion()
        with _with_fake_connector(fake):
            out = _mixin("_control_notion", FakeSelf(), "dbs")
        assert "Tasks" in out

    def test_query_filters_client_side(self):
        fake = _FakeNotion()
        with _with_fake_connector(fake):
            out = _mixin("_control_notion", FakeSelf(), "query Tasks milk")
        assert "Buy milk" in out
        assert "Read book" not in out

    def test_query_by_id(self):
        fake = _FakeNotion()
        with _with_fake_connector(fake):
            out = _mixin("_control_notion", FakeSelf(), "query db-1")
        assert "Buy milk" in out and "Read book" in out

    def test_add_builds_page_confirmed(self):
        fake = _FakeNotion()
        with _with_fake_connector(fake):
            out = _mixin("_control_notion", FakeSelf(),
                         "add parent-1 Shopping | milk | eggs")
        assert "created" in out
        _name, kw = fake.calls[-1]
        assert _name == "create_page"
        assert kw["parent_page_id"] == "parent-1"
        assert kw["title"] == "Shopping"
        assert kw["confirmed"] is True  # owner typed the exact page
        paras = [c["paragraph"]["rich_text"][0]["text"]["content"]
                 for c in kw["children"]]
        assert paras == ["milk", "eggs"]
        assert "notion.so" in out

    def test_add_usage(self):
        fake = _FakeNotion()
        with _with_fake_connector(fake):
            out = _mixin("_control_notion", FakeSelf(), "add only-one-seg")
        assert "usage:" in out

    def test_title_extraction(self):
        m = RuntimeSystemMixin._notion_title_of
        db = {"title": [{"plain_text": "Tasks"}]}
        assert m(db) == "Tasks"
        page = {"properties": {
            "Name": {"type": "title",
                     "title": [{"plain_text": "Buy milk"}]},
            "Status": {"type": "status", "status": {"name": "Todo"}},
        }}
        assert m(page) == "Buy milk"
        assert m({}) == "(untitled)"

    def test_resolve_db_by_name(self):
        fake = _FakeNotion()
        m = RuntimeSystemMixin._notion_resolve_db.__get__(FakeSelf())
        assert m(fake, "tasks") == "db-1"
        assert m(fake, "db-1") == "db-1"
        with pytest.raises(Exception) as excinfo:
            m(fake, "nope")
        assert "no database" in str(excinfo.value)

    def test_empty_usage(self):
        fake = _FakeNotion()
        with _with_fake_connector(fake):
            out = _mixin("_control_notion", FakeSelf(), "")
        assert "/notion dbs" in out

    def test_not_connected(self, chat_env, monkeypatch):
        monkeypatch.setenv("NM_VAULT_PASSPHRASE", "test")
        out = _mixin("_control_notion", FakeSelf(chat_env.db), "dbs")
        assert "/connectors connect notion" in out


class TestGCalCommand:
    def test_agenda(self):
        fake = _FakeGCal()
        with _with_fake_connector(fake):
            out = _mixin("_control_gcal", FakeSelf(), "agenda 2")
        assert "Dentist" in out and "Gym" in out
        _name, kw = fake.calls[0]
        assert _name == "list_events"
        assert kw["time_min"] and kw["time_max"]

    def test_add(self):
        fake = _FakeGCal()
        with _with_fake_connector(fake):
            out = _mixin("_control_gcal", FakeSelf(),
                         "add Dentist | 2026-10-10 14:00 | 60")
        assert "created" in out
        _name, summary, start, end, kw = fake.calls[-1]
        assert summary == "Dentist"
        assert kw["confirmed"] is True  # owner typed the exact event
        assert start["dateTime"].startswith("2026-10-10T14:00:00")
        assert "+00:00" in start["dateTime"] or start["dateTime"][-3] == ":"
        assert end["dateTime"].startswith("2026-10-10T15:00:00")

    def test_add_with_end_time(self):
        fake = _FakeGCal()
        with _with_fake_connector(fake):
            _mixin("_control_gcal", FakeSelf(),
                   "add Call | 2026-10-10 14:00 | 2026-10-10 15:30")
        _name, _s, start, end, _kw = fake.calls[-1]
        assert end["dateTime"].startswith("2026-10-10T15:30:00")

    def test_add_bad_input(self):
        fake = _FakeGCal()
        with _with_fake_connector(fake):
            out = _mixin("_control_gcal", FakeSelf(), "add Dentist")
        assert "usage:" in out
        with _with_fake_connector(fake):
            out = _mixin("_control_gcal", FakeSelf(),
                         "add X | not-a-date | 60")
        assert "can't parse time" in out

    def test_parse_time_naive_gets_owner_tz(self, monkeypatch):
        monkeypatch.setenv("TZ", "Africa/Lagos")
        dt = RuntimeSystemMixin._gcal_parse_time("2026-10-10 14:00",
                                                 "Africa/Lagos")
        assert dt.utcoffset().total_seconds() == 3600

    def test_not_connected(self, chat_env, monkeypatch):
        monkeypatch.setenv("NM_VAULT_PASSPHRASE", "test")
        out = _mixin("_control_gcal", FakeSelf(chat_env.db), "")
        assert "/connectors connect gcalendar" in out


class TestTrelloCommand:
    def test_boards(self):
        fake = _FakeTrello()
        with _with_fake_connector(fake):
            out = _mixin("_control_trello", FakeSelf(), "boards")
        assert "Personal" in out

    def test_lists_by_board_name(self):
        fake = _FakeTrello()
        with _with_fake_connector(fake):
            out = _mixin("_control_trello", FakeSelf(), "lists personal")
        assert "Todo" in out and "Done" in out

    def test_cards(self):
        fake = _FakeTrello()
        with _with_fake_connector(fake):
            out = _mixin("_control_trello", FakeSelf(), "cards Todo")
        assert "Buy milk" in out

    def test_add(self):
        fake = _FakeTrello()
        with _with_fake_connector(fake):
            out = _mixin("_control_trello", FakeSelf(),
                         "add Todo | buy eggs | dozen free-range")
        assert "created" in out
        _name, list_id, name, kw = fake.calls[-1]
        assert (list_id, name) == ("l1", "buy eggs")
        assert kw["description"] == "dozen free-range"
        assert kw["confirmed"] is True  # owner typed the exact card

    def test_resolve_board_ambiguity(self):
        fake = _FakeTrello()
        fake.boards = [{"id": "b1", "name": "Work stuff"},
                       {"id": "b2", "name": "Work notes"}]
        m = RuntimeSystemMixin._trello_resolve_board.__get__(FakeSelf())
        with pytest.raises(Exception) as excinfo:
            m(fake, "work")
        assert "several boards" in str(excinfo.value)

    def test_add_usage(self):
        fake = _FakeTrello()
        with _with_fake_connector(fake):
            out = _mixin("_control_trello", FakeSelf(), "add")
        assert "usage:" in out

    def test_not_connected(self, chat_env, monkeypatch):
        monkeypatch.setenv("NM_VAULT_PASSPHRASE", "test")
        out = _mixin("_control_trello", FakeSelf(chat_env.db), "boards")
        assert "/connectors connect trello" in out


class TestNeverRaises:
    """Every new chat command returns text for arbitrary input."""

    @pytest.mark.parametrize("cmd,tails", [
        ("_control_notion", ["", "xyz", "dbs " * 50, "add | | |"]),
        ("_control_gcal", ["", "xyz", "agenda abc", "add | |"]),
        ("_control_trello", ["", "xyz", "boards " * 50, "add |"]),
        ("_control_connectors", ["", "xyz", "status", "connect"]),
        ("_control_exness", ["", "xyz", "buy", "buy XAUUSD",
                             "closeall", "closeall confirm"]),
        ("_control_stripe", ["", "xyz", "link", "link abc usd x"]),
    ])
    def test_never_raises(self, cmd, tails, chat_env, monkeypatch):
        monkeypatch.setenv("NM_VAULT_PASSPHRASE", "test")
        for tail in tails:
            out = _mixin(cmd, FakeSelf(chat_env.db), tail)
            assert isinstance(out, str) and out.strip(), (cmd, tail)


class _FakeExness:
    def __init__(self):
        self.calls = []

    def get_snapshot(self):
        self.calls.append(("get_snapshot",))
        return {"account_state": {"balance": "1000.5", "equity": "1002.0",
                                  "used_margin": "10.0"},
                "positions": [{"position_id": "p1", "direction": "buy",
                               "volume": "0.01", "instrument": "XAUUSD",
                               "open_price": "2650.00"}],
                "orders": []}

    def get_candles(self, instrument, timeframe, **kw):
        self.calls.append(("get_candles", instrument, timeframe, kw))
        return [{"time": "2026-10-09T10:00:00Z", "open": 2648.0,
                 "high": 2652.0, "low": 2647.0, "close": 2650.5,
                 "volume": 120}]

    def open_position(self, instrument, side, volume, **kw):
        self.calls.append(("open_position", instrument, side, volume, kw))
        return {"status": "accepted", "operation_id": "op-1"}

    def close_position(self, position_id, **kw):
        self.calls.append(("close_position", position_id, kw))
        return {"status": "accepted", "operation_id": "op-2"}

    def close_all_positions(self, **kw):
        self.calls.append(("close_all_positions", kw))
        return {"status": "accepted", "operation_id": "op-3"}


class TestExnessCommand:
    def test_balance(self):
        fake = _FakeExness()
        with _with_fake_connector(fake):
            out = _mixin("_control_exness", FakeSelf(), "balance")
        assert "1000.5" in out and "1002.0" in out

    def test_positions(self):
        fake = _FakeExness()
        with _with_fake_connector(fake):
            out = _mixin("_control_exness", FakeSelf(), "positions")
        assert "p1" in out and "XAUUSD" in out

    def test_price(self):
        fake = _FakeExness()
        with _with_fake_connector(fake):
            out = _mixin("_control_exness", FakeSelf(), "price xauusd h1")
        assert "2650.5" in out
        _name, sym, tf, _kw = fake.calls[-1]
        assert (sym, tf) == ("XAUUSD", "H1")

    def test_buy_confirmed(self):
        fake = _FakeExness()
        with _with_fake_connector(fake):
            out = _mixin("_control_exness", FakeSelf(),
                         "buy XAUUSD 0.01 2640 2660")
        assert "accepted" in out
        _name, sym, side, vol, kw = fake.calls[-1]
        assert (sym, side, vol) == ("XAUUSD", "buy", 0.01)
        assert kw["confirmed"] is True  # owner typed the exact order
        assert kw["stop_loss"] == 2640.0 and kw["take_profit"] == 2660.0

    def test_buy_bad_lots(self):
        fake = _FakeExness()
        with _with_fake_connector(fake):
            out = _mixin("_control_exness", FakeSelf(), "buy XAUUSD abc")
        assert "must be a number" in out

    def test_close(self):
        fake = _FakeExness()
        with _with_fake_connector(fake):
            out = _mixin("_control_exness", FakeSelf(), "close p1")
        assert "op-2" in out

    def test_closeall_needs_confirm_word(self):
        fake = _FakeExness()
        with _with_fake_connector(fake):
            out = _mixin("_control_exness", FakeSelf(), "closeall")
        assert "confirm" in out
        assert not [c for c in fake.calls if c[0] == "close_all_positions"]
        with _with_fake_connector(fake):
            out = _mixin("_control_exness", FakeSelf(), "closeall confirm")
        assert "op-3" in out

    def test_usage(self):
        fake = _FakeExness()
        with _with_fake_connector(fake):
            out = _mixin("_control_exness", FakeSelf(), "")
        assert "/exness balance" in out

    def test_not_connected(self, chat_env, monkeypatch):
        monkeypatch.setenv("NM_VAULT_PASSPHRASE", "test")
        out = _mixin("_control_exness", FakeSelf(chat_env.db), "balance")
        assert "/connectors connect exness" in out


class _FakeStripe:
    def __init__(self):
        self.calls = []

    def get_balance(self):
        self.calls.append(("get_balance",))
        return {"available": [{"amount": 25000, "currency": "usd"}],
                "pending": [{"amount": 1000, "currency": "usd"}]}

    def list_customers(self, **kw):
        self.calls.append(("list_customers", kw))
        return [{"id": "cus_1", "name": "Ada"}]

    def list_charges(self, **kw):
        self.calls.append(("list_charges", kw))
        return [{"id": "ch_1", "amount": 2500, "currency": "usd",
                 "status": "succeeded"}]

    def create_payment_link(self, line_items, **kw):
        self.calls.append(("create_payment_link", line_items, kw))
        return {"id": "plink_1", "url": "https://pay.stripe.test/1"}


class TestStripeCommand:
    def test_balance(self):
        fake = _FakeStripe()
        with _with_fake_connector(fake):
            out = _mixin("_control_stripe", FakeSelf(), "balance")
        assert "250.00 USD" in out and "10.00 USD" in out

    def test_customers(self):
        fake = _FakeStripe()
        with _with_fake_connector(fake):
            out = _mixin("_control_stripe", FakeSelf(), "customers")
        assert "cus_1" in out and "Ada" in out

    def test_charges(self):
        fake = _FakeStripe()
        with _with_fake_connector(fake):
            out = _mixin("_control_stripe", FakeSelf(), "charges")
        assert "ch_1" in out and "25.00 USD" in out

    def test_link(self):
        fake = _FakeStripe()
        with _with_fake_connector(fake):
            out = _mixin("_control_stripe", FakeSelf(),
                         "link 25.50 usd logo gig")
        assert "https://pay.stripe.test/1" in out
        _name, items, _kw = fake.calls[-1]
        pd = items[0]["price_data"]
        assert (pd["unit_amount"], pd["currency"]) == (2550, "usd")
        assert pd["product_data"]["name"] == "logo gig"

    def test_link_bad_amount(self):
        fake = _FakeStripe()
        with _with_fake_connector(fake):
            out = _mixin("_control_stripe", FakeSelf(), "link abc usd x")
        assert "must be a number" in out

    def test_usage(self):
        fake = _FakeStripe()
        with _with_fake_connector(fake):
            out = _mixin("_control_stripe", FakeSelf(), "")
        assert "/stripe balance" in out

    def test_not_connected(self, chat_env, monkeypatch):
        monkeypatch.setenv("NM_VAULT_PASSPHRASE", "test")
        out = _mixin("_control_stripe", FakeSelf(chat_env.db), "balance")
        assert "/connectors connect stripe" in out