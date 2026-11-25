"""Craft-AI-Integrations port (census §3A.22, lane 115).

The ai-client layer Devon never had: ConsentTerm + agree/forget_lapsed,
ModelFolder store field, ConnectionDraft + presets, the image.segment
contract, layered secret stores, the file-backed SharedStore, the
ProviderFactory (incl. the OpenAI-compatible adapter over llm/ and the
declarative spec adapter), AiClient + ToolDesk, and the ai-http single
TLS policy with its workspace scan. Base: 64821b6c. No new tool is added
here, so the registry count stays 733; if lane 114 (craftfonts) ships
first, the parent re-bases that pin to 734.
"""

from __future__ import annotations

import json
import os
import struct
import threading
import unittest
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

from nomorals.craftlibs.ai.capability import CapabilityId
from nomorals.craftlibs.ai.client import (
    NO_ADAPTER,
    AiClient,
    Choice,
    Finished,
    Job,
    PickedChoice,
    Plan,
    RoleReady,
    RoleStatus,
)
from nomorals.craftlibs.ai.conformance import run_lifecycle
from nomorals.craftlibs.ai.connection import (
    Auth,
    Billing,
    Connection,
    Locality,
)
from nomorals.craftlibs.ai.consent import ConsentLedger, Disclosure
from nomorals.craftlibs.ai.consent_term import ConsentTerm
from nomorals.craftlibs.ai.draft import (
    PRESETS,
    ConnectionDraft,
    DraftError,
    Field,
    KeyChange,
    preset_models,
)
from nomorals.craftlibs.ai.error import AiError
from nomorals.craftlibs.ai.factory import (
    OPENAI_FAMILY,
    OpenAiCompatProvider,
    ProviderFactory,
    check_provider,
    register_checked,
)
from nomorals.craftlibs.ai.history import HistoryLog
from nomorals.craftlibs.ai.http import (
    agent_for,
    hosted_proxy,
    loopback_proxy,
    proxy_for,
    scan_agents,
    tls_cafile,
    tls_context,
)
from nomorals.craftlibs.ai.ids import ModelId
from nomorals.craftlibs.ai.local import LocalMotionProvider
from nomorals.craftlibs.ai.mock import MockProvider
from nomorals.craftlibs.ai.model import (
    CapabilitySpec,
    InputRole,
    ModelInfo,
)
from nomorals.craftlibs.ai.model_folders import (
    KINDS,
    MAX_DIRS_PER_KIND,
    MAX_FOLDERS,
    FolderError,
    FolderOrigin,
    ModelFolder,
    download_target,
    is_kind,
    is_subfolder,
    looks_absolute,
    validate_folders,
)
from nomorals.craftlibs.ai.payload import Payload
from nomorals.craftlibs.ai.request import Request
from nomorals.craftlibs.ai.role import Binding, Role
from nomorals.craftlibs.ai.secrets import (
    EnvStore,
    LayeredStore,
    MemoryStore,
    secret_for,
)
from nomorals.craftlibs.ai.segment import (
    CAPABILITY,
    FOUND_MEDIA_TYPE,
    MASK_MEDIA_TYPE,
    MAX_NAME_CHARS,
    MAX_POINTS,
    Area,
    Click,
    Found,
    area,
    clicks,
    description,
    found_output,
    mask,
    read_found,
    with_area,
    with_clicks,
)
from nomorals.craftlibs.ai.shared import (
    HistoryFile,
    SharedError,
    SharedStore,
)
from nomorals.craftlibs.ai.store import StoreError, StoreFile
from nomorals.craftlibs.ai.tools import (
    Accepted,
    SampleReport,
    TestStart,
    ToolDesk,
)
from nomorals.llm.benchmark import _LlamaServerClient
from nomorals.tools.registry import ToolRegistry

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "craftai"


def _dot_png() -> bytes:
    """One real 1x1 RGBA PNG, built from bytes (no PIL)."""
    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return (struct.pack(">I", len(data)) + body
                + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF))

    raw = b"\x00" + b"\xff\x00\x00\xff"
    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw))
    png += chunk(b"IEND", b"")
    return png


def _text_model(model_id: str = "mock-text") -> ModelInfo:
    return ModelInfo.new(
        ModelId.new(model_id), "Mock text model").with_spec(
            CapabilitySpec.new(CapabilityId.parse("text.generate"))
            .accepting(InputRole.PROMPT, 1)
            .requiring(InputRole.PROMPT))


def _segment_model(model_id: str = "mock-segment") -> ModelInfo:
    return ModelInfo.new(
        ModelId.new(model_id), "Mock segment model").with_spec(
            CapabilitySpec.new(CapabilityId.parse("image.segment"))
            .accepting(InputRole.SOURCE, 1)
            .accepting(InputRole.PROMPT, 1)
            .requiring(InputRole.SOURCE))


def _mock_connection(models=None) -> Connection:
    return Connection(
        "mock-conn", "Mock", "mock", locality=Locality.LOCAL,
        auth=Auth.none(), billing=Billing.FREE,
        models=models or [_text_model()])


def _text_request(prompt: str = "hello") -> Request:
    return Request.new(CapabilityId.parse("text.generate")).with_input(
        InputRole.PROMPT, Payload.text(prompt))


def _role(role_id: str = "inpaint", capability: str = "text.generate",
          inputs=("prompt",)) -> Role:
    return Role(role_id, "Inpaint", capability,
                [InputRole.parse(i) for i in inputs])


def _disclosure(app: str = "craftai-test") -> Disclosure:
    # Hosted, so consent applies: a local disclosure never asks,
    # and the term tests need a disclosure that does.
    hosted = Connection(
        "hosted-conn", "Hosted", "mock",
        endpoint="https://example.com", locality=Locality.REMOTE,
        models=[_text_model()])
    return Disclosure.for_request(app, hosted, _text_request())


def _store_with_mock(tmp: Path, app: str = "craftai-test") -> SharedStore:
    """A SharedStore whose file holds one mock connection and a default
    binding for the inpaint role."""
    shared = SharedStore(tmp / "ai.json")
    store = StoreFile(
        connections=[_mock_connection()],
        defaults=[Binding("inpaint", "mock-conn", "mock-text")],
        apps={})
    shared.replace(store)
    return shared


def _client(tmp: Path, **kwargs) -> AiClient:
    shared = _store_with_mock(tmp)
    history = HistoryFile(tmp / "history.jsonl")
    return (AiClient("craftai-test", shared, MemoryStore(),
                     factory=ProviderFactory.standard())
            .with_history(history)
            .with_factory(ProviderFactory.standard())
            .keep_prompts_in_history(True))


class ConsentTermTest(unittest.TestCase):
    def test_choices_run_shortest_first_with_distinct_ids(self):
        ids = [t.id() for t in ConsentTerm.ALL]
        self.assertEqual(
            ids, ["this_time", "one_hour", "one_day", "until_revoked"])
        self.assertEqual(len(set(ids)), 4)
        now = 1000
        ends = [t.expires_at(now) for t in ConsentTerm.ALL]
        self.assertEqual(ends, [1000, 4600, 87400, None])

    def test_this_time_is_the_only_term_that_remembers_nothing(self):
        self.assertFalse(ConsentTerm.this_time().remembers())
        for term in (ConsentTerm.one_hour(), ConsentTerm.one_day(),
                     ConsentTerm.until_revoked()):
            self.assertTrue(term.remembers())

    def test_agree_this_time_covers_the_job_without_writing_the_file(self):
        ledger = ConsentLedger()
        disclosure = _disclosure()
        self.assertTrue(ledger.needs_consent(disclosure, 1000))
        job_ledger = ledger.agree(disclosure, ConsentTerm.this_time(), 1000)
        # The caller's ledger is unchanged: nothing was remembered.
        self.assertTrue(ledger.needs_consent(disclosure, 1000))
        # The returned ledger covers the job.
        self.assertFalse(job_ledger.needs_consent(disclosure, 1000))

    def test_agree_one_hour_lapses_after_the_hour(self):
        ledger = ConsentLedger()
        disclosure = _disclosure()
        ledger.agree(disclosure, ConsentTerm.one_hour(), 1000)
        self.assertFalse(ledger.needs_consent(disclosure, 4599))
        self.assertTrue(ledger.needs_consent(disclosure, 4600))

    def test_agree_until_revoked_never_lapses_and_revoke_ends_it(self):
        ledger = ConsentLedger()
        disclosure = _disclosure()
        ledger.agree(disclosure, ConsentTerm.until_revoked(), 1000)
        self.assertFalse(ledger.needs_consent(disclosure, 10 ** 9))
        ledger.revoke("craftai-test", "hosted-conn")
        self.assertTrue(ledger.needs_consent(disclosure, 10 ** 9))

    def test_forget_lapsed_drops_only_expired_agreements(self):
        ledger = ConsentLedger()
        disclosure = _disclosure()
        other_conn = Connection(
            "mock-conn-2", "Mock 2", "mock",
            endpoint="https://example.com/other",
            locality=Locality.REMOTE, auth=Auth.none(),
            models=[_text_model()])
        other = Disclosure.for_request(
            "craftai-test", other_conn, _text_request())
        ledger.agree(disclosure, ConsentTerm.one_hour(), 1000)
        ledger.agree(other, ConsentTerm.until_revoked(), 1000)
        ledger.forget_lapsed(4600)
        # The hour grant lapsed; the until-revoked grant has no expiry.
        self.assertTrue(ledger.needs_consent(disclosure, 4600))
        self.assertFalse(ledger.needs_consent(other, 10 ** 9))


class ModelFoldersTest(unittest.TestCase):
    def test_kinds_are_the_eleven_comfyui_names(self):
        self.assertEqual(
            list(KINDS),
            ["checkpoints", "diffusion_models", "text_encoders", "vae",
             "loras", "controlnet", "clip_vision", "upscale_models",
             "embeddings", "sam3", "background_removal"])
        self.assertEqual(MAX_FOLDERS, 32)
        self.assertEqual(MAX_DIRS_PER_KIND, 64)

    def test_is_kind_is_subfolder_and_looks_absolute(self):
        self.assertTrue(is_kind("checkpoints"))
        self.assertFalse(is_kind("Checkpoints"))
        self.assertFalse(is_kind("bad kind"))
        self.assertTrue(is_subfolder("checkpoints"))
        self.assertTrue(is_subfolder(""))
        self.assertFalse(is_subfolder("../escape"))
        self.assertFalse(is_subfolder("/absolute"))
        self.assertFalse(is_subfolder("a:b"))
        self.assertTrue(looks_absolute("/home/user/models"))
        self.assertTrue(looks_absolute("C:\\models"))
        self.assertTrue(looks_absolute("C:/models"))
        self.assertFalse(looks_absolute("models"))

    def _folder(self, folder_id: str = "main", **overrides) -> ModelFolder:
        data = {"id": folder_id, "label": "Main",
                "path": "/home/user/models", "default": True,
                "origin": FolderOrigin.CHOSEN,
                "kinds": {"checkpoints": ["checkpoints"]}}
        data.update(overrides)
        return ModelFolder(**data)

    def test_validate_accepts_a_good_list(self):
        validate_folders([self._folder()])

    def test_validate_names_each_refusal(self):
        with self.assertRaises(FolderError) as ctx:
            validate_folders([self._folder("a"), self._folder("a")])
        self.assertIn("a", str(ctx.exception))
        with self.assertRaises(FolderError):
            validate_folders([self._folder("Bad Id")])
        with self.assertRaises(FolderError):
            validate_folders([self._folder(path="relative/models")])
        with self.assertRaises(FolderError) as ctx:
            validate_folders(
                [self._folder(kinds={"Bad Kind": ["x"]})])
        self.assertIn("Bad Kind", str(ctx.exception))
        with self.assertRaises(FolderError):
            validate_folders(
                [self._folder(kinds={"checkpoints": ["../out"]})])
        with self.assertRaises(FolderError):
            validate_folders([self._folder(
                kinds={"checkpoints": [f"d{i}" for i in range(65)]})])
        with self.assertRaises(FolderError):
            validate_folders([self._folder(), self._folder("b")])
        with self.assertRaises(FolderError):
            validate_folders(
                [self._folder(str(i)) for i in range(MAX_FOLDERS + 1)])
        with self.assertRaises(FolderError):
            validate_folders([self._folder(label="")])

    def test_download_target_uses_the_default_chosen_folder(self):
        folders = [self._folder(kinds={"checkpoints": ["ckpt"]}),
                   self._folder("second", default=False,
                                path="/home/user/other")]
        folder, sub = download_target(folders, "checkpoints")
        self.assertEqual(folder.id, "main")
        self.assertEqual(sub, "ckpt")
        # An unmapped kind names a new sub-folder under the default.
        folder, sub = download_target(folders, "loras")
        self.assertEqual(sub, "loras")
        # No default folder: nowhere to save.
        self.assertIsNone(download_target(
            [self._folder(default=False)], "checkpoints"))

    def test_store_file_round_trips_model_folders(self):
        store = StoreFile(model_folders=[self._folder()])
        data = store.to_dict()
        self.assertEqual(len(data["model_folders"]), 1)
        back = StoreFile.from_dict(data)
        self.assertEqual(back.model_folders[0].id, "main")
        self.assertEqual(back.model_folders[0].kinds["checkpoints"],
                         ["checkpoints"])
        # An empty list is not written, so older builds read the file.
        self.assertNotIn("model_folders", StoreFile().to_dict())
        # A store without the field loads with no folders.
        self.assertEqual(StoreFile.from_dict(
            {"schema_version": 1}).model_folders, [])

    def test_store_file_refuses_bad_folders(self):
        with self.assertRaises(StoreError):
            StoreFile.from_dict(
                {"schema_version": 1,
                 "model_folders": [{"id": "Bad Id", "label": "x",
                                    "path": "/tmp"}]})


class DraftTest(unittest.TestCase):
    def test_presets_are_the_eleven_builtin_drafts(self):
        keys = [p.key for p in PRESETS]
        self.assertEqual(
            keys, ["openai", "openrouter", "bfl", "seedance",
                   "vertex-express", "artcraft-omni", "ollama",
                   "lmstudio", "llamacpp", "comfyui", "custom"])
        by_key = {p.key: p for p in PRESETS}
        self.assertTrue(by_key["openai"].needs_key)
        self.assertFalse(by_key["ollama"].needs_key)
        self.assertEqual(by_key["ollama"].endpoint,
                         "http://127.0.0.1:11434/v1")
        self.assertEqual(by_key["custom"].endpoint, "")

    def test_preset_models_are_unverified_and_only_where_documented(self):
        by_key = {p.key: p for p in PRESETS}
        omni = preset_models(by_key["artcraft-omni"])
        self.assertEqual(
            [m.id.as_str() for m in omni],
            ["nano_banana", "seedance_2p0"])
        self.assertEqual(
            [m.capabilities[0].capability.as_str() for m in omni],
            ["image.generate", "video.generate"])
        vertex = preset_models(by_key["vertex-express"])
        self.assertEqual([m.id.as_str() for m in vertex],
                         ["gemini-2.5-flash-image"])
        self.assertEqual(preset_models(by_key["openai"]), [])
        self.assertEqual(preset_models(by_key["ollama"]), [])

    def test_set_models_refuses_a_late_result_for_another_address(self):
        draft = ConnectionDraft.from_preset(
            next(p for p in PRESETS if p.key == "ollama"))
        self.assertFalse(draft.set_models(
            "http://127.0.0.1:9999/v1", [_text_model()]))
        self.assertEqual(draft.models, [])
        self.assertTrue(draft.set_models(
            "http://127.0.0.1:11434/v1", [_text_model()]))
        self.assertEqual(draft.models_for, "http://127.0.0.1:11434/v1")
        # Editing the address afterwards drops the provenance: the
        # models no longer fit the draft, even though the new address
        # is still loopback.
        draft.endpoint = "http://127.0.0.1:8080/v1"
        self.assertEqual(draft.locality(), Locality.LOCAL)
        prepared = draft.prepare([])
        self.assertEqual(prepared.connection.models, [])

    def test_locality_comes_from_the_address_never_a_choice(self):
        draft = ConnectionDraft.from_preset(
            next(p for p in PRESETS if p.key == "ollama"))
        self.assertEqual(draft.locality(), Locality.LOCAL)
        draft.endpoint = "https://api.openai.com/v1"
        self.assertEqual(draft.locality(), Locality.REMOTE)
        draft.endpoint = "not an address"
        self.assertIsNone(draft.locality())

    def test_prepare_names_the_field_that_is_wrong(self):
        draft = ConnectionDraft(label="", provider="openai",
                                endpoint="https://api.openai.com/v1")
        with self.assertRaises(DraftError) as ctx:
            draft.prepare([])
        self.assertEqual(ctx.exception.field, Field.LABEL)
        draft.label = "x" * 81
        with self.assertRaises(DraftError) as ctx:
            draft.prepare([])
        self.assertEqual(ctx.exception.field, Field.LABEL)
        draft.label = "OpenAI"
        draft.endpoint = "not an address"
        with self.assertRaises(DraftError) as ctx:
            draft.prepare([])
        self.assertEqual(ctx.exception.field, Field.ADDRESS)
        draft.endpoint = "http://api.openai.com/v1"
        with self.assertRaises(DraftError) as ctx:
            draft.prepare([])
        # http:// to a hosted address is refused by the endpoint rule.
        self.assertEqual(ctx.exception.field, Field.ADDRESS)

    def test_prepare_builds_the_connection_and_keeps_the_key_change(self):
        draft = ConnectionDraft.from_preset(
            next(p for p in PRESETS if p.key == "ollama"))
        draft.label = "Local Ollama"
        prepared = draft.prepare([])
        self.assertEqual(prepared.connection.label, "Local Ollama")
        self.assertEqual(prepared.connection.locality, Locality.LOCAL)
        self.assertIsInstance(prepared.key_change, KeyChange)
        self.assertEqual(prepared.key_change.kind, "keep")
        # A typed key becomes a Set change on a keyed preset; the
        # saved connection carries only the secret's locator.
        keyed = ConnectionDraft.from_preset(
            next(p for p in PRESETS if p.key == "openai"))
        keyed.label = "OpenAI"
        keyed.key = "sk-test-key"
        prepared = keyed.prepare([])
        self.assertEqual(prepared.key_change.kind, "set")
        self.assertEqual(prepared.key_change.value, "sk-test-key")
        self.assertEqual(prepared.connection.auth.kind, "api_key")
        self.assertEqual(prepared.connection.auth.secret, "openai.key")

    def test_prepare_refuses_a_key_with_spaces(self):
        draft = ConnectionDraft.from_preset(
            next(p for p in PRESETS if p.key == "openai"))
        draft.label = "OpenAI"
        draft.key = "has a space"
        with self.assertRaises(DraftError) as ctx:
            draft.prepare([])
        self.assertEqual(ctx.exception.field, Field.KEY)


class SegmentTest(unittest.TestCase):
    def _segment_request(self) -> Request:
        return Request.new(CapabilityId.parse(CAPABILITY)).with_input(
            InputRole.SOURCE, Payload.inline("image/png", _dot_png()))

    def test_with_clicks_round_trips_and_validates(self):
        request = with_clicks(self._segment_request(),
                              [Click(0.5, 0.4, True),
                               Click(0.1, 0.9, False)])
        got = clicks(request)
        self.assertEqual(got, [Click(0.5, 0.4, True),
                               Click(0.1, 0.9, False)])
        self.assertIsNone(area(request))
        with self.assertRaises(AiError):
            clicks(with_clicks(self._segment_request(),
                               [Click(0.5, 0.4, True)] * (MAX_POINTS + 1)))
        with self.assertRaises(AiError):
            clicks(with_clicks(self._segment_request(),
                               [Click(1.5, 0.4, True)]))
        with self.assertRaises(AiError):
            clicks(with_clicks(self._segment_request(),
                               [Click(0.5, -0.1, True)]))

    def test_with_area_round_trips_and_validates(self):
        request = with_area(self._segment_request(),
                            Area(0.1, 0.1, 0.6, 0.9))
        self.assertEqual(area(request), Area(0.1, 0.1, 0.6, 0.9))
        self.assertEqual(clicks(request), [])
        with self.assertRaises(AiError):
            area(with_area(self._segment_request(),
                           Area(0.6, 0.1, 0.1, 0.9)))
        with self.assertRaises(AiError):
            area(with_area(self._segment_request(),
                           Area(0.1, 0.1, 1.5, 0.9)))

    def test_description_reads_the_prompt_input(self):
        request = self._segment_request().with_input(
            InputRole.PROMPT, Payload.text("  the cat  "))
        self.assertEqual(description(request), "  the cat  ")
        self.assertIsNone(description(self._segment_request()))
        blank = self._segment_request().with_input(
            InputRole.PROMPT, Payload.text("   "))
        self.assertIsNone(description(blank))

    def test_found_output_and_read_found_round_trip(self):
        found = Found(matched=True, score=0.87, name="cat")
        output = found_output(found)
        self.assertEqual(output["payload"]["media_type"],
                         FOUND_MEDIA_TYPE)
        read = read_found([output])
        self.assertEqual(read, found)
        # A name that is too long, a score outside 0..1 and control
        # characters are dropped, not passed to the screen.
        bad = Found(matched=True, score=2.0, name="x" * (MAX_NAME_CHARS + 1))
        self.assertEqual(read_found([found_output(bad)]),
                         Found(matched=True, score=None, name=None))
        self.assertIsNone(read_found([{"type": "output",
                                       "payload": Payload.text("hi")
                                       .to_dict()}]))

    def test_mask_finds_the_first_inline_png_and_skips_handles(self):
        handle = {"type": "output",
                  "payload": Payload.handle(
                      "media://x", "image/png").to_dict()}
        png_payload = Payload.inline("image/png", _dot_png())
        png_output = {"type": "output", "payload": png_payload.to_dict()}
        self.assertIsNone(mask([handle]))
        got = mask([handle, png_output])
        self.assertEqual(got["media_type"], MASK_MEDIA_TYPE)
        self.assertEqual(got["data"], png_payload.to_dict()["data"])


class SecretsTest(unittest.TestCase):
    def test_memory_store_round_trips_and_misses(self):
        store = MemoryStore()
        self.assertIsNone(store.get("OPENAI_API_KEY"))
        store.set("OPENAI_API_KEY", "sk-test")
        self.assertEqual(store.get("OPENAI_API_KEY"), "sk-test")
        self.assertEqual(store.name, "memory")

    def test_env_store_reads_the_environment(self):
        store = EnvStore()
        # The variable name is the reference flattened to upper case
        # under the CRAFT_AI_SECRET_ prefix.
        self.assertEqual(EnvStore.var_name("craftai-test-key"),
                         "CRAFT_AI_SECRET_CRAFTAI_TEST_KEY")
        os.environ["CRAFT_AI_SECRET_CRAFTAI_TEST_KEY"] = "env-value"
        try:
            self.assertEqual(store.get("CRAFTAI_TEST_KEY"), "env-value")
        finally:
            del os.environ["CRAFT_AI_SECRET_CRAFTAI_TEST_KEY"]
        self.assertIsNone(store.get("CRAFTAI_TEST_KEY"))
        self.assertEqual(store.name, "environment")
        # The environment is read-only: writes are refused, never
        # silently dropped.
        with self.assertRaises(AiError):
            store.set("CRAFTAI_TEST_KEY", "x")

    def test_layered_store_takes_the_first_hit(self):
        first = MemoryStore()
        first.set("KEY", "first")
        second = MemoryStore()
        second.set("KEY", "second")
        second.set("OTHER", "other")
        layered = LayeredStore([first, second])
        self.assertEqual(layered.get("KEY"), "first")
        self.assertEqual(layered.get("OTHER"), "other")
        self.assertIsNone(layered.get("MISSING"))
        self.assertEqual(layered.layer_names(), ["memory", "memory"])

    def test_secret_for_reads_the_ref_and_names_the_connection(self):
        store = MemoryStore()
        store.set("sk-openai", "sk-live")
        keyed = Connection("openai-conn", "OpenAI", "openai",
                           endpoint="https://api.openai.com/v1",
                           auth=Auth.api_key("sk-openai"))
        self.assertEqual(secret_for(store, keyed), "sk-live")
        self.assertIsNone(secret_for(store, _mock_connection()))
        missing = Connection("openai-conn", "OpenAI", "openai",
                             endpoint="https://api.openai.com/v1",
                             auth=Auth.api_key("no-such-key"))
        with self.assertRaises(AiError) as ctx:
            secret_for(store, missing)
        self.assertIn("OpenAI", str(ctx.exception))
        sub = Connection("sub-conn", "Sub", "openai",
                         endpoint="https://api.openai.com/v1",
                         auth=Auth.subscription("oauth", "no-such-key"))
        with self.assertRaises(AiError) as ctx:
            secret_for(store, sub)
        self.assertIn("sign-in", str(ctx.exception))


class SharedStoreTest(unittest.TestCase):
    def test_missing_file_reads_as_an_empty_store(self):
        tmp = Path(self._tmpdir())
        shared = SharedStore(tmp / "ai.json")
        store = shared.load()
        self.assertEqual(store.connections, [])
        self.assertIsNone(shared.load_existing())

    def test_update_writes_atomically_and_preserves_unknown_fields(self):
        tmp = Path(self._tmpdir())
        path = tmp / "ai.json"
        path.write_text(json.dumps(
            {"schema_version": 1, "connections": [],
             "future_field": {"kept": True}}), encoding="utf-8")
        shared = SharedStore(path)
        shared.update(lambda store: store.connections.append(
            _mock_connection()))
        data = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(data["future_field"], {"kept": True})
        self.assertEqual(len(data["connections"]), 1)
        self.assertEqual(shared.load().connections[0].id.as_str(),
                         "mock-conn")

    def test_newer_schema_is_never_overwritten(self):
        tmp = Path(self._tmpdir())
        path = tmp / "ai.json"
        path.write_text(json.dumps(
            {"schema_version": 9999, "connections": []}),
            encoding="utf-8")
        shared = SharedStore(path)
        with self.assertRaises(SharedError):
            shared.load()
        with self.assertRaises(SharedError):
            shared.update(lambda store: None)
        # The file is untouched.
        self.assertEqual(
            json.loads(path.read_text(encoding="utf-8"))["schema_version"],
            9999)

    def test_replace_round_trips_the_store(self):
        tmp = Path(self._tmpdir())
        shared = SharedStore(tmp / "ai.json")
        shared.replace(StoreFile(connections=[_mock_connection()]))
        self.assertEqual(shared.load().connections[0].label, "Mock")

    def test_history_file_counts_skipped_lines(self):
        tmp = Path(self._tmpdir())
        history = HistoryFile(tmp / "history.jsonl")
        history.append({"app": "craftai-test", "role": "inpaint"})
        history.append({"app": "craftai-test", "role": "segment"})
        (tmp / "history.jsonl").write_text(
            (tmp / "history.jsonl").read_text(encoding="utf-8")
            + "not json\n", encoding="utf-8")
        read = history.read()
        self.assertEqual(len(read.records), 2)
        self.assertEqual(read.skipped, 1)
        history.clear()
        self.assertEqual(history.read().records, [])

    def _tmpdir(self) -> str:
        import tempfile
        return tempfile.mkdtemp(prefix="craftai-shared-")


class FactoryTest(unittest.TestCase):
    def test_standard_knows_the_openai_family_and_the_local_providers(self):
        factory = ProviderFactory.standard()
        for name in list(OPENAI_FAMILY) + ["mock", "local-motion",
                                           "artcraft-omni", "bfl",
                                           "seedance", "vertex-express",
                                           "sam3", "comfyui",
                                           "declarative"]:
            self.assertTrue(factory.knows(name), name)
        self.assertTrue(factory.knows("OLLAMA"))
        self.assertFalse(factory.knows("fal"))

    def test_build_refuses_an_unknown_provider_by_name(self):
        factory = ProviderFactory.standard()
        ghost = Connection("ghost", "Ghost", "fal",
                           endpoint="https://example.com")
        with self.assertRaises(AiError) as ctx:
            factory.build(ghost, None)
        self.assertIn("fal", str(ctx.exception))

    def test_build_returns_the_mock_for_the_mock_connection(self):
        factory = ProviderFactory.standard()
        provider = factory.build(_mock_connection(), None)
        self.assertIsInstance(provider, MockProvider)
        self.assertEqual([m.id.as_str() for m in provider.models()],
                         ["mock-text"])

    def test_build_returns_the_local_motion_provider(self):
        factory = ProviderFactory.standard()
        local = Connection("local-motion", "Local motion", "local-motion",
                           locality=Locality.LOCAL,
                           models=[LocalMotionProvider().models()[0]])
        provider = factory.build(local, None)
        self.assertIsInstance(provider, LocalMotionProvider)

    def test_register_and_unregister_round_trip(self):
        factory = ProviderFactory.standard()
        factory.register("echo", lambda c, s: MockProvider([_text_model()]))
        self.assertTrue(factory.knows("echo"))
        factory.unregister("echo")
        self.assertFalse(factory.knows("echo"))

    def test_accelerators_are_declared_per_provider(self):
        factory = ProviderFactory.standard()
        self.assertIsNone(factory.accelerators("mock"))
        factory.declare_accelerators("mock", ["cpu"])
        self.assertEqual(factory.accelerators("mock"), ["cpu"])

    def test_probing_uses_short_timeouts(self):
        probing = ProviderFactory.probing()
        standard = ProviderFactory.standard()
        conn = Connection("ollama", "Ollama", "ollama",
                          endpoint="http://127.0.0.1:11434/v1",
                          locality=Locality.LOCAL,
                          models=[_text_model()])
        probe_provider = probing.build(conn, None)
        standard_provider = standard.build(conn, None)
        self.assertIsInstance(probe_provider, OpenAiCompatProvider)
        self.assertLess(probe_provider.timeout,
                        standard_provider.timeout)

    def test_check_provider_passes_the_mock_and_names_a_failure(self):
        report = check_provider(MockProvider([_text_model()]),
                                _text_model(), _text_request())
        self.assertTrue(report["passed"])
        self.assertEqual(report["failed"], [])
        self.assertIn("lifecycle", report["checks"])
        # The raw conformance suite agrees with the report.
        run_lifecycle(MockProvider([_text_model()]), _text_model(),
                      _text_request())

        class BrokenProvider(MockProvider):
            def poll(self, job_id: str):
                raise AiError.invalid("poll always fails")

        report = check_provider(BrokenProvider([_text_model()]),
                                _text_model(), _text_request())
        self.assertFalse(report["passed"])
        self.assertIn("lifecycle", report["failed"])

    def test_register_checked_refuses_a_provider_that_fails_conformance(self):
        factory = ProviderFactory.standard()

        class BrokenProvider(MockProvider):
            def poll(self, job_id: str):
                raise AiError.invalid("poll always fails")

        with self.assertRaises(AiError) as ctx:
            register_checked(
                factory, "broken",
                lambda c, s: BrokenProvider([_text_model()]),
                probe=(_text_model(), _text_request()))
        self.assertIn("lifecycle", str(ctx.exception))
        self.assertFalse(factory.knows("broken"))
        # A provider that passes conformance registers fine.
        register_checked(
            factory, "echo",
            lambda c, s: MockProvider([_text_model()]),
            probe=(_text_model(), _text_request()))
        self.assertTrue(factory.knows("echo"))


class OpenAiAdapterTest(unittest.TestCase):
    """The OpenAI-compatible adapter against a real local server."""

    def setUp(self):
        self.server = _ChatServer()
        self.server.start()
        self.addCleanup(self.server.stop)

    def test_adapter_runs_a_chat_job_through_the_real_http_path(self):
        factory = ProviderFactory.standard()
        conn = Connection("local-llm", "Local LLM", "openai-compatible",
                          endpoint=self.server.base_url,
                          locality=Locality.LOCAL,
                          models=[_text_model("local-text")])
        provider = factory.build(conn, None)
        self.assertIsInstance(provider, OpenAiCompatProvider)
        request = Request.new(CapabilityId.parse("text.generate")) \
            .with_input(InputRole.PROMPT, Payload.text("say hi"))
        job_id = provider.submit(_text_model("local-text"), request)
        snapshot = provider.poll(job_id)
        self.assertEqual(snapshot["state"]["kind"], "succeeded")
        outputs = [e for e in snapshot["events"]
                   if e["type"] == "output"]
        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0]["payload"]["text"], "hello from local")

    def test_declarative_spec_builds_a_provider_from_a_file(self):
        # The spec file is <dir>/<connection id>.json and its base
        # must be the connection's address; the test server's port
        # is dynamic, so the spec is written for it here.
        import tempfile
        specs_dir = Path(tempfile.mkdtemp(prefix="craftai-specs-"))
        spec_text = (FIXTURES / "spec-conn.json").read_text(
            encoding="utf-8")
        spec = json.loads(spec_text)
        spec["base"] = self.server.base_url
        (specs_dir / "spec-conn.json").write_text(
            json.dumps(spec), encoding="utf-8")
        factory = ProviderFactory.standard()
        factory.set_declarative_specs(specs_dir)
        conn = Connection("spec-conn", "Spec LLM", "declarative",
                          endpoint=self.server.base_url,
                          locality=Locality.LOCAL,
                          models=[_text_model("spec-text")])
        provider = factory.build(conn, None)
        request = Request.new(CapabilityId.parse("text.generate")) \
            .with_input(InputRole.PROMPT, Payload.text("say hi"))
        job_id = provider.submit(_text_model("spec-text"), request)
        snapshot = provider.poll(job_id)
        self.assertEqual(snapshot["state"]["kind"], "succeeded")
        outputs = [e for e in snapshot["events"]
                   if e["type"] == "output"]
        self.assertEqual(outputs[0]["payload"]["text"],
                         "hello from local")

    def test_declarative_spec_refuses_a_different_address(self):
        factory = ProviderFactory.standard()
        factory.set_declarative_specs(FIXTURES)
        conn = Connection("spec-conn", "Spec LLM", "declarative",
                          endpoint="http://127.0.0.1:9999/v1",
                          locality=Locality.LOCAL,
                          models=[_text_model("spec-text")])
        with self.assertRaises(AiError) as ctx:
            factory.build(conn, None)
        self.assertIn("different address", str(ctx.exception))


class _ChatServer:
    """A tiny OpenAI-shaped chat server on loopback, for adapter tests."""

    def __init__(self):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802 - stdlib name
                length = int(self.headers.get("Content-Length") or 0)
                self.rfile.read(length)
                body = json.dumps({
                    "choices": [{"message": {"content": "hello from local"},
                                 "finish_reason": "stop"}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):  # noqa: D102 - quiet
                pass

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self._httpd.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}/v1"
        self._thread = threading.Thread(target=self._httpd.serve_forever,
                                        daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._httpd.shutdown()
        self._httpd.server_close()


class ClientTest(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmp = Path(tempfile.mkdtemp(prefix="craftai-client-"))
        self.client = _client(self.tmp)

    def test_status_ready_names_the_binding(self):
        status = self.client.status(_role())
        self.assertIsInstance(status, RoleStatus)
        self.assertTrue(status.is_ready)
        ready = status.ready
        self.assertIsInstance(ready, RoleReady)
        self.assertEqual(ready.connection, "mock-conn")
        self.assertEqual(ready.connection_label, "Mock")
        self.assertEqual(ready.model, "mock-text")
        self.assertEqual(ready.locality, Locality.LOCAL)

    def test_status_unavailable_for_an_unbound_role(self):
        status = self.client.status(_role("unknown-role"))
        self.assertFalse(status.is_ready)
        self.assertIn("bind", status.reason.lower())

    def test_status_unavailable_when_no_adapter_knows_the_provider(self):
        import tempfile
        tmp = Path(tempfile.mkdtemp(prefix="craftai-noadapter-"))
        shared = SharedStore(tmp / "ai.json")
        shared.replace(StoreFile(
            connections=[Connection(
                "openai-conn", "OpenAI", "openai",
                endpoint="https://api.openai.com/v1",
                models=[_text_model()])],
            defaults=[Binding("inpaint", "openai-conn", "mock-text")]))
        factory = ProviderFactory()
        factory.register("mock", lambda c, s: MockProvider([_text_model()]))
        client = AiClient("craftai-test", shared, MemoryStore(),
                          factory=factory)
        status = client.status(_role())
        self.assertFalse(status.is_ready)
        self.assertEqual(status.reason, NO_ADAPTER)

    def test_bind_refuses_a_connection_or_model_the_store_does_not_hold(self):
        with self.assertRaises(AiError):
            self.client.bind(_role(), "ghost-conn", "mock-text")
        with self.assertRaises(AiError):
            self.client.bind(_role(), "mock-conn", "ghost-model")

    def test_plan_needs_consent_until_agreed_then_start_runs(self):
        request = _text_request()
        plan = self.client.plan(_role(), request)
        self.assertIsInstance(plan, Plan)
        self.assertEqual(plan.connection_label(), "Mock")
        self.assertEqual(plan.model().as_str(), "mock-text")
        ledger = ConsentLedger()
        # A local connection never asks: the disclosure is local.
        self.assertTrue(plan.disclosure().local)
        self.assertFalse(plan.needs_consent(ledger))
        job = self.client.start(plan, ledger)
        self.assertIsInstance(job, Job)
        finished = job.wait(0.05, 5.0)
        self.assertIsInstance(finished, Finished)
        self.assertEqual(finished.state.kind, "succeeded")

    def test_start_refuses_when_consent_is_missing_for_a_hosted_route(self):
        import tempfile
        tmp = Path(tempfile.mkdtemp(prefix="craftai-hosted-"))
        shared = SharedStore(tmp / "ai.json")
        hosted = Connection("hosted-conn", "Hosted", "mock",
                            endpoint="https://example.com",
                            locality=Locality.REMOTE,
                            models=[_text_model()])
        shared.replace(StoreFile(
            connections=[hosted],
            defaults=[Binding("inpaint", "hosted-conn", "mock-text")]))
        client = AiClient("craftai-test", shared, MemoryStore(),
                          factory=ProviderFactory.standard())
        plan = client.plan(_role(), _text_request())
        ledger = ConsentLedger()
        self.assertTrue(plan.needs_consent(ledger))
        with self.assertRaises(AiError) as ctx:
            client.start(plan, ledger)
        self.assertIn("consent", str(ctx.exception).lower())
        ledger.agree(plan.disclosure(), ConsentTerm.until_revoked(), 1000)
        job = client.start(plan, ledger)
        self.assertEqual(job.wait(0.05, 5.0).state.kind, "succeeded")

    def test_start_refuses_a_request_the_bound_model_cannot_run(self):
        request = Request.new(CapabilityId.parse("image.generate")) \
            .with_input(InputRole.PROMPT, Payload.text("a cat"))
        # The plan itself refuses: the request is for a capability
        # the role does not have, so nothing is ever sent.
        with self.assertRaises(AiError):
            self.client.plan(_role(), request)

    def test_choices_and_picks_follow_the_binding(self):
        choices = self.client.choices(_role())
        self.assertTrue(any(isinstance(c, Choice) for c in choices))
        self.assertTrue(any(c.model.as_str() == "mock-text"
                            for c in choices))
        picks = self.client.picks(_role())
        self.assertTrue(any(isinstance(p, PickedChoice) for p in picks))
        self.assertEqual(picks[0].choice.model.as_str(), "mock-text")

    def test_recommended_offers_the_catalog_pick_for_an_unready_role(self):
        from nomorals.craftlibs.ai.picks import (
            Catalog,
            CatalogEntry,
            Pick,
        )
        catalog = Catalog(entries=[CatalogEntry(
            "text.generate", Pick.RECOMMENDED, "mock", "mock-text")])
        client = _client(self.tmp).with_catalog(catalog)
        role = _role("summarize")
        self.assertFalse(client.status(role).is_ready)
        recommended = client.recommended([role])
        self.assertEqual(len(recommended), 1)
        self.assertEqual(recommended[0][0].as_str(), "summarize")
        self.assertEqual(recommended[0][1].model.as_str(), "mock-text")
        # A role that already runs is never offered a replacement.
        self.assertEqual(client.recommended([_role()]), [])

    def test_clear_binding_restores_the_launcher_default(self):
        self.client.bind(_role(), "mock-conn", "mock-text")
        self.client.clear_binding("inpaint")
        status = self.client.status(_role())
        self.assertTrue(status.is_ready)
        self.assertEqual(status.ready.model, "mock-text")

    def test_history_records_the_run_and_clears(self):
        plan = self.client.plan(_role(), _text_request())
        job = self.client.start(plan, ConsentLedger())
        job.wait(0.05, 5.0)
        log = self.client.history()
        self.assertIsInstance(log, HistoryLog)
        self.assertEqual(len(log.entries), 1)
        self.assertEqual(log.entries[0].start.role_id, "inpaint")
        self.client.clear_history()
        self.assertEqual(self.client.history().entries, [])

    def test_conformance_runs_against_the_real_mock(self):
        report = self.client.conformance(_role())
        self.assertTrue(report["passed"])

    def test_conformance_passes_the_local_motion_provider(self):
        # The local provider answers synchronously, so its jobs are
        # already finished when cancel is called — cancelling a
        # finished job must leave it succeeded, never flip a terminal
        # state to cancelled (the transition law).
        provider = LocalMotionProvider()
        model = provider.models()[0]
        request = Request.new(
            CapabilityId.parse("motion.suggest")).with_input(
                InputRole.PROMPT, Payload.text("a ball bouncing"))
        report = check_provider(provider, model, request)
        self.assertTrue(report["passed"], report)

    def test_cancel_after_finish_leaves_the_job_succeeded(self):
        from nomorals.craftlibs.ai.job import JobState

        provider = LocalMotionProvider()
        model = provider.models()[0]
        request = Request.new(
            CapabilityId.parse("motion.suggest")).with_input(
                InputRole.PROMPT, Payload.text("a ball bouncing"))
        job_id = provider.submit(model, request)
        first = provider.poll(job_id)
        self.assertEqual(first["state"]["kind"], "succeeded")
        provider.cancel(job_id)
        provider.cancel(job_id)
        second = provider.poll(job_id)
        self.assertEqual(second["state"]["kind"], "succeeded")
        self.assertEqual(second["events"], [])
        self.assertTrue(JobState.from_dict(
            first["state"]).can_become(
                JobState.from_dict(second["state"])))


class ToolDeskTest(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmp = Path(tempfile.mkdtemp(prefix="craftai-desk-"))
        self.client = _client(self.tmp)
        self.desk = ToolDesk(self.client, self.client.shared,
                             builtin_ids=("craftlibs", "craftfonts"))

    def test_view_shows_the_current_choice(self):
        view = self.desk.view(_role())
        self.assertTrue(any(c.model.as_str() == "mock-text"
                            for c in view.choices))
        # The launcher default for the role.
        self.assertEqual(view.bound, ("mock-conn", "mock-text"))
        self.assertTrue(view.status.is_ready)

    def test_choose_and_choose_if_unchosen(self):
        role = _role("summarize")
        self.assertTrue(self.desk.choose_if_unchosen(
            role, "mock-conn", "mock-text"))
        # A choice already made is never overwritten.
        self.assertFalse(self.desk.choose_if_unchosen(
            role, "mock-conn", "mock-text"))
        self.assertEqual(self.desk.view(role).bound,
                         ("mock-conn", "mock-text"))
        self.desk.choose(role, "mock-conn", "mock-text")
        self.assertEqual(self.desk.view(role).bound,
                         ("mock-conn", "mock-text"))

    def test_auto_bind_offer_and_accept_recommended(self):
        self.desk.clear(_role())
        offers = self.desk.auto_bind_offer([_role()])
        self.assertEqual(len(offers), 1)
        self.assertEqual(offers[0][1].model.as_str(), "mock-text")
        accepted = self.desk.accept_recommended([_role()])
        self.assertIsInstance(accepted, Accepted)
        self.assertEqual([r.as_str() for r in accepted.bound], ["inpaint"])
        self.assertEqual(accepted.skipped, [])
        self.assertEqual(self.desk.view(_role()).bound,
                         ("mock-conn", "mock-text"))

    def test_register_tool_refuses_builtin_ids_and_duplicates(self):
        with self.assertRaises(AiError):
            self.desk.register_tool(_role("craftlibs"))
        self.desk.register_tool(_role("my-tool"))
        with self.assertRaises(AiError):
            self.desk.register_tool(_role("my-tool"))
        self.assertIn("my-tool", [t.id.as_str()
                                  for t in self.desk.custom_tools()])
        self.desk.remove_tool("my-tool")
        self.assertNotIn("my-tool", [t.id.as_str()
                                     for t in self.desk.custom_tools()])

    def test_missing_model_names_the_gap(self):
        self.assertIsNone(self.desk.missing_model(_role()))
        self.assertIsNotNone(self.desk.missing_model(_role("no-such-tool")))

    def test_sample_run_passes_on_the_mock(self):
        started = self.desk.sample(_role(), _text_request())
        self.assertIsInstance(started, TestStart)
        self.assertTrue(started.is_started)
        report = ToolDesk.finish(_role(), started.job, 5.0)
        self.assertIsInstance(report, SampleReport)
        self.assertTrue(report.passed)
        self.assertIn("text result", report.summary)

    def test_sample_run_needs_consent_on_a_hosted_route(self):
        import tempfile
        tmp = Path(tempfile.mkdtemp(prefix="craftai-desk-hosted-"))
        shared = SharedStore(tmp / "ai.json")
        hosted = Connection("hosted-conn", "Hosted", "mock",
                            endpoint="https://example.com",
                            locality=Locality.REMOTE,
                            models=[_text_model()])
        shared.replace(StoreFile(
            connections=[hosted],
            defaults=[Binding("inpaint", "hosted-conn", "mock-text")]))
        client = AiClient("craftai-test", shared, MemoryStore(),
                          factory=ProviderFactory.standard())
        desk = ToolDesk(client, shared)
        started = desk.sample(_role(), _text_request())
        self.assertFalse(started.is_started)
        self.assertEqual(started.connection_label, "Hosted")


class HttpPolicyTest(unittest.TestCase):
    def test_tls_context_verifies_against_the_platform_store(self):
        import ssl
        context = tls_context()
        self.assertIsInstance(context, ssl.SSLContext)
        self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)
        self.assertEqual(tls_cafile(), _tls_cafile_expected())

    def test_loopback_takes_no_proxy_whatever_the_environment_says(self):
        self.assertEqual(loopback_proxy(), "")
        self.assertEqual(proxy_for("http://127.0.0.1:11434/v1"), "")
        self.assertEqual(proxy_for("http://localhost:8080"), "")
        self.assertEqual(proxy_for("http://[::1]:8080"), "")

    def test_hosted_keeps_the_environment_proxy(self):
        from nomorals.core.http import get_default_proxy, set_default_proxy
        old = get_default_proxy()
        saved = {name: os.environ.get(name)
                 for name in ("HTTPS_PROXY", "HTTP_PROXY")}
        try:
            set_default_proxy("http://proxy:3128")
            self.assertEqual(hosted_proxy(), "http://proxy:3128")
            self.assertEqual(
                proxy_for("https://api.openai.com/v1"),
                "http://proxy:3128")
            set_default_proxy("")
            # The box itself sets HTTP_PROXY (egress); clear both
            # variables so "no proxy" is really what is tested.
            for name in saved:
                os.environ.pop(name, None)
            os.environ["HTTPS_PROXY"] = "http://env-proxy:8080"
            try:
                self.assertEqual(hosted_proxy(), "http://env-proxy:8080")
            finally:
                del os.environ["HTTPS_PROXY"]
            self.assertEqual(hosted_proxy(), "")
        finally:
            set_default_proxy(old)
            for name, value in saved.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    def test_agent_for_loopback_forces_direct_even_with_a_default_proxy(self):
        from nomorals.core.http import get_default_proxy, set_default_proxy
        old = get_default_proxy()
        set_default_proxy("http://proxy:3128")
        try:
            agent = agent_for("http://127.0.0.1:11434/v1")
            self.assertEqual(agent._effective_proxy(
                "http://127.0.0.1:11434/v1"), "")
            self.assertTrue(agent.verify_tls)
        finally:
            set_default_proxy(old)

    def test_agent_for_reaches_a_real_loopback_server(self):
        server = _ChatServer()
        server.start()
        try:
            agent = agent_for(server.base_url)
            response = agent.post_json(
                f"{server.base_url}/chat/completions",
                {"messages": []})
            self.assertEqual(response.status, 200)
            self.assertIn("hello from local",
                          response.body.decode("utf-8"))
        finally:
            server.stop()

    def test_scan_agents_finds_a_bypass_and_clears_the_ai_path(self):
        import tempfile
        tmp = Path(tempfile.mkdtemp(prefix="craftai-scan-"))
        bad = tmp / "bad.py"
        bad.write_text("import urllib.request\n"
                       "urllib.request.urlopen(req)\n", encoding="utf-8")
        violations = scan_agents([tmp])
        self.assertEqual(len(violations), 1)
        self.assertIn("bad.py", violations[0])
        tree = Path(__file__).resolve().parents[1] / "nomorals"
        clean = scan_agents([tree / "llm", tree / "craftlibs" / "ai"])
        self.assertEqual(clean, [])


def _tls_cafile_expected():
    from nomorals.core.http import tls_cafile
    return tls_cafile()


class BenchmarkRewireTest(unittest.TestCase):
    def test_fetch_completion_uses_the_shared_http_client(self):
        from nomorals.core.http import HttpClient
        server = _ChatServer()
        server.start()
        try:
            client = _LlamaServerClient(server.base_url)
            self.assertIsInstance(client._http, HttpClient)
            self.assertTrue(client._http.force_direct)
            self.assertTrue(client._http.allow_private_ips)
            self.assertEqual(client.generate("say hi"), "hello from local")
        finally:
            server.stop()
        import inspect

        import nomorals.llm.benchmark as benchmark_module
        source = inspect.getsource(benchmark_module)
        self.assertNotIn("urlopen", source)
        self.assertIn("HttpClient", source)


class ParityTest(unittest.TestCase):
    def test_craftai_parity_passes_every_case(self):
        from nomorals.craftlibs.ai.parity import run_craftai_case
        from nomorals.craftlibs.parity import run_parity
        result = run_parity(FIXTURES.parent.parent.parent
                            / "nomorals" / "craftlibs"
                            / "craftai-parity.toml",
                            run_case=run_craftai_case)
        self.assertEqual(result["failed"], [], result["failures"])
        self.assertGreater(result["passed"], 0)

    def test_missing_parity_toml_fails_never_skips(self):
        from nomorals.craftlibs.parity import run_parity
        result = run_parity("/nonexistent/craftai-parity.toml")
        self.assertEqual(result["passed"], 0)
        self.assertEqual(result["failed"], ["<toml>"])


class AttributionTest(unittest.TestCase):
    def test_craftai_attribution_verifies_every_unit(self):
        from nomorals.craftlibs.attribution import verify_attribution
        result = verify_attribution(
            Path(__file__).resolve().parents[1] / "nomorals" / "craftlibs"
            / "craftai-attribution.toml")
        self.assertEqual(result["failed"], [], result["failed"])
        self.assertEqual(result["skipped"], [])


class ToolCountTest(unittest.TestCase):
    def test_tool_count_is_786_after_craftai(self):
        # No new tool is added by this lane: the craftlibs tool's ai
        # action is extended in place. 733 at base 64821b6c, 734 after
        # the craft-fonts port (lane 114) shipped first and added the
        # craftfonts tool, 735 after the indd-utils port (lane 117)
        # added the inddutils tool, 748 after the auth-plugin wiring
        # added 5 tools on top of the bughunter port (lane 121, 8 tools). 749 after the mocap port (lane 125) added the mocap tool. 774 after the secprobe wire-up added 25 tools (scope_load, scope_check, param_mine, param_confirm, inject_probe, jwt_attack, graphql_probe, graphql_amplify, race_wave, cors_probe, csrf_poc, plan_next, plan_add, plan_resolve, leaks_harvest, dossier_build, ban_report, ban_state, ban_set_state, ban_appeal, report_capture, report_amplify, replay_run, replay_diagnose, sec_chain). 775 after this lane (B.19) added scene_spawn. 776 after this lane (B.20) added cloudcam. 777 after this lane (B.22) added show. 778 after this lane (B.26) added scene_project. 779 after this lane (B.23) added facemcap. 780 after this lane (B.25) added facemesh. 781 after this lane (B.27) added vtuber. 782 after this lane (C.2) registered tonal_g2p; 783 after this lane (C.1) added smarthome; 784 after this lane (C.4) added camwall; 785 after this lane (C.8) added robot; 786 after this lane (audit N4) added build_swarm; 787 after gapfix74 added museum_library.
        registry = ToolRegistry(SimpleNamespace())
        registry.register_builtins()
        self.assertEqual(len(registry._tools), 787)
        self.assertIn("craftlibs", registry._tools)

    def test_craftlibs_ai_action_accepts_the_new_kinds(self):
        from nomorals.tools.craftlibs import run as craftlibs_run
        ctx = SimpleNamespace()
        out = craftlibs_run(ctx, "ai", {"kind": "status",
                                        "role": "inpaint"})
        self.assertIn("status", out)
        out = craftlibs_run(ctx, "ai", {"kind": "http",
                                        "base": "http://127.0.0.1:11434"})
        self.assertIn("proxy", out)


if __name__ == "__main__":
    unittest.main()
