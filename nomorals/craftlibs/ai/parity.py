"""The Craft-AI parity cases (census §3A.22).

``nomorals/craftlibs/craftai-parity.toml`` lists real fixtures +
operations + expected outcomes; :func:`run_craftai_case` runs one
case and raises on any mismatch, and the shared runner in
:mod:`nomorals.craftlibs.parity` runs the whole file. Missing
fixture = fail, never skip.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from .capability import CapabilityId
from .client import AiClient
from .connection import Auth, Billing, Connection, Locality
from .consent import ConsentLedger, Disclosure
from .consent_term import ConsentTerm
from .draft import PRESETS, ConnectionDraft
from .error import AiError
from .factory import ProviderFactory, check_provider
from .http import agent_for, hosted_proxy, proxy_for, scan_agents
from .ids import ConnectionId, ModelId, RoleId
from .mock import MockProvider
from .model import CapabilitySpec, InputRole, ModelInfo
from .model_folders import ModelFolder, download_target, validate_folders
from .payload import Payload
from .request import Request
from .role import Binding, Role
from .secrets import EnvStore
from .segment import (
    CAPABILITY as SEGMENT_CAPABILITY,
)
from .segment import (
    Click,
    Found,
    clicks,
    found_output,
    read_found,
    with_clicks,
)
from .shared import HistoryFile, SharedStore
from .store import StoreFile

__all__ = ["run_craftai_case"]


def _hosted_connection() -> Connection:
    return Connection(
        "hosted-conn", "Hosted", "mock",
        endpoint="https://example.com", locality=Locality.REMOTE,
        auth=Auth.none(), models=[_text_model()])


def _text_model(model_id: str = "mock-text") -> ModelInfo:
    return ModelInfo.new(
        ModelId.new(model_id), "Mock text model").with_spec(
            CapabilitySpec.new(CapabilityId.parse("text.generate"))
            .accepting(InputRole.PROMPT, 1)
            .requiring(InputRole.PROMPT))


def _motion_model() -> ModelInfo:
    return ModelInfo.new(
        ModelId.new("mock-motion"), "Mock motion model").with_spec(
            CapabilitySpec.new(CapabilityId.parse("motion.generate"))
            .accepting(InputRole.PROMPT, 1)
            .requiring(InputRole.PROMPT))


def _text_request(prompt: str = "hello") -> Request:
    return Request.new(CapabilityId.parse("text.generate")).with_input(
        InputRole.PROMPT, Payload.text(prompt))


def _folders(case: dict) -> list:
    return [ModelFolder.from_dict(item) for item in case["folders"]]


def run_craftai_case(case: dict) -> None:
    """Run one parity case; raise on any mismatch."""
    op = str(case.get("operation", ""))
    tmp = Path(tempfile.mkdtemp(prefix="craftai-parity-"))
    try:
        _run(op, case, tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _run(op: str, case: dict, tmp: Path) -> None:
    if op == "consent_term_agree":
        disclosure = Disclosure.for_request(
            "craftai-test", _hosted_connection(), _text_request())
        ledger = ConsentLedger()
        term = ConsentTerm.from_id(str(case["term"]))
        covered = ledger.agree(disclosure, term, 1000)
        record = covered.records[0]
        offset = int(case["offset_seconds"])
        want = None if offset < 0 else 1000 + offset
        if record.expires_at != want:
            raise AssertionError(
                f"term {case['term']} expires at "
                f"{record.expires_at}, want {want}")
        if not term.remembers():
            # Nothing is remembered: the caller's ledger still asks,
            # the returned one covers the job.
            if not ledger.needs_consent(disclosure, 1000):
                raise AssertionError(
                    "a this_time agreement was remembered")
            if covered.needs_consent(disclosure, 1000):
                raise AssertionError(
                    "a this_time agreement does not cover the job")
    elif op == "consent_term_forget":
        first = Disclosure.for_request(
            "craftai-test", _hosted_connection(), _text_request())
        other_conn = Connection(
            "hosted-conn-2", "Hosted 2", "mock",
            endpoint="https://example.com/other",
            locality=Locality.REMOTE, auth=Auth.none(),
            models=[_text_model()])
        second = Disclosure.for_request(
            "craftai-test", other_conn, _text_request())
        ledger = ConsentLedger()
        ledger.agree(first, ConsentTerm.one_hour(), 1000)
        ledger.agree(second, ConsentTerm.until_revoked(), 1000)
        ledger.forget_lapsed(4600)
        if not ledger.needs_consent(first, 4600):
            raise AssertionError("the lapsed hour grant still covers")
        if ledger.needs_consent(second, 10 ** 9):
            raise AssertionError("the until-revoked grant lapsed")
    elif op == "model_folders_validate":
        validate_folders(_folders(case))
    elif op == "model_folders_download":
        folder, sub = download_target(
            _folders(case), str(case["kind"]))
        if folder is None or sub is None:
            raise AssertionError("no download target")
        if folder.id != str(case["folder"]):
            raise AssertionError(
                f"download folder {folder.id}, want {case['folder']}")
        if sub != str(case["subfolder"]):
            raise AssertionError(
                f"download subfolder {sub}, want {case['subfolder']}")
    elif op == "draft_set_models":
        preset = next(p for p in PRESETS if p.key == str(case["preset"]))
        draft = ConnectionDraft.from_preset(preset)
        got = draft.set_models(str(case["address"]), [_text_model()])
        if got != bool(case["expected"]):
            raise AssertionError(
                f"set_models({case['address']}) = {got}, "
                f"want {case['expected']}")
    elif op == "draft_prepare":
        preset = next(p for p in PRESETS if p.key == str(case["preset"]))
        draft = ConnectionDraft.from_preset(preset)
        draft.label = str(case["label"])
        if case.get("key"):
            draft.key = str(case["key"])
        prepared = draft.prepare([])
        if prepared.connection.provider != str(case["provider"]):
            raise AssertionError(
                f"provider {prepared.connection.provider}, "
                f"want {case['provider']}")
        if prepared.connection.label != str(case["label"]):
            raise AssertionError(
                f"label {prepared.connection.label}, "
                f"want {case['label']}")
    elif op == "segment_clicks":
        request = Request.new(CapabilityId.parse(SEGMENT_CAPABILITY))
        points = [Click(float(x), float(y), True)
                  for x, y in case["points"]]
        try:
            got = clicks(with_clicks(request, points))
        except AiError as exc:
            if str(case.get("expect_error", "")) in str(exc):
                return
            raise
        if case.get("expect_error"):
            raise AssertionError(
                f"expected {case['expect_error']}, got {len(got)} clicks")
        if len(got) != int(case["expected_count"]):
            raise AssertionError(
                f"{len(got)} clicks, want {case['expected_count']}")
    elif op == "segment_found":
        found = Found(bool(case["matched"]),
                      case.get("score"), case.get("name"))
        back = read_found([found_output(found)])
        if back != found:
            raise AssertionError(f"read_found = {back!r}, want {found!r}")
    elif op == "binding_refusal":
        mock = Connection(
            "mock-conn", "Mock", "mock", locality=Locality.LOCAL,
            auth=Auth.none(), billing=Billing.FREE,
            models=[_text_model(), _motion_model()])
        shared = SharedStore(tmp / "ai.json")
        shared.replace(StoreFile(
            connections=[mock],
            defaults=[Binding(RoleId.new("inpaint"),
                              ConnectionId.new("mock-conn"),
                              ModelId.new("mock-motion"))]))
        client = AiClient("craftai-test", shared, EnvStore(),
                          factory=ProviderFactory.standard())
        role = Role("inpaint", "Inpaint", "text.generate",
                    [InputRole.PROMPT])
        try:
            client.plan(role, _text_request())
        except AiError as exc:
            if exc.kind.value != str(case["kind"]):
                raise AssertionError(
                    f"refusal kind {exc.kind.value}, want {case['kind']}"
                    ) from exc
            return
        raise AssertionError("the wrong binding was not refused")
    elif op == "conformance_pass":
        report = check_provider(MockProvider([_text_model()]),
                                _text_model(), _text_request())
        if not report["passed"]:
            raise AssertionError(
                f"conformance failed on {report['failed']}")
    elif op == "conformance_fail":
        class BrokenProvider(MockProvider):
            def poll(self, job_id: str):
                raise AiError.invalid("poll always fails")

        report = check_provider(BrokenProvider([_text_model()]),
                                _text_model(), _text_request())
        if report["passed"]:
            raise AssertionError("the broken provider passed")
        if str(case["check"]) not in report["failed"]:
            raise AssertionError(
                f"failed checks {report['failed']}, want "
                f"{case['check']} among them")
    elif op == "tls_policy":
        loopback = agent_for(str(case["loopback_base"]))
        if loopback._effective_proxy(str(case["loopback_base"])) != "":
            raise AssertionError("the loopback agent takes a proxy")
        if not loopback.allow_private_ips:
            raise AssertionError(
                "the loopback agent refuses loopback addresses")
        if not loopback.verify_tls:
            raise AssertionError("the loopback agent skips TLS checks")
        hosted = agent_for(str(case["hosted_base"]))
        if hosted._effective_proxy(str(case["hosted_base"])) != \
                hosted_proxy():
            raise AssertionError(
                "the hosted agent does not take the environment proxy")
    elif op == "tls_proxy_for":
        if proxy_for(str(case["base"])) != str(case["expected"]):
            raise AssertionError(
                f"proxy_for({case['base']}) = "
                f"{proxy_for(str(case['base']))!r}, want "
                f"{case['expected']!r}")
    elif op == "scan_agents_clean":
        root = Path(__file__).resolve().parents[3]
        violations = scan_agents([root / "nomorals" / "llm",
                                  root / "nomorals" / "craftlibs" / "ai"])
        if violations:
            raise AssertionError(
                f"agents outside the single TLS policy: {violations[:3]}")
    elif op == "secret_env":
        var = EnvStore.var_name(str(case["reference"]))
        os.environ[var] = str(case["value"])
        try:
            got = EnvStore().get(str(case["reference"]))
        finally:
            del os.environ[var]
        if got != str(case["value"]):
            raise AssertionError(
                f"env store gave {got!r}, want {case['value']!r}")
    elif op == "history_round_trip":
        history = HistoryFile(tmp / "history.jsonl")
        history.append({"phase": "start", "run": "a"})
        history.append({"phase": "end", "run": "a"})
        read = history.read()
        if len(read.records) != 2 or read.skipped != 0:
            raise AssertionError(
                f"history read {len(read.records)} records, "
                f"{read.skipped} skipped")
    else:
        raise AssertionError(f"unknown parity operation: {op!r}")
