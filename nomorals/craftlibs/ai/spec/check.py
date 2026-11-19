"""The spec checker (Craft-Libs port of ai-core ``spec/check.rs``).

Every rule a spec must keep, applied once when it is loaded. An
error says where (``flow.submit.call.path``) and why, in words a
person can act on.
"""

from __future__ import annotations

import json

from ..capability import CapabilityId
from ..connection import check_key_header
from ..endpoint import Endpoint
from ..ids import ConnectionId, ModelId
from ..payload import validate_media_type
from . import template as _template
from .limits import (
    MAX_MODELS,
    MAX_PATH_BYTES,
    MAX_POINTER_DEPTH,
    MAX_SPEC_BYTES,
    MAX_STATES,
    MAX_TEMPLATE_BYTES,
    SPEC_VERSION,
)
from .types import Call, Extract, Flow, Spec, SpecAuth, SpecLimits, SpecModel

__all__ = ["SpecError", "parse"]

_TOP_KEYS = {"spec_version", "id", "label", "base", "auth", "probe",
             "models", "flow", "limits"}
_AUTH_KEYS = {"kind", "name"}
_CALL_KEYS = {"method", "path", "query", "body"}
_EXTRACT_KEYS = {"at", "as", "media_type"}
_MODEL_KEYS = {"id", "label", "capability", "accepts", "requires",
               "max_outputs"}
_LIMIT_KEYS = {"max_polls", "max_seconds"}
_SUBMIT_KEYS = {"call", "job_id_at"}
_POLL_KEYS = {"call", "status_at", "states", "result"}

_SEND = ("prompt", "model")
_FOLLOW = ("model", "job_id")
_NOTHING = ()


class SpecError(ValueError):
    """Why a spec was refused. ``at`` says where in the spec,
    ``why`` says why, in words a person can act on."""

    def __init__(self, at: str, why: str) -> None:
        super().__init__(f"{at}: {why}")
        self.at = at
        self.why = why


def _err(at: str, why: str) -> SpecError:
    return SpecError(at, why)


def _unknown_keys(data: dict, allowed, at: str) -> None:
    for key in data:
        if key not in allowed:
            raise _err(at, f"unknown field '{key}'")


def parse(text: str) -> Spec:
    """Read and check a spec. Nothing is returned that has not
    passed every rule."""
    if len(text.encode("utf-8")) > MAX_SPEC_BYTES:
        raise _err("spec", "this spec is too large (the limit is "
                            f"{MAX_SPEC_BYTES} bytes)")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise _err("spec",
                   f"not a spec this version can read: {exc}") from exc
    if not isinstance(data, dict):
        raise _err("spec", "not a spec this version can read: not an "
                           "object")
    _unknown_keys(data, _TOP_KEYS, "spec")
    try:
        spec = _build(data)
    except SpecError:
        raise
    except (ValueError, KeyError, TypeError) as exc:
        raise _err("spec",
                   f"not a spec this version can read: {exc}") from exc
    _validate(spec)
    return spec


def _build(data: dict) -> Spec:
    auth_data = data.get("auth", {"kind": "bearer"})
    if not isinstance(auth_data, dict):
        raise _err("auth", "not an object")
    _unknown_keys(auth_data, _AUTH_KEYS, "auth")
    auth = SpecAuth.from_dict(auth_data)
    probe = None
    if data.get("probe") is not None:
        _unknown_keys(data["probe"], _CALL_KEYS, "probe")
        probe = Call.from_dict(data["probe"])
    models = []
    for item in data.get("models", []):
        if not isinstance(item, dict):
            raise _err("models", "a model is an object")
        _unknown_keys(item, _MODEL_KEYS, "models")
        models.append(SpecModel.from_dict(item))
    flow_data = data.get("flow")
    if not isinstance(flow_data, dict):
        raise _err("flow", "not an object")
    flow = Flow.from_dict(flow_data)
    limits = SpecLimits.from_dict(data.get("limits", {}))
    if data.get("limits") is not None:
        _unknown_keys(data["limits"], _LIMIT_KEYS, "limits")
    return Spec(int(data.get("spec_version", 0)), str(data.get("id", "")),
                str(data.get("label", "")), str(data.get("base", "")),
                auth, models, flow, probe, limits)


def _validate(spec: Spec) -> None:
    if spec.spec_version != SPEC_VERSION:
        raise _err("spec_version", "only version 1 is read")
    try:
        ConnectionId.new(spec.id)
    except ValueError as exc:
        raise _err("id", str(exc)) from exc
    _check_label("label", spec.label)
    _check_base(spec.base)
    if spec.auth.kind == "header":
        try:
            check_key_header(spec.auth.name or "")
        except ValueError as exc:
            raise _err("auth.name", str(exc)) from exc
    if spec.probe is not None:
        if spec.probe.method != "GET":
            raise _err("probe.method", "the probe must be a GET")
        _check_call("probe", spec.probe, _NOTHING, False)
    _check_models(spec.models)
    sent = _check_flow(spec.flow)
    if "prompt" not in sent:
        raise _err("flow", "the prompt is never sent: use {prompt} in "
                           "the request")
    for i, model in enumerate(spec.models):
        if "prompt" not in model.requires:
            raise _err(f"models[{i}].requires",
                       "a spec that sends {prompt} needs every model "
                       "to require a prompt")
    _check_limits(spec)


def _check_label(at: str, label: str) -> None:
    if (not label.strip() or len(label) > 120
            or any(ord(c) < 0x20 for c in label)):
        raise _err(at, "a label is 1 to 120 plain characters")


def _check_base(base: str) -> None:
    try:
        Endpoint.parse(base)
    except ValueError as exc:
        raise _err("base", str(exc)) from exc
    if "?" in base or "#" in base:
        raise _err("base", "a base has no query or fragment")
    rest = base.split("://", 1)[1] if "://" in base else ""
    slash = rest.find("/")
    if slash >= 0:
        prefix = rest[slash:].rstrip("/")
        if prefix:
            try:
                _check_path(prefix)
            except ValueError as exc:
                raise _err("base", str(exc)) from exc


def _check_path(path: str) -> None:
    if "://" in path or path.startswith("//"):
        raise ValueError("a path cannot name another host")
    if not path.startswith("/"):
        raise ValueError("a path starts with /")
    if len(path.encode("utf-8")) > MAX_PATH_BYTES:
        raise ValueError("a path is too long (at most 256 bytes)")
    if "?" in path or "#" in path:
        raise ValueError("no ? or # in a path (put query values in "
                         "query)")
    if "%" in path or "\\" in path:
        raise ValueError("no backslash or % escape in a path")
    if not all(0x21 <= ord(c) <= 0x7E for c in path):
        raise ValueError("a path uses plain visible characters only")
    if any(part == ".." for part in path.split("/")):
        raise ValueError("no '..' in a path")
    if "//" in path:
        raise ValueError("no '//' in a path")


def _check_pointer(at: str, pointer: str) -> None:
    if not pointer.startswith("/"):
        raise _err(at, "a pointer starts with /")
    if len(pointer) > 128 or any(ord(c) < 0x20 for c in pointer):
        raise _err(at, "a pointer is at most 128 plain characters")
    if pointer.count("/") > MAX_POINTER_DEPTH:
        raise _err(at, "a pointer goes at most 8 levels deep")


def _check_call(at: str, call: Call, allow, need_job: bool) -> list:
    try:
        _check_path(call.path)
    except ValueError as exc:
        raise _err(f"{at}.path", str(exc)) from exc
    try:
        path_slots = _template.slots(call.path)
    except ValueError as exc:
        raise _err(f"{at}.path", str(exc)) from exc
    used: list[str] = []
    for slot in path_slots:
        if slot != "job_id":
            raise _err(f"{at}.path",
                       f"only {{job_id}} may be used in a path, not "
                       f"{{{slot}}}")
        used.append(slot)
    if len(call.query) > 8:
        raise _err(f"{at}.query", "a call has at most 8 query values")
    for key, value in call.query.items():
        ok = (bool(key) and len(key) <= 64
              and all(c.isascii() and (c.isalnum() or c in "-_.")
                      for c in key))
        if not ok or len(str(value)) > MAX_PATH_BYTES:
            raise _err(f"{at}.query", "query names are plain, values at "
                                      "most 256 bytes")
        try:
            used.extend(_template.slots(str(value)))
        except ValueError as exc:
            raise _err(f"{at}.query", str(exc)) from exc
    if call.body is not None:
        if call.method != "POST":
            raise _err(f"{at}.body", "only a POST has a body")
        if len(json.dumps(call.body, ensure_ascii=False)
               .encode("utf-8")) > MAX_TEMPLATE_BYTES:
            raise _err(f"{at}.body", "a body template is at most 4 KiB")
        try:
            used.extend(_template.value_slots(call.body))
        except ValueError as exc:
            raise _err(f"{at}.body", str(exc)) from exc
    for slot in used:
        if slot not in allow:
            raise _err(at, f"{{{slot}}} is not available in this call")
    if need_job and "job_id" not in used:
        raise _err(at, "this call never uses {job_id}, so it cannot "
                       "name the job")
    return used


def _check_models(models: list) -> None:
    if not models:
        raise _err("models", "a spec offers at least one model")
    if len(models) > MAX_MODELS:
        raise _err("models", "a spec offers at most 16 models")
    seen: set[str] = set()
    for i, model in enumerate(models):
        at = f"models[{i}]"
        try:
            ModelId.new(model.id)
        except ValueError as exc:
            raise _err(f"{at}.id", str(exc)) from exc
        if model.id in seen:
            raise _err(f"{at}.id", "this model id is used twice")
        seen.add(model.id)
        _check_label(f"{at}.label", model.label)
        try:
            CapabilityId.parse(model.capability)
        except ValueError as exc:
            raise _err(f"{at}.capability", str(exc)) from exc
        if not model.accepts:
            raise _err(f"{at}.accepts",
                       "a model accepts at least one input")
        for role, count in model.accepts.items():
            if role != "prompt":
                raise _err(f"{at}.accepts",
                           f"unsupported: input '{role}' (version 1 "
                           "sends a prompt only)")
            if int(count) != 1:
                raise _err(f"{at}.accepts",
                           "the prompt count must be 1 (one prompt is "
                           "sent)")
        for required in model.requires:
            if required not in model.accepts:
                raise _err(f"{at}.requires",
                           f"'{required}' is required, so it must also "
                           "be accepted")
        if not 1 <= model.max_outputs <= 4:
            raise _err(f"{at}.max_outputs", "outputs are 1 to 4")


def _check_extract(at: str, extract: Extract) -> None:
    _check_pointer(f"{at}.at", extract.at)
    if extract.kind == "text" and extract.media_type is not None:
        raise _err(at, "a text result carries no media_type")
    if extract.kind in ("url", "base64") and extract.media_type is None:
        raise _err(at, "a media_type is needed for a link or base64 "
                       "result")
    if extract.media_type is not None:
        try:
            validate_media_type(extract.media_type)
        except ValueError:
            raise _err(at, f"media_type {extract.media_type!r} is not "
                           "a media type") from None


def _check_flow(flow: Flow) -> list:
    if flow.kind == "request_response":
        sent = _check_call("flow.call", flow.call, _SEND, False)
        _check_extract("flow.result", flow.result)
        return sent
    submit = flow.submit or {}
    poll = flow.poll or {}
    if not isinstance(submit, dict) or not isinstance(poll, dict):
        raise _err("flow", "a queue flow needs submit and poll")
    _unknown_keys(submit, _SUBMIT_KEYS, "flow.submit")
    _unknown_keys(poll, _POLL_KEYS, "flow.poll")
    sent = _check_call("flow.submit.call",
                       Call.from_dict(submit["call"]), _SEND, False)
    _check_pointer("flow.submit.job_id_at", str(submit.get("job_id_at",
                                                           "")))
    _check_call("flow.poll.call", Call.from_dict(poll["call"]), _FOLLOW,
                True)
    _check_pointer("flow.poll.status_at", str(poll.get("status_at", "")))
    states = poll.get("states", {})
    if len(states) > MAX_STATES:
        raise _err("flow.poll.states",
                   "a status table has at most 16 words")
    for word in states:
        if (not word or len(word) > 64
                or any(ord(c) < 0x20 for c in word)):
            raise _err("flow.poll.states",
                       "a status word is 1 to 64 plain characters")
    for needed in ("succeeded", "failed"):
        if needed not in states.values():
            raise _err("flow.poll.states",
                       f"the table needs a word that means {needed}")
    _check_extract("flow.poll.result",
                   Extract.from_dict(poll["result"]))
    if flow.cancel is not None:
        _check_call("flow.cancel", flow.cancel, _FOLLOW, True)
    return sent


def _check_limits(spec: Spec) -> None:
    if not 1 <= spec.limits.max_polls <= 10_000:
        raise _err("limits.max_polls", "status checks are 1 to 10000")
    if not 1 <= spec.limits.max_seconds <= 3600:
        raise _err("limits.max_seconds", "seconds are 1 to 3600")
