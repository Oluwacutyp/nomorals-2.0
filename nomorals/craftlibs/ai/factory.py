"""The provider factory (Craft-AI port of ai-client ``factory.rs`` +
``declarative.rs`` + ai-openai's OpenAI-shaped adapter).

A factory maps provider names to adapters. The OpenAI family
(openai, openai-compatible, ollama, lmstudio, llamacpp, vllm,
openrouter) is served by one adapter over Devon's own
``llm`` OpenAI-compatible provider — the seam never grows a second
HTTP client. ``mock`` and ``local-motion`` are Devon's own. The
``declarative`` entry reads ``<dir>/<connection id>.json`` and
refuses a spec whose ``base`` is not the connection's address, so a
spec file can never send data somewhere the agreement did not cover.
Paid hosted image/video adapters (artcraft-omni, bfl, seedance,
vertex-express, sam3) are registered as names the factory knows but
refuses to build without a live key and a real adapter — the factory
never pretends to run a model it cannot reach.

:func:`check_provider` runs the seam's conformance suite against a
provider and reports per check. :func:`register_checked` refuses a
registration whose probe provider fails conformance, naming the
failing check — a provider that fails conformance is refused at
registration, not at call time.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from . import conformance
from .capability import CapabilityId
from .connection import Auth, Connection, Locality
from .endpoint import Endpoint
from .error import AiError, ErrorKind
from .job import JobSnapshot, JobState
from .local import LocalMotionProvider
from .mock import MockProvider
from .model import InputRole
from .payload import Payload
from .provider import Provider
from .request import Request
from .spec import Spec, SpecError, render_text, render_value

__all__ = [
    "OPENAI_FAMILY",
    "DeclarativeProvider",
    "OpenAiCompatProvider",
    "ProviderFactory",
    "check_provider",
    "register_checked",
]

OPENAI_FAMILY = ["openai", "openai-compatible", "ollama", "lmstudio",
                 "llamacpp", "vllm", "openrouter"]

#: Providers the factory knows by name but cannot build without a
#: live key and a real adapter. The refusal names the provider.
_HOSTED_WITHOUT_ADAPTER = ("artcraft-omni", "bfl", "seedance",
                           "vertex-express", "sam3", "comfyui")


def _model_id(model) -> str:
    return model.id.as_str() if hasattr(model, "id") else str(model)


def _find_model(models: list, model):
    wanted = _model_id(model)
    for info in models:
        if info.id.as_str() == wanted:
            return info
    return None


def _prompt_text(request: Request) -> str | None:
    for role, payload in request.inputs:
        if role == InputRole.PROMPT and payload.as_text() is not None:
            return payload.as_text()
    return None


class OpenAiCompatProvider(Provider):
    """The seam's ``Provider`` over Devon's ``llm`` OpenAI-compatible
    provider. One adapter serves the whole OpenAI family; the chat
    runs synchronously inside :meth:`submit`, so a submitted job is
    already finished when :meth:`poll` first sees it."""

    def __init__(self, endpoint: str, api_key: str | None = None,
                 models: list | None = None, timeout: float = 120.0,
                 key_header: str | None = None) -> None:
        from nomorals.llm.base import Message
        from nomorals.llm.providers.openai_compat import (
            OpenAICompatProvider as _LlmProvider,
        )

        self._endpoint = endpoint.rstrip("/")
        self._api_key = api_key or ""
        self._models = list(models or [])
        self._timeout = timeout
        self._key_header = key_header
        self._message_cls = Message
        first = self._models[0].id.as_str() if self._models else ""
        self._llm = _LlmProvider(base_url=self._endpoint,
                                 api_key=self._api_key, model=first,
                                 timeout=timeout)
        # A loopback endpoint is this computer: the SSRF guard must
        # let it through, exactly as the ollama and llama.cpp
        # providers do for their local servers.
        try:
            if Endpoint.parse(self._endpoint).is_loopback():
                self._llm.http.allow_private_ips = True
        except (ValueError, AiError):
            pass
        self._jobs: dict[str, dict] = {}
        self._next = 0

    @property
    def timeout(self) -> float:
        return self._timeout

    def models(self) -> list:
        return list(self._models)

    def estimate(self, model, request: Request) -> dict:
        return {}

    def submit(self, model, request: Request) -> str:
        info = _find_model(self._models, model)
        if info is None:
            raise AiError.invalid(f"unknown model {_model_id(model)}")
        spec = info.spec_for(request.capability)
        if spec is None:
            raise AiError.unsupported(
                f"model {_model_id(model)} cannot do "
                f"{request.capability.as_str()}")
        from .payload import Limits

        request.validate(spec, Limits())
        if request.capability.as_str() != "text.generate":
            raise AiError.unsupported(
                "the OpenAI-compatible adapter only runs text.generate; "
                f"{request.capability.as_str()} needs another adapter")
        prompt = _prompt_text(request)
        if prompt is None:
            raise AiError.invalid("a text request needs a prompt input")
        self._next += 1
        job_id = f"openai-{self._next}"
        try:
            response = self._llm.chat(
                [self._message_cls(role="user", content=prompt)],
                timeout=self._timeout)
        except Exception as exc:
            raise AiError.new(
                ErrorKind.NETWORK,
                f"the chat call failed: {exc}") from exc
        if not response.ok:
            raise AiError.new(
                ErrorKind.PROVIDER,
                f"the chat call failed: {response.error or 'unknown error'}")
        state = JobState.SUCCEEDED
        snapshot = JobSnapshot(state, [
            {"type": "output",
             "payload": Payload.text(response.text or "").to_dict()}])
        self._jobs[job_id] = {"done": state, "snapshot": snapshot}
        return job_id

    def poll(self, job_id: str) -> dict:
        job = self._jobs.get(job_id)
        if job is None:
            raise AiError.invalid(f"unknown job {job_id}")
        return job["snapshot"].to_dict()

    def cancel(self, job_id: str) -> None:
        if job_id not in self._jobs:
            raise AiError.invalid(f"unknown job {job_id}")


def _json_pointer(document, pointer: str):
    """Read a JSON pointer (``/choices/0/message/content``) out of a
    parsed reply. ``None`` when it points nowhere."""
    if pointer == "":
        return document
    if not pointer.startswith("/"):
        return None
    current = document
    for part in pointer.split("/")[1:]:
        part = part.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict):
            if part not in current:
                return None
            current = current[part]
        elif isinstance(current, list):
            try:
                current = current[int(part)]
            except (ValueError, IndexError):
                return None
        else:
            return None
    return current


_MEDIA_MAGIC = {
    b"\x89PNG\r\n\x1a\n": "image/png",
    b"\xff\xd8\xff": "image/jpeg",
    b"GIF87a": "image/gif",
    b"GIF89a": "image/gif",
    b"RIFF": "image/webp",
}


class DeclarativeProvider(Provider):
    """A provider described by a data file (``craft_ai_core::spec``).
    Runs a ``request_response`` flow in one call, or a ``queue``
    flow by submit-then-poll under the spec's limits. The key never
    comes from the spec — it comes from the connection's secret
    reference, passed in at build time."""

    def __init__(self, spec: Spec, api_key: str | None = None,
                 timeout: float = 120.0) -> None:
        from .http import agent_for

        self._spec = spec
        self._api_key = api_key or ""
        self._timeout = timeout
        self._agent = agent_for(spec.base)
        self._jobs: dict[str, dict] = {}
        self._next = 0

    @property
    def timeout(self) -> float:
        return self._timeout

    def models(self) -> list:
        return self._spec.model_infos()

    def estimate(self, model, request: Request) -> dict:
        return {}

    def _headers(self) -> dict:
        auth = self._spec.auth
        if auth.kind == "none" or not self._api_key:
            return {}
        if auth.kind == "header":
            return {auth.name or "X-Api-Key": self._api_key}
        return {"Authorization": f"Bearer {self._api_key}"}

    def _call(self, call, values: dict):
        try:
            path = render_text(call.path, **values)
            query = {k: render_text(v, **values)
                     for k, v in (call.query or {}).items()}
            body = (render_value(call.body, **values)
                    if call.body is not None else None)
        except ValueError as exc:
            raise AiError.new(ErrorKind.INTERNAL,
                              f"could not fill in a call: {exc}") from exc
        url = self._spec.base.rstrip("/") + path
        if call.method == "GET":
            response = self._agent.get(
                url, headers=self._headers(), params=query or None,
                timeout=self._timeout)
        elif call.method == "POST":
            response = self._agent.post_json(
                url, body or {}, headers=self._headers(),
                params=query or None, timeout=self._timeout)
        elif call.method == "DELETE":
            response = self._agent.request(
                "DELETE", url, headers=self._headers(),
                params=query or None, timeout=self._timeout)
        else:
            raise AiError.invalid(f"unsupported method {call.method}")
        if response.status < 200 or response.status >= 300:
            raise AiError.new(
                ErrorKind.PROVIDER,
                f"the service answered HTTP {response.status}")
        try:
            return json.loads(response.body.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise AiError.new(
                ErrorKind.PROVIDER,
                "the service did not answer JSON") from exc

    def _extract(self, reply, extract) -> Payload:
        from .payload import base64_decode

        value = _json_pointer(reply, extract.at)
        if not isinstance(value, str):
            raise AiError.new(
                ErrorKind.PROVIDER,
                "the service did not give a result where the spec says "
                "it would")
        media = extract.media_type or ""
        if extract.kind == "text":
            return Payload.text(value)
        if extract.kind == "url":
            try:
                parsed = Endpoint.parse(value)
            except (ValueError, AiError):
                parsed = None
            if (parsed is None or not parsed.https
                    or parsed.is_loopback() or "#" in value):
                raise AiError.new(
                    ErrorKind.PROVIDER,
                    "the service gave a result link that is not a "
                    "public https link")
            return Payload.handle(value, media)
        if extract.kind == "base64":
            raw = base64_decode(value)
            if raw is None:
                raise AiError.new(
                    ErrorKind.PROVIDER,
                    "the service gave a result that is not valid base64")
            for magic, kind in _MEDIA_MAGIC.items():
                if raw.startswith(magic) and media and media != kind:
                    raise AiError.new(
                        ErrorKind.PROVIDER,
                        "the service gave a file that is not the kind "
                        "the spec says")
            return Payload.inline(media or "application/octet-stream", raw)
        raise AiError.invalid(f"unknown extract kind {extract.kind!r}")

    def submit(self, model, request: Request) -> str:
        info = _find_model(self.models(), model)
        if info is None:
            raise AiError.invalid(f"unknown model {_model_id(model)}")
        spec_model = next(
            (m for m in self._spec.models if m.id == info.id.as_str()),
            None)
        if spec_model is None:
            raise AiError.invalid(f"unknown model {_model_id(model)}")
        spec = info.spec_for(request.capability)
        if spec is None:
            raise AiError.unsupported(
                f"model {_model_id(model)} cannot do "
                f"{request.capability.as_str()}")
        from .payload import Limits

        request.validate(spec, Limits())
        prompt = _prompt_text(request)
        values = {"prompt": prompt or "",
                  "model": info.id.as_str(), "job_id": None}
        self._next += 1
        job_id = f"spec-{self._next}"
        flow = self._spec.flow
        if flow.kind == "request_response":
            reply = self._call(flow.call, values)
            payload = self._extract(reply, flow.result)
            snapshot = JobSnapshot(JobState.SUCCEEDED, [
                {"type": "output", "payload": payload.to_dict()}])
            self._jobs[job_id] = {"done": JobState.SUCCEEDED,
                                  "snapshot": snapshot}
            return job_id
        if flow.kind == "queue":
            from .spec import Call

            reply = self._call(Call.from_dict(flow.submit["call"]), values)
            remote_id = _json_pointer(
                reply, flow.submit.get("job_id_at", ""))
            if not isinstance(remote_id, str) or not remote_id:
                raise AiError.new(
                    ErrorKind.PROVIDER,
                    "the service did not give a job id")
            self._jobs[job_id] = {"done": None, "remote_id": remote_id,
                                  "info": info, "request": request,
                                  "polls": 0, "started": time.monotonic()}
            return job_id
        raise AiError.invalid(f"unknown flow kind {flow.kind!r}")

    def poll(self, job_id: str) -> dict:
        job = self._jobs.get(job_id)
        if job is None:
            raise AiError.invalid(f"unknown job {job_id}")
        if job["done"] is not None:
            return job["snapshot"].to_dict()
        flow = self._spec.flow
        limits = self._spec.limits
        job["polls"] += 1
        if (job["polls"] > limits.max_polls
                or time.monotonic() - job["started"] > limits.max_seconds):
            state = JobState.failed(AiError.new(
                ErrorKind.PROVIDER, "the service did not finish in time"))
            snapshot = JobSnapshot(state, [])
            job["done"] = state
            job["snapshot"] = snapshot
            return snapshot.to_dict()
        values = {"prompt": _prompt_text(job["request"]) or "",
                  "model": job["info"].id.as_str(),
                  "job_id": job["remote_id"]}
        from .spec import Call, Extract

        reply = self._call(Call.from_dict(flow.poll["call"]), values)
        status_word = _json_pointer(reply, flow.poll.get("status_at", ""))
        state = flow.poll.get("states", {}).get(status_word)
        if state == "succeeded":
            payload = self._extract(
                reply, Extract.from_dict(flow.poll["result"]))
            snapshot = JobSnapshot(JobState.SUCCEEDED, [
                {"type": "output", "payload": payload.to_dict()}])
            job["done"] = JobState.SUCCEEDED
            job["snapshot"] = snapshot
            return snapshot.to_dict()
        if state == "failed":
            error = AiError.new(ErrorKind.PROVIDER,
                                "the service failed the job")
            state = JobState.failed(error)
            snapshot = JobSnapshot(state, [])
            job["done"] = state
            job["snapshot"] = snapshot
            return snapshot.to_dict()
        return JobSnapshot(JobState.running(
            min(job["polls"] * 10, 99)), []).to_dict()

    def cancel(self, job_id: str) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            raise AiError.invalid(f"unknown job {job_id}")
        if job["done"] is not None:
            return
        flow = self._spec.flow
        if flow.cancel is not None:
            values = {"prompt": _prompt_text(job["request"]) or "",
                      "model": job["info"].id.as_str(),
                      "job_id": job["remote_id"]}
            self._call(flow.cancel, values)
        state = JobState.CANCELLED
        job["done"] = state
        job["snapshot"] = JobSnapshot(state, [])


class ProviderFactory:
    """Maps provider names to adapters."""

    def __init__(self, timeouts=None) -> None:
        self._builders: dict[str, object] = {}
        self._accelerators: dict[str, list] = {}
        self._declarative_specs = None
        self._timeouts = timeouts

    @classmethod
    def standard(cls) -> ProviderFactory:
        return cls._built(None)

    @classmethod
    def probing(cls) -> ProviderFactory:
        """Like :meth:`standard` but with short timeouts (5 s to
        connect, 20 s in all), for the "Test" button: a server that
        does not answer must not leave the screen waiting for the ten
        minutes a real image job is allowed."""
        return cls._built((5.0, 20.0))

    @classmethod
    def _built(cls, timeouts) -> ProviderFactory:
        factory = cls(timeouts)
        for name in OPENAI_FAMILY:
            factory.register(
                name,
                lambda c, s, t=timeouts: _build_openai(c, s, t))
        factory.register(
            "mock",
            lambda c, s: MockProvider(list(c.models or [])))
        factory.register("local-motion", lambda c, s: LocalMotionProvider())
        factory.register_declarative()
        return factory

    def register(self, name: str, builder) -> None:
        """Register ``builder(connection, secret)`` under ``name``.
        Names are matched case-insensitively."""
        self._builders[name.lower()] = builder

    def register_declarative(self) -> None:
        self.register("declarative",
                      lambda c, s: _build_declarative(
                          c, s, self._declarative_specs, self._timeouts))

    def unregister(self, name: str) -> None:
        self._builders.pop(name.lower(), None)

    def knows(self, name: str) -> bool:
        if name.lower() in _HOSTED_WITHOUT_ADAPTER:
            return True
        return name.lower() in self._builders

    def set_declarative_specs(self, directory) -> None:
        """Point the ``declarative`` entry at a folder of spec
        files, and re-register it so the new folder takes effect."""
        self._declarative_specs = Path(directory) if directory else None
        self.register_declarative()

    def declare_accelerators(self, name: str, vendors: list) -> None:
        self._accelerators[name.lower()] = list(vendors)

    def accelerators(self, name: str):
        return self._accelerators.get(name.lower())

    def build(self, connection: Connection, secret):
        """Build the adapter for ``connection``. Unknown providers
        are refused by name; hosted providers without a live adapter
        are refused with their name."""
        name = connection.provider.lower()
        if name in _HOSTED_WITHOUT_ADAPTER:
            raise AiError.unsupported(
                f"This app can't use \"{connection.provider}\" yet "
                f"(connection \"{connection.label}\")")
        builder = self._builders.get(name)
        if builder is None:
            raise AiError.unsupported(
                f"This app can't use \"{connection.provider}\" yet "
                f"(connection \"{connection.label}\")")
        return builder(connection, secret)

    def __repr__(self) -> str:
        return f"ProviderFactory({sorted(self._builders)})"


def _build_openai(connection: Connection, secret, timeouts):
    endpoint = connection.endpoint
    if not endpoint:
        raise AiError.invalid(
            f"connection \"{connection.label}\" has no address to "
            "connect to")
    timeout = 120.0
    if timeouts is not None:
        timeout = timeouts[1]
    return OpenAiCompatProvider(
        endpoint, api_key=secret, models=list(connection.models or []),
        timeout=timeout, key_header=connection.key_header)


def _build_declarative(connection: Connection, secret, directory,
                       timeouts):
    spec_dir = directory
    if spec_dir is None:
        from .shared import default_store_path

        base = default_store_path()
        spec_dir = base.parent / "specs" if base is not None else None
    if spec_dir is None:
        raise AiError.invalid(
            f"This app can't read the settings for {connection.label}. "
            "Open Set up AI to fix or re-add them.")
    path = Path(spec_dir) / f"{connection.id.as_str()}.json"
    try:
        text = path.read_text(encoding="utf-8")[:64 * 1024]
    except OSError:
        raise AiError.invalid(
            f"This app can't read the settings for {connection.label}. "
            "Open Set up AI to fix or re-add them.") from None
    try:
        spec = Spec.parse(text)
    except SpecError:
        raise AiError.invalid(
            f"This app can't read the settings for {connection.label}. "
            "Open Set up AI to fix or re-add them.") from None
    same_address = False
    if connection.endpoint:
        try:
            left = Endpoint.parse(connection.endpoint).fingerprint()
            right = Endpoint.parse(spec.base).fingerprint()
            same_address = left == right
        except (ValueError, AiError):
            same_address = False
    if not same_address:
        raise AiError.invalid(
            f"The settings for {connection.label} send to a different "
            "address than the connection. Open Set up AI to fix this.")
    timeout = 120.0
    if timeouts is not None:
        timeout = timeouts[1]
    return DeclarativeProvider(spec, api_key=secret, timeout=timeout)


def check_provider(provider, model, request: Request) -> dict:
    """Run the conformance suite against ``provider`` and report
    per check. Nothing here touches the network unless the provider
    does."""
    checks: dict[str, dict] = {}

    def _run(name: str, fn) -> None:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - one check's failure
            checks[name] = {"ok": False, "error": str(exc)}
        else:
            checks[name] = {"ok": True, "error": None}

    _run("lifecycle", lambda: conformance.run_lifecycle(
        provider, model, request))
    _run("cancel", lambda: conformance.run_cancel(provider, model, request))
    wrong_capability = Request.new(
        CapabilityId.parse("image.generate")).with_input(
            InputRole.PROMPT, Payload.text("a cat"))
    _run("rejects_wrong_capability", lambda: conformance.run_rejects(
        provider, model, wrong_capability, ErrorKind.UNSUPPORTED))
    from .ids import ModelId
    from .model import ModelInfo

    ghost = ModelInfo.new(ModelId.new("ghost-model"), "Ghost")
    for spec in (model.capabilities or []):
        ghost.with_spec(spec)
    _run("rejects_unknown_model", lambda: conformance.run_rejects(
        provider, ghost, request, ErrorKind.INVALID_REQUEST))
    _run("unknown_job", lambda: conformance.run_unknown_job(provider))
    failed = [name for name, result in checks.items()
              if not result["ok"]]
    return {"passed": not failed, "failed": failed, "checks": checks}


def register_checked(factory: ProviderFactory, name: str, builder,
                     probe) -> None:
    """Register ``builder`` under ``name`` only if a probe provider
    built from it passes conformance. A provider that fails is
    refused at registration, naming the failing check."""
    model, request = probe
    probe_connection = Connection(
        "probe-conn", "Probe", name, endpoint="http://127.0.0.1:1",
        locality=Locality.LOCAL, auth=Auth.none(), models=[model])
    try:
        provider = builder(probe_connection, None)
    except AiError as exc:
        raise AiError.invalid(
            f"cannot register {name!r}: the probe provider could not "
            f"be built ({exc})") from exc
    report = check_provider(provider, model, request)
    if not report["passed"]:
        raise AiError.invalid(
            f"cannot register {name!r}: conformance failed on "
            f"{', '.join(report['failed'])}")
    factory.register(name, builder)
