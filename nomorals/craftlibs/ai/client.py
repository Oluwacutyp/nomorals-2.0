"""The AI client (Craft-AI port of ai-client ``client.rs`` +
``client_run.rs`` + ``history.rs``).

One AI connection for every app. The client resolves a role to its
bound connection and model, plans a request (resolving and checking
it, sending nothing and reading no key), and starts it only when the
consent ledger covers the plan's disclosure. The key is read only
inside :meth:`AiClient.start`, and only after consent has passed.
A plan holds no secret. A plan for another app is refused before key
access or any history side effect.

History is the seam's start/end record pair, joined by run id into
entries by :func:`~nomorals.craftlibs.ai.history.merge`. Prompts are
kept only when :meth:`keep_prompts_in_history` says so; a refusal
after consent has passed is recorded too, so a refused run is never
invisible.
"""

from __future__ import annotations

import contextlib
import json
import time
import uuid
from pathlib import Path

from .capability import CapabilityId
from .connection import Billing, Locality
from .consent import ConsentLedger, ConsentRecord, Disclosure
from .error import AiError, ErrorKind
from .factory import ProviderFactory, check_provider
from .history import (
    EndRecord,
    HistoryLog,
    HistoryRecord,
    Outcome,
    StartRecord,
    Usage,
    clip_prompt,
    merge,
)
from .job import JobSnapshot, JobState
from .payload import Limits
from .picks import Catalog, Pick, choose
from .role import Role, eligible, resolve
from .secrets import EnvStore, LayeredStore, MemoryStore, secret_for
from .shared import HistoryFile, SharedStore, default_store_path
from .store import StoreFile

__all__ = [
    "NO_ADAPTER",
    "AiClient",
    "Choice",
    "Finished",
    "Job",
    "PickedChoice",
    "Plan",
    "RoleReady",
    "RoleStatus",
]

#: What a person is told when the app has no adapter for the model
#: they chose. One sentence, so status, plan and start agree and the
#: plain-language test can pin it.
NO_ADAPTER = ("This app can't run the chosen model yet. "
              "Choose another in Set up AI.")


def _now_ms() -> int:
    return int(time.time() * 1000)


def _now_secs() -> int:
    return int(time.time())


class RoleStatus:
    """Whether a role can run right now."""

    def __init__(self, ready=None, reason: str | None = None) -> None:
        self.ready = ready
        self.reason = reason

    @property
    def is_ready(self) -> bool:
        return self.ready is not None

    @classmethod
    def unavailable(cls, reason: str) -> RoleStatus:
        return cls(None, reason)


class RoleReady:
    """What a ready role is bound to."""

    def __init__(self, connection: str, connection_label: str,
                 model: str, model_label: str, locality: Locality,
                 billing: Billing) -> None:
        self.connection = connection
        self.connection_label = connection_label
        self.model = model
        self.model_label = model_label
        self.locality = locality
        self.billing = billing


class Choice:
    """One entry in a model picker."""

    def __init__(self, connection, connection_label: str,
                 model, model_label: str,
                 locality: Locality) -> None:
        from .ids import ConnectionId, ModelId

        self.connection = (connection
                           if isinstance(connection, ConnectionId)
                           else ConnectionId.new(connection))
        self.connection_label = connection_label
        self.model = (model if isinstance(model, ModelId)
                      else ModelId.new(model))
        self.model_label = model_label
        self.locality = locality


class PickedChoice:
    """A model that earned a preset for a job, and why."""

    def __init__(self, pick: Pick, choice: Choice, by) -> None:
        self.pick = pick
        self.choice = choice
        self.by = by


class Plan:
    """A checked request waiting for consent. Holds no secret."""

    def __init__(self, connection, model, request: object,
                 disclosure: Disclosure, role_id, role_label: str,
                 app: str) -> None:
        self.connection = connection
        self._model = model
        self.request = request
        self._disclosure = disclosure
        self.role_id = role_id
        self.role_label = role_label
        self.app = app

    def disclosure(self) -> Disclosure:
        """What would be sent where. Show this to the user."""
        return self._disclosure

    def needs_consent(self, ledger: ConsentLedger) -> bool:
        """True when the person must agree first: nothing on file
        covers this app, connection, destination, capability and kind
        of data, or the agreement has lapsed."""
        return self.needs_consent_at(ledger, _now_secs())

    def needs_consent_at(self, ledger: ConsentLedger, now: int) -> bool:
        return ledger.needs_consent(self._disclosure, now)

    def connection_label(self) -> str:
        return self.connection.label

    def model(self):
        return self._model


class Finished:
    """Everything a finished job produced."""

    def __init__(self, state: JobState, events: list, usage=None) -> None:
        self.state = state
        self.events = list(events)
        self.usage = usage

    @property
    def snapshot(self) -> dict:
        """The last snapshot, as a dict."""
        return JobSnapshot(self.state, self.events,
                           self.usage).to_dict()


class Job:
    """A started job."""

    def __init__(self, provider, job_id: str, recorder=None) -> None:
        self.provider = provider
        self.id = job_id
        self._recorder = recorder
        self._usage = None

    def poll(self) -> dict:
        snapshot = self.provider.poll(self.id)
        if self._recorder is not None:
            state = JobState.from_dict(snapshot.get("state"))
            self._recorder.observe(state, snapshot.get("usage"))
        return snapshot

    def cancel(self) -> None:
        self.provider.cancel(self.id)

    def wait(self, interval: float = 0.5,
             timeout: float = 60.0) -> Finished:
        """Poll every ``interval`` seconds until the job ends or
        ``timeout`` passes, collecting events. For scripts and
        tests; an app with its own job system polls instead."""
        deadline = time.monotonic() + timeout
        events: list = []
        usage = None
        while True:
            snapshot = self.poll()
            events.extend(snapshot.get("events", []))
            usage = snapshot.get("usage") or usage
            state = JobState.from_dict(snapshot.get("state"))
            if state.is_terminal():
                return Finished(state, events, usage)
            if time.monotonic() >= deadline:
                with contextlib.suppress(AiError):
                    self.cancel()
                if self._recorder is not None:
                    self._recorder.gave_up()
                raise AiError.new(ErrorKind.NETWORK,
                                  "the job took too long")
            time.sleep(interval)


class _Recorder:
    """The history lines for one run: a start line, then an end line
    when the job ends, is refused, or is given up on."""

    def __init__(self, history_file: HistoryFile, plan: Plan,
                 run: str, estimate, keep_prompts: bool) -> None:
        self._file = history_file
        self._run = run
        self._done = False
        self._usage = None
        self._plan = plan
        prompt = _prompt_of(plan.request) if keep_prompts else None
        start = StartRecord(
            run, _now_ms(), plan.app, plan.role_id.as_str(),
            plan.role_label, plan.request.capability.as_str(),
            plan.connection.id.as_str(), plan.connection.label,
            plan.model().as_str(), plan.connection.locality,
            plan.disclosure().destination,
            sent=[{"kind": item["kind"], "bytes": item["bytes"]}
                  for item in plan.disclosure().items],
            params=plan.disclosure().param_names,
            provider_job=None, estimate=estimate,
            prompt=(clip_prompt(prompt) if prompt is not None else None))
        self._file.append(HistoryRecord.start(start).to_dict())

    def observe(self, state: JobState, usage) -> None:
        if usage is not None:
            self._usage = usage
        if not state.is_terminal() or self._done:
            return
        self._done = True
        if state.kind == "succeeded":
            outcome = Outcome("done")
        elif state.kind == "cancelled":
            outcome = Outcome("cancelled")
        else:
            error = getattr(state, "error", None)
            outcome = Outcome.failed(
                error.kind if error is not None else ErrorKind.PROVIDER)
        self._file.append(HistoryRecord.end(EndRecord(
            self._run, _now_ms(), outcome,
            Usage.from_dict(self._usage)
            if isinstance(self._usage, dict) else None)).to_dict())

    def refused(self, error: AiError) -> None:
        if self._done:
            return
        self._done = True
        self._file.append(HistoryRecord.end(EndRecord(
            self._run, _now_ms(),
            Outcome.failed(error.kind))).to_dict())

    def gave_up(self) -> None:
        if self._done:
            return
        self._done = True
        self._file.append(HistoryRecord.end(EndRecord(
            self._run, _now_ms(),
            Outcome.failed(ErrorKind.NETWORK))).to_dict())


def _prompt_of(request) -> str | None:
    from .model import InputRole

    for role, payload in request.inputs:
        if role == InputRole.PROMPT and payload.as_text() is not None:
            return payload.as_text()
    return None


class AiClient:
    """The app's connection to the AI layer."""

    def __init__(self, app, shared: SharedStore, secrets=None,
                 factory: ProviderFactory | None = None,
                 catalog: Catalog | None = None) -> None:
        self.app = app if isinstance(app, str) else app.as_str()
        self.shared = shared
        self.secrets = secrets or LayeredStore([EnvStore(), MemoryStore()])
        self.factory = factory or ProviderFactory.standard()
        self._catalog = catalog or Catalog.builtin()
        self._history: HistoryFile | None = None
        self._keep_prompts = True
        self._ledger: ConsentLedger | None = None

    @classmethod
    def standard(cls, app) -> AiClient:
        """The client at the standard locations: the shared store
        file, environment + memory secret stores, the standard
        factory."""
        path = default_store_path()
        if path is None:
            raise AiError.invalid(
                "no place to keep the connections file on this system")
        return cls(app, SharedStore(path))

    def with_history(self, history_file: HistoryFile) -> AiClient:
        self._history = history_file
        return self

    def without_history(self) -> AiClient:
        self._history = None
        return self

    def with_factory(self, factory: ProviderFactory) -> AiClient:
        self.factory = factory
        return self

    def with_catalog(self, catalog: Catalog) -> AiClient:
        self._catalog = catalog
        return self

    def keep_prompts_in_history(self, keep: bool) -> AiClient:
        self._keep_prompts = keep
        return self

    # ── consent ledger ──────────────────────────────────────────

    def _consent_path(self) -> Path:
        return self.shared.path.with_name("consent.json")

    def ledger(self) -> ConsentLedger:
        """This app's consent ledger, loaded from the consent file
        beside the store file (missing file = no agreements yet)."""
        if self._ledger is not None:
            return self._ledger
        ledger = ConsentLedger()
        try:
            data = json.loads(self._consent_path().read_text(
                encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        for item in data.get("records", []):
            try:
                ledger.records.append(ConsentRecord(
                    str(item["app"]), str(item["connection"]),
                    str(item["destination"]),
                    CapabilityId.parse(str(item["capability"])),
                    set(item.get("kinds", [])),
                    item.get("expires_at")))
            except (KeyError, ValueError, TypeError):
                continue
        self._ledger = ledger
        return ledger

    def _save_ledger(self) -> None:
        ledger = self._ledger
        if ledger is None:
            return
        data = {"records": [
            {"app": r.app, "connection": r.connection,
             "destination": r.destination,
             "capability": r.capability.as_str() if r.capability else None,
             "kinds": sorted(r.kinds),
             "expires_at": r.expires_at}
            for r in ledger.records]}
        path = self._consent_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def agree(self, disclosure: Disclosure, term, now: int | None = None
              ) -> ConsentLedger:
        """Apply a person's choice for ``disclosure``. A term that
        remembers is written to the consent file; a this-time choice
        covers only the returned ledger."""
        if now is None:
            now = _now_secs()
        ledger = self.ledger()
        job_ledger = ledger.agree(disclosure, term, now)
        if term.remembers():
            self._save_ledger()
        return job_ledger

    # ── status / choices / picks ────────────────────────────────

    def status(self, role: Role) -> RoleStatus:
        """Can ``role`` run now? The reason, when it cannot, is safe
        to show."""
        try:
            store = self.shared.load()
        except Exception as exc:  # noqa: BLE001 - store read failure
            return RoleStatus.unavailable(str(exc))
        try:
            resolved = store.resolve(self.app, role)
        except Exception as exc:  # noqa: BLE001 - resolution refusal
            message = str(exc)
            if "no model is chosen" in message:
                message = (f"no model is bound for role "
                           f"'{role.id.as_str()}'; bind one in Set up AI")
            return RoleStatus.unavailable(message)
        if not self.factory.knows(resolved.connection.provider):
            return RoleStatus.unavailable(NO_ADAPTER)
        try:
            secret_for(self.secrets, resolved.connection)
        except AiError as exc:
            return RoleStatus.unavailable(exc.message)
        model = resolved.model
        if (model.local_file is not None
                and not Path(model.local_file).is_file()):
            return RoleStatus.unavailable(
                "The file for the model you chose is no longer there. "
                "Open Set up AI to fix it.")
        return RoleStatus(RoleReady(
            resolved.connection.id.as_str(),
            resolved.connection.label,
            resolved.model.id.as_str(), resolved.model.label,
            resolved.connection.locality, resolved.connection.billing))

    def status_for(self, role_id: str) -> RoleStatus:
        """Status for a role known only by id. A custom tool carries
        its own contract; a bound model supplies one otherwise; an
        unbound id is unavailable, naming binding as the fix."""
        store = self.shared.load()
        for tool in store.custom_tools:
            if tool.id.as_str() == role_id:
                return self.status(Role(
                    tool.id, tool.label, tool.capability,
                    list(tool.inputs)))
        binding = next(
            (b for b in store.defaults if b.role.as_str() == role_id),
            None)
        if binding is None:
            for settings in store.apps.values():
                binding = next(
                    (b for b in settings.get("overrides", [])
                     if b.role.as_str() == role_id), None)
                if binding is not None:
                    break
        if binding is None:
            return RoleStatus.unavailable(
                f"no model is bound for role '{role_id}'; bind one in "
                "Set up AI")
        conn = next((c for c in store.connections
                     if c.id == binding.connection), None)
        if conn is None:
            return RoleStatus.unavailable(
                f"the connection for role '{role_id}' is gone")
        model = next((m for m in conn.models
                      if m.id == binding.model), None)
        if model is None or not model.capabilities:
            return RoleStatus.unavailable(
                f"the model for role '{role_id}' is gone")
        spec = model.capabilities[0]
        return self.status(Role(binding.role, role_id, spec.capability,
                               list(spec.requires)))

    def choices(self, role: Role) -> list:
        """Every model that could fill ``role``, for a picker."""
        store = self.shared.load()
        store.check_tool(role)
        return [Choice(r.connection.id.as_str(),
                       r.connection.label,
                       r.model.id.as_str(), r.model.label,
                       r.connection.locality)
                for r in eligible(role, store.connections)]

    def picks(self, role: Role) -> list:
        """The presets (recommended, lowest cost, newest and so on)
        each model earns for ``role``, in picker order. A preset
        nobody earns is left out. Nothing is bound."""
        store = self.shared.load()
        store.check_tool(role)
        found = eligible(role, store.connections)
        candidates = [{"provider": r.connection.provider,
                       "model": r.model,
                       "free": r.connection.billing is Billing.FREE}
                      for r in found]
        out = []
        for picked in choose(role.capability, candidates, self._catalog):
            resolved = found[picked.candidate]
            out.append(PickedChoice(
                picked.pick,
                Choice(resolved.connection.id.as_str(),
                       resolved.connection.label,
                       resolved.model.id.as_str(),
                       resolved.model.label,
                       resolved.connection.locality),
                picked.by))
        return out

    def recommended(self, roles: list) -> list:
        """For each of ``roles`` that cannot run yet, the model the
        catalog recommends. This is what to offer after a connection
        tests well; it binds nothing, so a model is never chosen for
        the person without their yes."""
        out = []
        for role in roles:
            if self.status(role).is_ready:
                continue
            for picked in self.picks(role):
                if picked.pick is Pick.RECOMMENDED:
                    out.append((role.id, picked.choice))
                    break
        return out

    def bind(self, role: Role, connection, model) -> None:
        """Set this app's own choice for ``role``, overriding the
        launcher default. Refused, and nothing written, if the
        connection or model cannot fill the role."""
        from .ids import ConnectionId, ModelId
        from .role import Binding

        binding = Binding(
            role.id,
            connection if isinstance(connection, ConnectionId)
            else ConnectionId.new(connection),
            model if isinstance(model, ModelId) else ModelId.new(model))

        def _apply(store: StoreFile):
            store.check_tool(role)
            try:
                resolve(role, [[binding]], store.connections)
            except Exception as exc:  # noqa: BLE001 - resolution refusal
                raise AiError.invalid(str(exc)) from exc
            settings = store.apps.setdefault(self.app, {"overrides": []})
            settings["overrides"] = [
                b for b in settings["overrides"] if b.role != role.id]
            settings["overrides"].append(binding)

        try:
            self.shared.update(_apply)
        except AiError:
            raise
        except Exception as exc:  # noqa: BLE001 - store write failure
            raise AiError.invalid(str(exc)) from exc

    def clear_binding(self, role_id) -> None:
        """Remove this app's own choice, so the launcher default
        applies again."""
        role_str = (role_id.as_str() if hasattr(role_id, "as_str")
                    else str(role_id))

        def _apply(store: StoreFile):
            settings = store.apps.get(self.app)
            if settings is not None:
                settings["overrides"] = [
                    b for b in settings["overrides"]
                    if b.role.as_str() != role_str]

        self.shared.update(_apply)

    # ── plan / start ────────────────────────────────────────────

    def plan(self, role: Role, request) -> Plan:
        """Resolve the role and check the request. Sends nothing and
        reads no key."""
        if request.capability != role.capability:
            raise AiError.invalid(
                f"this request is for {request.capability.as_str()}, "
                f"but \"{role.label}\" needs "
                f"{role.capability.as_str()}")
        store = self.shared.load()
        try:
            resolved = store.resolve(self.app, role)
        except Exception as exc:  # noqa: BLE001 - resolution refusal
            raise AiError.unsupported(str(exc)) from exc
        request.validate(resolved.spec, Limits())
        # Refuse here, before anyone is asked to agree to send
        # something this app cannot send.
        if not self.factory.knows(resolved.connection.provider):
            raise AiError.unsupported(NO_ADAPTER)
        disclosure = Disclosure.for_request(
            self.app, resolved.connection, request)
        return Plan(resolved.connection, resolved.model.id, request,
                    disclosure, role.id, role.label, self.app)

    def start(self, plan: Plan, ledger: ConsentLedger | None = None
              ) -> Job:
        """Start the job. Refuses with consent_required if ``ledger``
        does not cover the plan's disclosure. A plan for another app
        is an invalid request, refused before key access or provider
        and history side effects."""
        if plan.app != self.app:
            raise AiError.invalid("this plan belongs to a different app")
        ledger = ledger if ledger is not None else self.ledger()
        if plan.needs_consent(ledger):
            raise AiError.new(
                ErrorKind.CONSENT_REQUIRED,
                "sending to "
                f"{plan.disclosure().destination} needs your consent "
                "first")
        provider = self._prepare(plan)
        estimate = None
        try:
            estimate = provider.estimate(plan.model(), plan.request)
        except AiError:
            estimate = None
        run = uuid.uuid4().hex
        try:
            job_id = provider.submit(plan.model(), plan.request)
        except AiError as exc:
            self._log_refused(plan, run, estimate, exc)
            raise
        recorder = None
        if self._history is not None:
            recorder = _Recorder(self._history, plan, run, estimate,
                                 self._keep_prompts)
        return Job(provider, job_id, recorder)

    def _prepare(self, plan: Plan):
        # Without an adapter there is nothing to give the key to, so
        # do not fetch it.
        if not self.factory.knows(plan.connection.provider):
            raise AiError.unsupported(NO_ADAPTER)
        secret = secret_for(self.secrets, plan.connection)
        return self.factory.build(plan.connection, secret)

    def _log_refused(self, plan: Plan, run: str, estimate, error: AiError
                     ) -> None:
        if self._history is None:
            return
        recorder = _Recorder(self._history, plan, run, estimate,
                             self._keep_prompts)
        recorder.refused(error)

    def conformance(self, role: Role) -> dict:
        """Run the conformance suite against the provider the role
        resolves to, and report per check."""
        store = self.shared.load()
        resolved = store.resolve(self.app, role)
        request = _request_for_role(role)
        request.validate(resolved.spec, Limits())
        secret = secret_for(self.secrets, resolved.connection)
        provider = self.factory.build(resolved.connection, secret)
        return check_provider(provider, resolved.model, request)

    # ── history ─────────────────────────────────────────────────

    def history(self) -> HistoryLog:
        """The merged run history, newest first."""
        if self._history is None:
            return HistoryLog([])
        read = self._history.read()
        records = []
        for raw in read.records:
            try:
                records.append(HistoryRecord.from_dict(raw))
            except (KeyError, ValueError, TypeError):
                read.skipped += 1
        return merge(records, read.skipped)

    def clear_history(self) -> None:
        if self._history is not None:
            self._history.clear()


def _request_for_role(role: Role):
    """A minimal valid request for a role's capability, for
    conformance runs: one prompt input where the role takes one."""
    from .model import InputRole
    from .payload import Payload
    from .request import Request

    request = Request.new(role.capability)
    if InputRole.PROMPT in role.inputs:
        request.with_input(InputRole.PROMPT, Payload.text("test"))
    return request
