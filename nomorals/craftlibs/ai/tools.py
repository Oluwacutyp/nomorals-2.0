"""The tool desk (Craft-AI port of ai-client ``tools.rs`` +
``tools_sample.rs``).

The launcher's side of the client: the suite-wide (launcher
default) choice for each tool, custom-tool registration, and sample
runs. The desk writes defaults, never app overrides — an app's own
choice is the client's :meth:`~nomorals.craftlibs.ai.client.AiClient.bind`.
A choice made by another app a moment ago is never overwritten:
``choose_if_unchosen`` checks inside the same store lock as the
write. ``auto_bind_offer`` writes nothing; ``accept_recommended``
applies exactly what was offered and skips any tool that got a
choice in the meantime, one tool failing never stops the others.
Built-in tool ids are passed in (never imported from ``tools/`` —
the tool action passes the registry's ids), so a custom tool can
never take a built-in name.
"""

from __future__ import annotations

import time

from .client import AiClient
from .connection import Locality
from .consent import Disclosure
from .consent_term import ConsentTerm
from .error import AiError
from .role import Binding, Role, resolve
from .segment import CAPABILITY as SEGMENT_CAPABILITY
from .segment import FOUND_MEDIA_TYPE, read_found
from .shared import SharedStore
from .store import CustomTool, StoreFile

__all__ = [
    "Accepted",
    "MissingModel",
    "SampleReport",
    "Skipped",
    "TestStart",
    "ToolDesk",
    "ToolView",
]


class ToolView:
    """What the launcher shows for a tool: every model that could
    fill it, the launcher default (if any), and whether it can run
    now."""

    def __init__(self, role: Role, choices: list, bound,
                 status) -> None:
        self.role = role
        self.choices = choices
        self.bound = bound
        self.status = status


class TestStart:
    """How a sample started: a running job, or a disclosure the
    person must agree to first."""

    __test__ = False

    def __init__(self, job=None, disclosure: Disclosure | None = None,
                 connection_label: str | None = None,
                 billing=None) -> None:
        self.job = job
        self.disclosure = disclosure
        self.connection_label = connection_label
        self.billing = billing

    @property
    def is_started(self) -> bool:
        return self.job is not None

    @classmethod
    def started(cls, job) -> TestStart:
        return cls(job=job)

    @classmethod
    def needs_consent(cls, disclosure: Disclosure,
                      connection_label: str, billing) -> TestStart:
        return cls(disclosure=disclosure,
                   connection_label=connection_label, billing=billing)


class SampleReport:
    """How a sample went."""

    def __init__(self, passed: bool, summary: str | None = None,
                 message: str | None = None, millis: int = 0) -> None:
        self.passed = passed
        self.summary = summary
        self.message = message
        self.millis = millis

    @classmethod
    def passed(cls, summary: str, millis: int) -> SampleReport:
        return cls(True, summary=summary, millis=millis)

    @classmethod
    def failed(cls, message: str) -> SampleReport:
        return cls(False, message=message)


class Skipped:
    """A recommended model that was not applied."""

    def __init__(self, tool, why: str) -> None:
        self.tool = tool
        self.why = why


class Accepted:
    """What saying yes to the recommended models did."""

    def __init__(self, bound: list | None = None,
                 skipped: list | None = None) -> None:
        self.bound = list(bound or [])
        self.skipped = list(skipped or [])


class MissingModel:
    """What the app's popup shows when a tool cannot run."""

    def __init__(self, tool, title: str, reason: str, picker: list
                 ) -> None:
        self.tool = tool
        self.title = title
        self.reason = reason
        self.picker = picker


def _role_id(role) -> str:
    return role.id.as_str() if hasattr(role, "id") else str(role)


def _media_kind(payload: dict) -> str:
    kind = payload.get("type")
    if kind == "text":
        return "text"
    media_type = payload.get("media_type", "")
    return media_type.split("/")[0] if media_type else ""


class ToolDesk:
    """The launcher's desk over one client and the shared store."""

    def __init__(self, client: AiClient, shared: SharedStore,
                 builtin_ids=()) -> None:
        self.client = client
        self.shared = shared
        self._builtin_ids = {str(i) for i in builtin_ids}

    def missing_model(self, role: Role):
        """What the app's popup shows when ``role`` cannot run, or
        ``None`` when it can."""
        status = self.client.status(role)
        if status.is_ready:
            return None
        return MissingModel(role.id, role.label, status.reason or "",
                            self.client.choices(role))

    def view(self, role: Role) -> ToolView:
        store = self.shared.load()
        bound = next(
            ((b.connection.as_str(), b.model.as_str())
             for b in store.defaults if b.role == role.id),
            None)
        return ToolView(role, self.client.choices(role), bound,
                        self.client.status(role))

    def choose(self, role: Role, connection, model) -> None:
        """Make ``connection`` and ``model`` the launcher's choice
        for ``role``. Refused, with nothing written, when that model
        cannot do the job."""
        self._write_choice(role, connection, model, False)

    def choose_if_unchosen(self, role: Role, connection, model) -> bool:
        """Choose the model only if the tool has no suite-wide
        choice, checked inside the same store lock as the write, so
        a choice made by another app a moment ago is never
        overwritten. Returns ``False``, writing nothing, when a
        choice already exists."""
        return self._write_choice(role, connection, model, True)

    def _write_choice(self, role: Role, connection, model,
                      only_if_unchosen: bool) -> bool:
        from .ids import ConnectionId, ModelId

        binding = Binding(
            role.id,
            connection if isinstance(connection, ConnectionId)
            else ConnectionId.new(connection),
            model if isinstance(model, ModelId) else ModelId.new(model))

        def _apply(store: StoreFile):
            if (only_if_unchosen
                    and any(b.role == role.id for b in store.defaults)):
                return False
            store.check_tool(role)
            try:
                resolve(role, [[binding]], store.connections)
            except Exception as exc:  # noqa: BLE001 - resolution refusal
                raise AiError.invalid(str(exc)) from exc
            store.defaults = [b for b in store.defaults
                              if b.role != role.id]
            store.defaults.append(binding)
            return True

        return self.shared.update(_apply)

    def auto_bind_offer(self, roles: list) -> list:
        """What could be chosen for every tool with no model chosen:
        the best connected one, on this computer first, then the
        first hosted one. **Nothing is written.** A choice the person
        made is never offered over, even one that no longer works, so
        it is still there when the connection comes back."""
        offer = []
        for role in roles:
            view = self.view(role)
            if view.bound is not None:
                continue
            pick = next(
                (c for c in view.choices
                 if c.locality is Locality.LOCAL),
                view.choices[0] if view.choices else None)
            if pick is not None:
                offer.append((role.id, pick))
        return offer

    def accept_recommended(self, roles: list) -> Accepted:
        """The person said yes to "use the recommended models for N
        features?": makes each offered model the launcher's choice
        for its tool. A tool the person has since chosen a model for
        keeps their choice, a model that has gone away is skipped,
        and one tool failing never stops the others. Each skipped
        tool says why."""
        done = Accepted()
        offer = self.auto_bind_offer(roles)
        by_id = {r.id.as_str(): r for r in roles}
        for tool_id, choice in offer:
            tool = tool_id.as_str()
            role = by_id.get(tool)
            if role is None:
                done.skipped.append(Skipped(
                    tool_id, "This app no longer has that feature."))
                continue
            if self.view(role).bound is not None:
                done.skipped.append(Skipped(
                    tool_id,
                    "A model was chosen for this feature in the "
                    "meantime."))
                continue
            try:
                wrote = self.choose_if_unchosen(
                    role, choice.connection, choice.model)
            except AiError:
                done.skipped.append(Skipped(
                    tool_id, "That model is no longer available."))
                continue
            if wrote:
                done.bound.append(tool_id)
            else:
                done.skipped.append(Skipped(
                    tool_id,
                    "A model was chosen for this feature in the "
                    "meantime."))
        return done

    def custom_tools(self) -> list:
        """Custom tool declarations shared with every Craft app;
        declaring one does not install an adapter."""
        store = self.shared.load()
        return [Role(t.id, t.label, t.capability, list(t.inputs))
                for t in store.custom_tools]

    def register_tool(self, role: Role) -> None:
        """Register a user-defined tool across the suite, without
        changing the built-in catalogue. A name collision is refused
        rather than replacing a tool or its existing binding."""
        if role.id.as_str() in self._builtin_ids:
            raise AiError.invalid(
                "that tool name is already used by a built-in tool")

        def _apply(store: StoreFile):
            if any(t.id == role.id for t in store.custom_tools):
                raise AiError.invalid("that tool name is already in use")
            store.custom_tools.append(CustomTool(
                role.id, role.label, role.capability, list(role.inputs)))

        self.shared.update(_apply)

    def remove_tool(self, tool_id) -> None:
        """Remove a custom tool and its stored choices from every
        app. Built-in tools cannot be removed."""
        role = _role_id(tool_id)

        def _apply(store: StoreFile):
            if not any(t.id.as_str() == role for t in store.custom_tools):
                raise AiError.invalid("that custom tool does not exist")
            store.custom_tools = [t for t in store.custom_tools
                                  if t.id.as_str() != role]
            store.defaults = [b for b in store.defaults
                              if b.role.as_str() != role]
            for settings in store.apps.values():
                settings["overrides"] = [
                    b for b in settings["overrides"]
                    if b.role.as_str() != role]

        self.shared.update(_apply)

    def clear(self, role) -> None:
        """Forget the launcher's choice for a tool."""
        role_id = _role_id(role)

        def _apply(store: StoreFile):
            store.defaults = [b for b in store.defaults
                              if b.role.as_str() != role_id]

        self.shared.update(_apply)

    def sample(self, role: Role, request, reviewed=None) -> TestStart:
        """Plan a sample without approval when ``reviewed`` is
        ``None``. Show the returned ``NeedsConsent`` disclosure,
        retain it, and pass it back after agreement. If the current
        disclosure differs, asks again without saving consent or
        building a provider. An existing agreement can still cover
        the new plan."""
        plan = self.client.plan(role, request)
        ledger = self.client.ledger()
        if plan.needs_consent(ledger):
            if reviewed is None or reviewed != plan.disclosure():
                return TestStart.needs_consent(
                    plan.disclosure(), plan.connection_label(),
                    plan.connection.billing)
            self.client.agree(plan.disclosure(),
                              ConsentTerm.until_revoked())
        return TestStart.started(self.client.start(plan, ledger))

    @staticmethod
    def finish(role: Role, job, timeout: float = 60.0) -> SampleReport:
        """Wait for a started sample and say whether it produced
        something of the right kind."""
        started = time.monotonic()
        try:
            done = job.wait(0.5, timeout)
        except AiError as exc:
            return SampleReport.failed(exc.message)
        millis = int((time.monotonic() - started) * 1000)
        if done.state.kind == "failed":
            error = getattr(done.state, "error", None)
            return SampleReport.failed(
                error.message if error is not None
                else "the test failed")
        if done.state.kind == "cancelled":
            return SampleReport.failed("the test was cancelled")
        if done.state.kind != "succeeded":
            return SampleReport.failed("the test did not finish")
        outputs = [event["payload"]
                   for event in done.events
                   if event.get("type") == "output"]
        if not outputs:
            return SampleReport.failed(
                "the model finished but returned nothing")
        # A selection that ran but chose nothing did not do the job,
        # however well it answered. Only the small result-metadata
        # outputs are copied, never a picture or video.
        if role.capability.as_str() == SEGMENT_CAPABILITY:
            answers = [event for event in done.events
                       if event.get("type") == "output"
                       and event.get("payload", {}).get("media_type")
                       == FOUND_MEDIA_TYPE]
            found = read_found(answers)
            if found is None:
                return SampleReport.failed(
                    "the model returned a mask but did not say whether "
                    "it selected anything")
            if not found.matched:
                return SampleReport.failed(
                    "the model ran but selected nothing in the test "
                    "picture")
        wanted = role.capability.modality().as_str()
        right_kind = sum(1 for p in outputs
                         if _media_kind(p) == wanted)
        if right_kind == 0:
            return SampleReport.failed(
                f"the model returned something other than {wanted} "
                "output")
        return SampleReport.passed(
            f"Returned {right_kind} {wanted} result(s)", millis)
