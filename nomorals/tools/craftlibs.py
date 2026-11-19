"""Craft-Libs: the shared libraries, as a tool.

A thin surface over :mod:`nomorals.craftlibs` — the organ owns the
ZIP package codec, both noise engines, the ai-core data model, the
PDF primitives, the i18n formatting rung and the shared parity +
attribution runners; the tool only routes arguments and shapes the
reply. Every action reads or builds bytes in memory (or runs the
local motion provider); nothing here writes a file, sends data
anywhere or keeps state, so its declared effects are none.
"""

from __future__ import annotations

import json
from typing import Any

from ..core.policy import Capability
from ..craftlibs import ai as _ai
from ..craftlibs import codec as _codec
from ..craftlibs import i18n as _i18n
from ..craftlibs import noise as _noise
from ..craftlibs import pdf as _pdf
from ..craftlibs.attribution import verify_attribution
from ..craftlibs.parity import run_parity

__all__ = ["craftlibs_action", "register", "run"]

_ACTIONS = ("noise", "codec", "pdf", "i18n", "ai", "parity",
            "attribution", "status")


def _as_jsonable(value):
    """A noise result as plain JSON values (numpy arrays become
    nested lists)."""
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, tuple):
        return [_as_jsonable(v) for v in value]
    return value


def register(registry: Any) -> None:
    """Attach the craftlibs family to a registry."""
    @registry.register(
        "craftlibs",
        description=(
            "The shared libraries — ZIP package codec, both noise "
            "engines, ai-core data model, PDF primitives, i18n "
            "formatting rung, parity + attribution runners. "
            "Craft-Libs actions: noise=<sample a noise engine, "
            "function=<point_value|point_perlin|point_fbm|point_worley|"
            "point_cellular|grid_perlin|grid_value_noise|grid_fbm|"
            "grid_worley>, args=<list>>; codec=<round-trip a document, "
            "document=<dict>>; pdf=<safe_url=<url> | write=<text>, "
            "title=<str>>; i18n=<format=<number>, locale=<tag>, "
            "decimals=<n> | parse=<text>, locale=<tag> | "
            "file_size=<bytes>, style=<binary|decimal|iec>>; "
            "ai=<suggest=<prompt>, kind=<motion|easing> | models | "
            "conformance | status=<role> | bind=<role, connection, "
            "model> | plan=<role, prompt> | tools=<role> | "
            "http=<base> | consent=<role, prompt, term>>; parity=<run "
            "the machine-checked parity TOML>; attribution=<verify the "
            "SHA-256 attribution DB>; status=<library status>. action="
            "<noise|codec|pdf|i18n|ai|parity|attribution|status>."
        ),
        capability=Capability.FS_READ,
        effects=0,
    )
    def craftlibs_tool(**kwargs: Any) -> dict[str, Any]:
        return craftlibs_action(**kwargs)


def run(ctx: Any, action: str, args: dict | None = None
        ) -> dict[str, Any]:
    """Call the craftlibs tool as ``run(ctx, action, args)``: the
    same entry the brain's other tools expose, so a caller can
    drive it without a registry."""
    return craftlibs_action(action, **(args or {}))


def _role_for_tool(client, role_id: str, connection_id: str = "",
                   model_id: str = ""):
    """The role the tool acts on: a custom tool carries its own
    contract; otherwise the named (or bound) model supplies its
    first capability and required inputs."""
    store = client.shared.load()
    for tool in store.custom_tools:
        if tool.id.as_str() == role_id:
            return _ai.Role(tool.id, tool.label, tool.capability,
                            list(tool.inputs))
    conn = None
    model = None
    if connection_id:
        conn = next((c for c in store.connections
                     if c.id.as_str() == connection_id), None)
        if conn is None:
            raise ValueError(f"no connection {connection_id!r}")
        if model_id:
            model = next((m for m in conn.models
                          if m.id.as_str() == model_id), None)
            if model is None:
                raise ValueError(
                    f"no model {model_id!r} on {connection_id!r}")
    if model is None:
        binding = next(
            (b for b in store.defaults if b.role.as_str() == role_id),
            None)
        if binding is not None:
            conn = next((c for c in store.connections
                         if c.id == binding.connection), None)
            if conn is not None:
                model = next((m for m in conn.models
                              if m.id == binding.model), None)
    if model is None or not model.capabilities:
        raise ValueError(
            f"role {role_id!r} has no model to describe it; bind one "
            "first")
    spec = model.capabilities[0]
    return _ai.Role(role_id, role_id, spec.capability,
                    list(spec.requires))


def _ai_client_action(kind: str, role: str = "", base: str = "",
                     connection: str = "", model: str = "",
                     prompt: str = "", term: str = "") -> dict[str, Any]:
    """The ai-client kinds: bind a role, check its status, plan a
    request, view the desk, show the HTTP policy, agree to send.
    The client is the standard one, so the store file is the
    suite's shared connections file."""
    from ..craftlibs.ai.client import AiClient
    from ..craftlibs.ai.consent_term import ConsentTerm
    from ..craftlibs.ai.http import proxy_for, tls_cafile
    from ..craftlibs.ai.tools import ToolDesk

    if kind == "http":
        base_url = base or "https://api.openai.com/v1"
        proxy = proxy_for(base_url)
        return {"base": base_url, "proxy": proxy,
                "tls_cafile": tls_cafile(),
                "reply": (f"AI calls to {base_url} use the shared TLS "
                          f"policy and proxy {proxy or 'none'}.")}
    if not role:
        raise ValueError(f"craftlibs ai {kind} needs role=<id>")
    client = AiClient.standard("craftlibs")
    if kind == "status":
        status = client.status_for(role)
        if status.is_ready:
            ready = status.ready
            return {"status": "ready", "role": role,
                    "connection": ready.connection,
                    "connection_label": ready.connection_label,
                    "model": ready.model,
                    "locality": ready.locality.value,
                    "reply": (f"{role} runs on {ready.model} "
                              f"({ready.connection_label}).")}
        return {"status": "unavailable", "role": role,
                "reason": status.reason,
                "reply": f"{role} cannot run: {status.reason}"}
    if kind == "bind":
        if not connection or not model:
            raise ValueError(
                "craftlibs ai bind needs connection=<id> model=<id>")
        tool_role = _role_for_tool(client, role, connection, model)
        client.bind(tool_role, connection, model)
        return {"bound": True, "role": role, "connection": connection,
                "model": model,
                "reply": f"{role} is now bound to {model}."}
    if kind == "tools":
        tool_role = _role_for_tool(client, role)
        desk = ToolDesk(client, client.shared)
        view = desk.view(tool_role)
        return {"role": role,
                "choices": [{"connection": c.connection.as_str(),
                             "model": c.model.as_str(),
                             "label": c.model_label,
                             "locality": c.locality.value}
                            for c in view.choices],
                "bound": view.bound,
                "status": ("ready" if view.status.is_ready
                           else "unavailable"),
                "reply": (f"{role}: {len(view.choices)} models could "
                          f"fill it, "
                          f"{'bound to ' + view.bound[1] if view.bound else 'unbound'}.")}
    tool_role = _role_for_tool(client, role)
    request = _ai.Request(tool_role.capability)
    if (_ai.InputRole.PROMPT in tool_role.inputs
            or not tool_role.inputs):
        request = request.with_input(
            _ai.InputRole.PROMPT,
            _ai.Payload.text(prompt or "hello"))
    plan = client.plan(tool_role, request)
    if kind == "plan":
        return {"role": role,
                "connection": plan.connection_label(),
                "model": plan.model().as_str(),
                "needs_consent": plan.needs_consent(client.ledger()),
                "disclosure": plan.disclosure().to_dict(),
                "reply": (f"plan for {role}: {plan.model().as_str()} on "
                          f"{plan.connection_label()}"
                          f"{' — needs consent' if plan.needs_consent(client.ledger()) else ''}.")}
    if kind == "consent":
        terms = {"this_time": ConsentTerm.this_time(),
                 "one_hour": ConsentTerm.one_hour(),
                 "one_day": ConsentTerm.one_day(),
                 "until_revoked": ConsentTerm.until_revoked()}
        chosen = terms.get((term or "until_revoked").lower())
        if chosen is None:
            raise ValueError(
                f"unknown consent term {term!r}; use one of "
                f"{', '.join(terms)}")
        client.agree(plan.disclosure(), chosen)
        return {"agreed": chosen.id(), "role": role,
                "reply": f"consent recorded for {role} ({chosen.id()})."}
    raise ValueError(f"unknown craftlibs ai kind {kind!r}")


def craftlibs_action(action: str = "status",
                   function: str = "",
                   args: list | None = None,
                   document: dict | None = None,
                   safe_url: str = "",
                   write: str = "",
                   title: str = "",
                   locale: str = "en",
                   format: float | None = None,
                   parse: str = "",
                   decimals: int = 2,
                   file_size: int | None = None,
                   style: str = "binary",
                   suggest: str = "",
                   kind: str = "motion",
                   role: str = "",
                   base: str = "",
                   connection: str = "",
                   model: str = "",
                   prompt: str = "",
                   term: str = "",
                   ) -> dict[str, Any]:
    action = (action or "status").lower()
    if action not in _ACTIONS:
        raise ValueError(f"unknown craftlibs action {action!r}; "
                         f"use one of {', '.join(_ACTIONS)}")
    if action == "noise":
        if not function:
            raise ValueError("craftlibs noise needs function=<name>")
        fn = getattr(_noise, function, None)
        if fn is None or not callable(fn):
            raise ValueError(f"no noise function {function!r}")
        if function == "point_fbm":
            call_args = list(args or [])
            if len(call_args) == 4 \
                    and isinstance(call_args[3], str):
                call_args[3] = getattr(_noise, call_args[3])
            value = fn(*call_args)
        else:
            value = fn(*(args or []))
        return {"function": function, "value": _as_jsonable(value),
                "reply": f"{function} = {_as_jsonable(value)}."}
    if action == "codec":
        doc = document or {"title": "craftlibs"}
        raw = json.dumps(doc, sort_keys=True).encode()
        package = _codec.write_package(
            {"document.json": raw,
             "meta.json": b'{"format_version": 1}'})
        back = _codec.read_package(package)
        same = back["document.json"] == raw
        return {"bytes": len(package), "round_trip": same,
                "reply": (f"codec round trip "
                          f"{'byte-identical' if same else 'CHANGED'} "
                          f"({len(package)} bytes).")}
    if action == "pdf":
        if safe_url:
            ok = _pdf.safe_url(safe_url)
            return {"url": safe_url, "safe": ok,
                    "reply": (f"{safe_url} is "
                              f"{'a safe link target' if ok else 'refused'}.")}
        if write:
            meta = _pdf.Metadata(creator="craftlibs",
                                 title=title or "Craftlibs")
            data = _pdf.write_pdf(write, meta)
            info = _pdf.read_pdf(data)
            return {"bytes": len(data), "pages": info["pages"],
                    "title": info["title"],
                    "text": info["text"][:200],
                    "reply": (f"PDF written ({len(data)} bytes, "
                              f"{info['pages']} page) and read back.")}
        raise ValueError("craftlibs pdf needs safe_url=<url> or "
                         "write=<text>")
    if action == "i18n":
        if parse:
            value = _i18n.parse_number(parse, locale)
            return {"text": parse, "locale": locale, "value": value,
                    "reply": f"{parse!r} in {locale} is {value}."}
        if file_size is not None:
            styles = {"binary": _i18n.FileSize.BINARY,
                      "decimal": _i18n.FileSize.DECIMAL,
                      "iec": _i18n.FileSize.IEC}
            chosen = styles.get(style.lower())
            if chosen is None:
                raise ValueError(
                    f"unknown file size style {style!r}")
            fmt = _i18n.Format.from_tag(locale)
            text = _i18n.file_size(fmt, int(file_size), chosen)
            return {"bytes": int(file_size), "size": text,
                    "reply": f"{file_size} bytes is {text}."}
        if format is not None:
            fmt = _i18n.Format.from_tag(locale)
            text = fmt.number(float(format), int(decimals))
            return {"value": format, "locale": locale,
                    "formatted": text,
                    "reply": f"{format} in {locale} is {text}."}
        raise ValueError("craftlibs i18n needs format=<number>, "
                         "parse=<text> or file_size=<bytes>")
    if action == "ai":
        if kind in ("status", "bind", "plan", "tools", "http",
                    "consent"):
            return _ai_client_action(kind, role=role, base=base,
                                     connection=connection,
                                     model=model, prompt=prompt,
                                     term=term)
        if suggest:
            provider = _ai.LocalMotionProvider()
            capability = ("motion.suggest" if kind != "easing"
                          else "motion.ease")
            model_id = ("local-motion" if kind != "easing"
                        else "local-motion-ease")
            request = _ai.Request(
                _ai.CapabilityId.parse(capability),
                [(_ai.InputRole.PROMPT,
                  _ai.Payload.text(suggest))])
            job_id = provider.submit(_ai.ModelId.new(model_id),
                                     request)
            snapshot = _ai.JobSnapshot.from_dict(
                provider.poll(job_id))
            outputs = snapshot.outputs()
            text = (outputs[0]["payload"]["text"]
                    if outputs else "")
            suggestion = json.loads(text) if text else {}
            return {"prompt": suggest, "kind": kind,
                    "suggestion": suggestion,
                    "reply": (f"suggestion for {suggest!r}: "
                              f"{suggestion.get('preset', '')} "
                              f"({suggestion.get('easing', '')}).")}
        if kind == "models":
            provider = _ai.LocalMotionProvider()
            models = [{"id": m.id.as_str(), "label": m.label,
                       "capability": m.spec.capability.as_str()}
                      for m in provider.models()]
            return {"models": models,
                    "reply": f"{len(models)} local models."}
        if kind == "conformance":
            from ..craftlibs.ai.conformance import (
                run_cancel,
                run_lifecycle,
                run_rejects,
                run_unknown_job,
            )
            from ..craftlibs.ai.mock import MockProvider
            model = _ai.ModelInfo.new(
                "mock-motion", "Mock motion").with_spec(
                _ai.CapabilitySpec.new(
                    _ai.CapabilityId.parse("motion.suggest"))
                .accepting(_ai.InputRole.PROMPT, 1)
                .requiring(_ai.InputRole.PROMPT)
                .with_max_outputs(1))
            provider = MockProvider(
                [model],
                {"running_polls": 2,
                 "text_chunks": ["Hello ", "world"],
                 "outcome": ("succeed",
                             [_ai.Payload.text("Hello world")])})
            request = _ai.Request(
                _ai.CapabilityId.parse("motion.suggest"),
                [(_ai.InputRole.PROMPT,
                  _ai.Payload.text("a ball bouncing"))])
            lifecycle = run_lifecycle(
                provider,
                _ai.ModelId.new("mock-motion"), request)
            run_cancel(provider, _ai.ModelId.new("mock-motion"),
                       request)
            run_rejects(provider, _ai.ModelId.new("mock-motion"),
                        _ai.Request(
                            _ai.CapabilityId.parse("text.complete"),
                            [(_ai.InputRole.PROMPT,
                              _ai.Payload.text("rewrite this"))]),
                        _ai.ErrorKind.UNSUPPORTED)
            run_unknown_job(provider)
            return {"lifecycle": lifecycle.state.kind,
                    "events": len(lifecycle.events),
                    "reply": "mock provider passes lifecycle, "
                             "cancel, rejects and unknown-job."}
        raise ValueError("craftlibs ai needs suggest=<prompt>, "
                         "kind=models or kind=conformance")
    if action == "parity":
        result = run_parity()
        return {"passed": result["passed"],
                "failed": result["failed"],
                "total": result["total"],
                "reply": (f"parity: {result['passed']}/"
                          f"{result['total']} passed.")}
    if action == "attribution":
        result = verify_attribution()
        return {"verified": result["verified"],
                "failed": result["failed"],
                "skipped": result["skipped"],
                "reply": (f"attribution: {result['verified']} "
                          "verified, "
                          f"{len(result['failed'])} failed.")}
    result = run_parity()
    attribution = verify_attribution()
    return {"tools": 1, "modules": ["codec", "noise", "ai", "pdf",
                                    "i18n", "parity", "attribution"],
            "parity_passed": result["passed"],
            "parity_total": result["total"],
            "attribution_verified": attribution["verified"],
            "reply": ("craftlibs: shared codec, both noise engines, "
                      "ai-core, PDF primitives, i18n formatting, "
                      "parity + attribution runners.")}
