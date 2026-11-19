"""Spec templates (Craft-Libs port of ai-core ``spec/template.rs``).

The only substitutions a spec may make: ``{prompt}``, ``{model}``
and ``{job_id}``. The structure of a request always comes from
the spec. A substitution replaces text inside a string value of
a JSON template, so a prompt can never add a field or change a
key, and the inserted text is not read again for more
substitutions. ``{{`` and ``}}`` write a literal brace.
"""

from __future__ import annotations

from .limits import MAX_RENDERED_BYTES

__all__ = ["Slot", "render_text", "render_value", "slots",
           "value_slots"]

_SLOTS = ("prompt", "model", "job_id")


class Slot:
    PROMPT = "prompt"
    MODEL = "model"
    JOB_ID = "job_id"


def _pieces(text: str):
    """Split ``text`` into literal text and slots, or say why its
    braces are wrong. Returns a list of ("text", s) / ("slot", name)
    pairs."""
    out = []
    literal: list[str] = []
    rest = text
    while True:
        cut = len(rest)
        for i, char in enumerate(rest):
            if char in "{}":
                cut = i
                break
        literal.append(rest[:cut])
        rest = rest[cut:]
        if not rest:
            break
        here = rest
        after = rest[1:]
        if here.startswith("{{") or here.startswith("}}"):
            literal.append(here[0])
            rest = after[1:]
        elif here.startswith("}"):
            raise ValueError("a lone } (write }} for a literal brace)")
        else:
            end = after.find("}")
            if end < 0:
                raise ValueError(
                    "a { that is never closed (write {{ for a literal "
                    "brace)")
            name = after[:end]
            if name not in _SLOTS:
                raise ValueError(f"unsupported: placeholder {{{name}}}")
            if literal:
                out.append(("text", "".join(literal)))
                literal = []
            out.append(("slot", name))
            rest = after[end + 1:]
    if literal:
        out.append(("text", "".join(literal)))
    return out


def slots(text: str) -> list[str]:
    """The slots ``text`` uses, or why its braces are wrong."""
    return [name for kind, name in _pieces(text) if kind == "slot"]


def value_slots(template) -> list[str]:
    """Every slot used by a string value anywhere in ``template``.
    Object keys must be literal."""
    out: list[str] = []
    if isinstance(template, str):
        out.extend(slots(template))
    elif isinstance(template, list):
        for item in template:
            out.extend(value_slots(item))
    elif isinstance(template, dict):
        for key, item in template.items():
            if "{" in key or "}" in key:
                raise ValueError(
                    f"a key cannot hold a brace ({key!r})")
            out.extend(value_slots(item))
    return out


def render_text(text: str, prompt: str | None = None,
                model: str | None = None,
                job_id: str | None = None) -> str:
    """Render one string. Fails past
    :data:`MAX_RENDERED_BYTES`."""
    left = [MAX_RENDERED_BYTES]
    return _render_text_within(text, prompt, model, job_id, left)


def _render_text_within(text: str, prompt, model, job_id, left) -> str:
    values = {"prompt": prompt, "model": model, "job_id": job_id}
    out: list[str] = []
    for kind, piece in _pieces(text):
        if kind == "text":
            add = piece
        else:
            add = values.get(piece)
            if add is None:
                raise ValueError(
                    f"{{{piece}}} is not available in this call")
        if len(add.encode("utf-8")) > left[0]:
            raise ValueError("a rendered template is over 4 MiB")
        left[0] -= len(add.encode("utf-8"))
        out.append(add)
    return "".join(out)


def render_value(template, prompt: str | None = None,
                 model: str | None = None,
                 job_id: str | None = None):
    """Render a whole JSON template value, sharing one 4 MiB
    budget over every string."""
    left = [MAX_RENDERED_BYTES]
    return _render_value_within(template, prompt, model, job_id, left)


def _render_value_within(template, prompt, model, job_id, left):
    if isinstance(template, str):
        return _render_text_within(template, prompt, model, job_id, left)
    if isinstance(template, list):
        return [_render_value_within(item, prompt, model, job_id, left)
                for item in template]
    if isinstance(template, dict):
        out = {}
        for key, item in template.items():
            if "{" in key or "}" in key:
                raise ValueError(
                    f"a key cannot hold a brace ({key!r})")
            out[key] = _render_value_within(item, prompt, model, job_id,
                                            left)
        return out
    return template
