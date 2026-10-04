"""Form-field resolution for rendered (playwright) tabs.

Real signup/login forms rarely use clean ``name`` attributes — fields are
identified by their visible label, placeholder, or aria-label. This module
holds the JavaScript that scores every candidate control on the page and
pins the best match with a ``data-nm-field="1"`` marker attribute, so the
caller can drive it through a stable CSS selector.

Scoring (exact beats substring, visible identity beats markup identity):

* ``aria-label`` exact (100) / substring (50)
* ``<label for>`` / wrapping ``<label>`` text exact (90) / substring (45)
* ``placeholder`` exact (80) / substring (40)
* ``name`` exact (70) / substring (30)
* ``id`` exact (60) / substring (20)
* ``data-testid`` exact (55)

Nothing here touches the network. ``resolve()`` needs a page object with
an ``evaluate(js, arg)`` method (playwright's real Page); when the driver
is duck-typed and lacks ``evaluate`` the caller falls back to the legacy
name/id selector.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "FIELD_MARKER",
    "FieldNotFound",
    "resolve",
    "clear_marker",
    "describe_fields",
    "set_value_js",
    "normalize_date",
]

#: marker attribute pinned on the resolved field before an action.
FIELD_MARKER = "data-nm-field"

#: stable selector for the pinned field.
MARKER_SELECTOR = f'[{FIELD_MARKER}="1"]'


class FieldNotFound(Exception):
    """No control on the page matched the requested field."""


_RESOLVER_JS = """(name) => {
  const q = String(name || '').trim().toLowerCase();
  if (!q) return null;
  const esc = (s) => (window.CSS && CSS.escape)
    ? CSS.escape(s) : String(s).replace(/"/g, '\\\\"');
  const fields = Array.from(document.querySelectorAll(
    'input:not([type="hidden"]), textarea, select, [role="textbox"], '
    + '[role="combobox"], [contenteditable="true"]'
  )).filter((el) => {
    if (el.disabled) return false;
    // hidden elements never accept typed input; keep the focused one.
    return el.offsetParent !== null || el === document.activeElement;
  });
  const norm = (s) => String(s || '').trim().toLowerCase().replace(/\\s+/g, ' ');
  const labelText = (el) => {
    if (el.id) {
      const lab = document.querySelector('label[for="' + esc(el.id) + '"]');
      if (lab) return norm(lab.innerText);
    }
    const wrap = el.closest('label');
    if (wrap) return norm(wrap.innerText);
    return '';
  };
  let best = null, bestScore = 0, bestBy = '';
  for (const el of fields) {
    const attrs = {
      aria: norm(el.getAttribute('aria-label')),
      label: labelText(el),
      placeholder: norm(el.getAttribute('placeholder')),
      name: norm(el.getAttribute('name')),
      id: norm(el.id || ''),
      testid: norm(el.getAttribute('data-testid')),
    };
    let score = 0, by = '';
    if (attrs.aria && attrs.aria === q) { score = 100; by = 'aria-label'; }
    else if (attrs.label && attrs.label === q) { score = 90; by = 'label'; }
    else if (attrs.placeholder && attrs.placeholder === q) { score = 80; by = 'placeholder'; }
    else if (attrs.name && attrs.name === q) { score = 70; by = 'name'; }
    else if (attrs.id && attrs.id === q) { score = 60; by = 'id'; }
    else if (attrs.testid && attrs.testid === q) { score = 55; by = 'data-testid'; }
    else if (attrs.aria && attrs.aria.indexOf(q) !== -1) { score = 50; by = 'aria-label~'; }
    else if (attrs.label && attrs.label.indexOf(q) !== -1) { score = 45; by = 'label~'; }
    else if (attrs.placeholder && attrs.placeholder.indexOf(q) !== -1) { score = 40; by = 'placeholder~'; }
    else if (attrs.name && attrs.name.indexOf(q) !== -1) { score = 30; by = 'name~'; }
    else if (attrs.id && attrs.id.indexOf(q) !== -1) { score = 20; by = 'id~'; }
    if (score > bestScore) { bestScore = score; best = el; bestBy = by; }
  }
  if (!best) return null;
  document.querySelectorAll('[' + '""" + FIELD_MARKER + """' + ']').forEach(
    (m) => m.removeAttribute('""" + FIELD_MARKER + """'));
  best.setAttribute('""" + FIELD_MARKER + """', '1');
  return {
    tag: (best.tagName || '').toLowerCase(),
    type: (best.getAttribute('type') || '').toLowerCase(),
    by: bestBy,
    score: bestScore,
    name: best.getAttribute('name') || '',
    id: best.id || '',
  };
}"""

_CLEAR_JS = """() => {
  document.querySelectorAll('[' + '""" + FIELD_MARKER + """' + ']').forEach(
    (m) => m.removeAttribute('""" + FIELD_MARKER + """'));
  return true;
}"""

_DESCRIBE_JS = """() => {
  const out = [];
  const esc = (s) => String(s).replace(/"/g, '\\\\"');
  const els = document.querySelectorAll(
    'input:not([type="hidden"]), textarea, select, button, [role="button"]');
  for (const el of els) {
    let label = '';
    if (el.id) {
      const lab = document.querySelector('label[for="' + esc(el.id) + '"]');
      if (lab) label = (lab.innerText || '').trim();
    }
    if (!label) {
      const wrap = el.closest('label');
      if (wrap) label = (wrap.innerText || '').trim();
    }
    out.push({
      tag: (el.tagName || '').toLowerCase(),
      type: (el.getAttribute('type') || '').toLowerCase(),
      name: el.getAttribute('name') || '',
      id: el.id || '',
      label: label.slice(0, 60),
      placeholder: (el.getAttribute('placeholder') || '').slice(0, 60),
      aria: (el.getAttribute('aria-label') || '').slice(0, 60),
    });
    if (out.length >= 100) break;
  }
  return out;
}"""

#: React/Vue-controlled-input-compatible value setter: goes through the
#: native property setter (so framework change detection fires) and then
#: dispatches input+change. Used as the fallback for date pickers and
#: other custom controls that swallow playwright's fill().
_SET_VALUE_JS = """(value) => {
  const el = document.querySelector('[' + '""" + FIELD_MARKER + """' + '="1"]');
  if (!el) return 'not-found';
  const tag = (el.tagName || '').toUpperCase();
  const proto = tag === 'TEXTAREA' ? window.HTMLTextAreaElement.prototype
    : tag === 'SELECT' ? window.HTMLSelectElement.prototype
    : window.HTMLInputElement.prototype;
  const desc = Object.getOwnPropertyDescriptor(proto, 'value');
  if (desc && desc.set) {
    desc.set.call(el, String(value));
  } else {
    el.value = String(value);
  }
  el.dispatchEvent(new Event('input', {bubbles: true}));
  el.dispatchEvent(new Event('change', {bubbles: true}));
  return 'ok';
}"""


def resolve(page: Any, name: str) -> tuple[str, dict[str, Any] | None]:
    """Resolve ``name`` to ``(selector, info)`` on the page.

    ``info`` is the resolver's ``{tag, type, by, score, name, id}`` dict,
    or ``None`` when the driver cannot run JS (duck-typed pages) — the
    caller then uses the legacy name/id selector with untyped behavior.
    Raises :class:`FieldNotFound` when nothing on the page matches.
    """
    name = (name or "").strip()
    if not name:
        raise FieldNotFound("field name must not be empty")
    evaluate = getattr(page, "evaluate", None)
    if evaluate is None:
        escaped = name.replace('"', '\\"')
        return (
            f'input[name="{escaped}"], textarea[name="{escaped}"], '
            f'select[name="{escaped}"], [id="{escaped}"]',
            None,
        )
    try:
        info = evaluate(_RESOLVER_JS, name)
    except Exception as exc:
        raise FieldNotFound(
            f"field resolver failed for {name!r}: {exc}") from exc
    if not info:
        raise FieldNotFound(f"no form field matching {name!r} on the page")
    return MARKER_SELECTOR, {
        "tag": str(info.get("tag", "")),
        "type": str(info.get("type", "")),
        "by": str(info.get("by", "")),
        "score": info.get("score", 0),
        "name": str(info.get("name", "")),
        "id": str(info.get("id", "")),
    }


def clear_marker(page: Any) -> None:
    """Remove the ``data-nm-field`` marker (best-effort, never raises)."""
    evaluate = getattr(page, "evaluate", None)
    if evaluate is None:
        return
    try:
        evaluate(_CLEAR_JS)
    except Exception:
        pass


def describe_fields(page: Any) -> list[dict[str, Any]]:
    """Every fillable control on the page with its visible identity.

    Used to make "field not found" errors actionable: the error can list
    what the page actually has instead of a bare name.
    """
    evaluate = getattr(page, "evaluate", None)
    if evaluate is None:
        return []
    try:
        raw = evaluate(_DESCRIBE_JS)
    except Exception:
        return []
    return [dict(item) for item in (raw or []) if isinstance(item, dict)]


def format_field_list(fields: list[dict[str, Any]],
                      limit: int = 12) -> str:
    """One-line-per-field summary for error messages."""
    lines = []
    for f in fields[: max(1, limit)]:
        ident = (f.get("label") or f.get("aria") or f.get("placeholder")
                 or f.get("name") or f.get("id") or "?")
        lines.append(
            f"  - <{f.get('tag')}> type={f.get('type') or '-'} "
            f"name={f.get('name') or '-'} label={ident!r}")
    if len(fields) > limit:
        lines.append(f"  ... and {len(fields) - limit} more")
    return "\n".join(lines)


def set_value_js(page: Any, value: str) -> str:
    """Set the resolved (marker-pinned) field's value via JS.

    Returns the JS result ("ok"/"not-found"). Raises the page's own
    error when evaluation fails.
    """
    return page.evaluate(_SET_VALUE_JS, str(value))


#: date formats accepted by set_date(); normalized to ISO for native inputs.
_DATE_FORMATS = (
    "%Y-%m-%d",       # 2026-10-04
    "%Y/%m/%d",       # 2026/10/04
    "%d-%m-%Y",       # 04-10-2026
    "%d/%m/%Y",       # 04/10/2026
    "%m/%d/%Y",       # 10/04/2026
    "%d %b %Y",       # 4 Oct 2026
    "%d %B %Y",       # 4 October 2026
    "%b %d, %Y",      # Oct 4, 2026
    "%B %d, %Y",      # October 4, 2026
)


def normalize_date(value: str) -> str:
    """Validate a human date string and return ISO ``YYYY-MM-DD``.

    Ambiguous numeric dates (e.g. "10/04/2026") parse day-first (DD/MM);
    month-first input must be unambiguous (e.g. "12/25/2026"). Raises
    ValueError listing the accepted formats — never a silently wrong
    date.
    """
    from datetime import datetime

    text = (value or "").strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    raise ValueError(
        f"cannot parse date {value!r}; use YYYY-MM-DD "
        "(also accepted: DD/MM/YYYY, MM/DD/YYYY, '4 Oct 2026')")
