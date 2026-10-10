"""Form-field resolution for rendered (playwright) tabs.

Real signup/login forms rarely use clean ``name`` attributes — fields are
identified by their visible label, placeholder, or aria-label. This module
holds the JavaScript that scores every candidate control on the page and
pins the best match with a ``data-nm-field="1"`` marker attribute, so the
caller can drive it through a stable CSS selector.

The resolver walks the *deep* DOM: open shadow roots (custom elements from
Shoelace, Ionic, Material Web, Salesforce LWC, ...) are pierced
recursively, and same-origin iframes are descended into — a plain
``document.querySelectorAll`` stops at the first shadow boundary and would
miss those fields entirely. Closed shadow roots stay unreachable (the page
itself chose that); the candidates report flags what was seen so the
failure is diagnosable.

Scoring (exact beats substring, visible identity beats markup identity):

* ``aria-label`` exact (100) / substring (50)
* ``<label for>`` / wrapping ``<label>`` text exact (90) / substring (45)
* ``placeholder`` exact (80) / substring (40)
* ``name`` exact (70) / substring (30)
* ``id`` exact (60) / substring (20)
* ``data-testid`` exact (55) / substring (15)
* token overlap (every query word present, any order): +10

Nothing here touches the network. ``resolve()`` needs a page object with
an ``evaluate(js, arg)`` method (playwright's real Page); when the driver
is duck-typed and lacks ``evaluate`` the caller falls back to the legacy
name/id selector.
"""

from __future__ import annotations

from typing import Any

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "FIELD_MARKER",
    "FieldNotFound",
    "resolve",
    "resolve_candidates",
    "clear_marker",
    "describe_fields",
    "describe_forms",
    "format_field_list",
    "set_value_js",
    "read_value_js",
    "normalize_date",
]

#: marker attribute pinned on the resolved field before an action.
FIELD_MARKER = "data-nm-field"

#: stable selector for the pinned field.
MARKER_SELECTOR = f'[{FIELD_MARKER}="1"]'


class FieldNotFound(Exception):
    """No control on the page matched the requested field.

    ``candidates`` carries the closest-ranked controls the page actually
    has (may be empty) so the caller can suggest "did you mean …?" instead
    of failing with a bare name.
    """

    def __init__(self, message: str,
                 candidates: list[dict[str, Any]] | None = None) -> None:
        super().__init__(message)
        self.candidates: list[dict[str, Any]] = list(candidates or [])


#: Deep-DOM walker shared by the resolver and describe scripts: visits
#: every element matching ``sel`` under ``root``, piercing open shadow
#: roots recursively and descending into same-origin iframes. Cross-origin
#: frames are skipped (the page's own security boundary — never worked
#: around); closed shadow roots are invisible by design.
#: controls the resolver scores (also used by the pin script — the two
#: must walk the identical selector or ordinals stop agreeing).
_FIELD_SELECTOR = (
    'input:not([type="hidden"]), textarea, select, [role="textbox"], '
    '[role="combobox"], [contenteditable="true"]'
)


_DEEP_WALK_JS = """
const __nmDeepEach = (root, sel, visit) => {
  const walk = (node) => {
    let matches = [];
    try { matches = node.querySelectorAll(sel); } catch (e) {}
    for (const el of matches) { try { visit(el); } catch (e) {} }
    let kids = [];
    try { kids = node.querySelectorAll('*'); } catch (e) {}
    for (const el of kids) {
      try { if (el.shadowRoot) walk(el.shadowRoot); } catch (e) {}
      const tag = (el.tagName || '').toUpperCase();
      if (tag === 'IFRAME' || tag === 'FRAME') {
        try { const doc = el.contentDocument; if (doc) walk(doc); }
        catch (e) { /* cross-origin: skip */ }
      }
    }
  };
  walk(root);
};
"""


_CANDIDATES_JS = """(name) => {
""" + _DEEP_WALK_JS + """
  const q = String(name || '').trim().toLowerCase();
  if (!q) return [];
  const qtokens = q.split(/\\s+/).filter(Boolean);
  const esc = (s) => (window.CSS && CSS.escape)
    ? CSS.escape(s) : String(s).replace(/"/g, '\\\\"');
  const norm = (s) => String(s || '').trim().toLowerCase().replace(/\\s+/g, ' ');
  const labelText = (el) => {
    try {
      const root = el.getRootNode && el.getRootNode();
      if (el.id && root && root.querySelector) {
        const lab = root.querySelector('label[for="' + esc(el.id) + '"]');
        if (lab) return norm(lab.innerText);
      }
    } catch (e) {}
    // walk up through labels, hopping out of shadow roots via .host
    let node = el;
    let hops = 0;
    while (node && hops < 8) {
      hops += 1;
      if ((node.tagName || '').toUpperCase() === 'LABEL')
        return norm(node.innerText);
      node = node.parentNode;
      if (node && node.nodeType === 11) node = node.host;
    }
    return '';
  };
  const out = [];
  const seen = new Set();
  let visitOrd = 0;
  __nmDeepEach(document, '""" + _FIELD_SELECTOR + """',
    (el) => {
      if (seen.has(el)) return;
      seen.add(el);
      const myOrd = visitOrd++;
      if (el.disabled) return;
      // hidden elements never accept typed input; keep the focused one.
      if (el.offsetParent === null && el !== document.activeElement) return;
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
      else if (attrs.testid && attrs.testid.indexOf(q) !== -1) { score = 15; by = 'data-testid~'; }
      // token overlap: every query word present somewhere, any order
      // ("email address" matches "Email Address").
      if (!score && qtokens.length > 1) {
        const hay = Object.values(attrs).join(' ');
        if (qtokens.every((t) => hay.indexOf(t) !== -1)) {
          score = 10; by = 'tokens';
        }
      }
      if (!score) return;
      let shadow = false, inFrame = false;
      try {
        const root = el.getRootNode && el.getRootNode();
        shadow = !!(root && root.nodeType === 11);
        let n = el, hops2 = 0;
        while (n && hops2 < 12) {
          hops2 += 1;
          const t = (n.tagName || '').toUpperCase();
          if (t === 'IFRAME' || t === 'FRAME') { inFrame = true; break; }
          n = n.parentNode;
          if (n && n.nodeType === 11) n = n.host;
        }
      } catch (e) {}
      out.push({
        tag: (el.tagName || '').toLowerCase(),
        type: (el.getAttribute('type') || '').toLowerCase(),
        by: by,
        score: score,
        name: el.getAttribute('name') || '',
        id: el.id || '',
        label: attrs.label.slice(0, 60),
        placeholder: attrs.placeholder.slice(0, 60),
        aria: attrs.aria.slice(0, 60),
        shadow: shadow,
        in_frame: inFrame,
        ord: myOrd,
      });
    });
  out.sort((a, b) => b.score - a.score);
  return out.slice(0, 25);
}"""

#: Pin the marker on the candidates script's winner by deep-DOM ordinal.
#: The ordinal is the element's visit index in the walker's traversal
#: order, so it stays exact even when ids repeat across shadow roots
#: (where an identity re-match would pin the wrong twin). Both scripts
#: walk the identical selector in the identical order — ordinals agree
#: by construction. Returns 'ok' / 'not-found'.
_PIN_ORD_JS = """(ord) => {
""" + _DEEP_WALK_JS + """
  const MARK = '""" + FIELD_MARKER + """';
  const SEL = '""" + _FIELD_SELECTOR + """';
  __nmDeepEach(document, '[' + MARK + ']', (m) => m.removeAttribute(MARK));
  const seen = new Set();
  let count = -1, pinned = false;
  __nmDeepEach(document, SEL, (el) => {
    if (seen.has(el)) return;
    seen.add(el);
    count += 1;
    if (pinned || count !== ord) return;
    try { el.setAttribute(MARK, '1'); pinned = true; } catch (e) {}
  });
  return pinned ? 'ok' : 'not-found';
}"""


_CLEAR_JS = """() => {
""" + _DEEP_WALK_JS + """
  const MARK = '""" + FIELD_MARKER + """';
  __nmDeepEach(document, '[' + MARK + ']', (m) => m.removeAttribute(MARK));
  return true;
}"""

_DESCRIBE_JS = """() => {
""" + _DEEP_WALK_JS + """
  const out = [];
  const seen = new Set();
  __nmDeepEach(document,
    'input:not([type="hidden"]), textarea, select, button, [role="button"]',
    (el) => {
      if (seen.has(el)) return;
      seen.add(el);
      let label = '';
      try {
        const root = el.getRootNode && el.getRootNode();
        if (el.id && root && root.querySelector) {
          const lab = root.querySelector(
            'label[for="' + el.id.replace(/"/g, '\\\\"') + '"]');
          if (lab) label = (lab.innerText || '').trim();
        }
      } catch (e) {}
      if (!label) {
        let node = el, hops = 0;
        while (node && hops < 8) {
          hops += 1;
          if ((node.tagName || '').toUpperCase() === 'LABEL') {
            label = (node.innerText || '').trim();
            break;
          }
          node = node.parentNode;
          if (node && node.nodeType === 11) node = node.host;
        }
      }
      let shadow = false;
      try {
        const root = el.getRootNode && el.getRootNode();
        shadow = !!(root && root.nodeType === 11);
      } catch (e) {}
      out.push({
        tag: (el.tagName || '').toLowerCase(),
        type: (el.getAttribute('type') || '').toLowerCase(),
        name: el.getAttribute('name') || '',
        id: el.id || '',
        label: label.slice(0, 60),
        placeholder: (el.getAttribute('placeholder') || '').slice(0, 60),
        aria: (el.getAttribute('aria-label') || '').slice(0, 60),
        shadow: shadow,
      });
      if (out.length >= 100) return;
    });
  return out.slice(0, 100);
}"""

#: Find the marker-pinned element across the deep DOM (shadow roots and
#: same-origin frames included) — a plain document.querySelector would
#: miss a field pinned inside a shadow tree.
_FIND_MARKED_JS = """
const __nmFindMarked = () => {
  let found = null;
  __nmDeepEach(document, '[data-nm-field="1"]',
    (el) => { if (!found) found = el; });
  return found;
};
"""

#: React/Vue-controlled-input-compatible value setter: goes through the
#: native property setter (so framework change detection fires) and then
#: dispatches input+change. Used as the fallback for date pickers and
#: other custom controls that swallow playwright's fill().
_SET_VALUE_JS = """(value) => {
""" + _DEEP_WALK_JS + _FIND_MARKED_JS + """
  const el = __nmFindMarked();
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

#: Group the page's controls by their owning <form>: fields plus the
#: submit control (input/button type=submit, or a typeless <button>).
#: Orphan controls outside any <form> go in a synthetic orphan group.
_DESCRIBE_FORMS_JS = """() => {
""" + _DEEP_WALK_JS + """
  const out = [];
  const forms = [];
  __nmDeepEach(document, 'form', (f) => forms.push(f));
  const ident = (el) => {
    let label = '';
    try {
      const root = el.getRootNode && el.getRootNode();
      if (el.id && root && root.querySelector) {
        const lab = root.querySelector(
          'label[for="' + el.id.replace(/"/g, '\\\\"') + '"]');
        if (lab) label = (lab.innerText || '').trim();
      }
    } catch (e) {}
    return {
      tag: (el.tagName || '').toLowerCase(),
      type: (el.getAttribute('type') || '').toLowerCase(),
      name: el.getAttribute('name') || '',
      id: el.id || '',
      label: label.slice(0, 60),
      placeholder: (el.getAttribute('placeholder') || '').slice(0, 60),
    };
  };
  const fieldsOf = (root) => {
    const list = [];
    __nmDeepEach(root,
      'input:not([type="hidden"]):not([type="submit"]), textarea, select',
      (el) => list.push(ident(el)));
    return list;
  };
  const submitOf = (root) => {
    let found = null;
    __nmDeepEach(root,
      'input[type="submit"], button[type="submit"], button:not([type])',
      (el) => { if (!found) found = el; });
    return found ? ident(found) : null;
  };
  forms.forEach((f, i) => {
    out.push({
      index: i,
      id: f.id || '',
      name: f.getAttribute('name') || '',
      action: f.getAttribute('action') || '',
      method: (f.getAttribute('method') || 'get').toLowerCase(),
      fields: fieldsOf(f),
      submit: submitOf(f),
    });
  });
  const orphans = [];
  __nmDeepEach(document,
    'input:not([type="hidden"]):not([type="submit"]), textarea, select',
    (el) => {
      let n = el, hops = 0, inForm = false;
      while (n && hops < 16) {
        hops += 1;
        if ((n.tagName || '').toUpperCase() === 'FORM') { inForm = true; break; }
        n = n.parentNode;
        if (n && n.nodeType === 11) n = n.host;
      }
      if (!inForm) orphans.push(ident(el));
    });
  if (orphans.length) {
    out.push({index: -1, id: '', name: '', action: '', method: '',
              fields: orphans, submit: null, orphan: true});
  }
  return out;
}"""


def _info_of(raw: Any) -> dict[str, Any]:
    """Normalize one candidate dict from the resolver JS."""
    return {
        "tag": str(raw.get("tag", "")),
        "type": str(raw.get("type", "")),
        "by": str(raw.get("by", "")),
        "score": raw.get("score", 0),
        "name": str(raw.get("name", "")),
        "id": str(raw.get("id", "")),
        "label": str(raw.get("label", "")),
        "placeholder": str(raw.get("placeholder", "")),
        "aria": str(raw.get("aria", "")),
        "shadow": bool(raw.get("shadow", False)),
        "in_frame": bool(raw.get("in_frame", False)),
        #: deep-DOM visit ordinal (-1 when the info came from the legacy
        #: single-winner protocol, which pins inline and needs no pin).
        "ord": raw.get("ord", -1),
    }


def _ranked_list(raw: Any) -> list[dict[str, Any]]:
    """Normalize the candidates script's payload.

    Accepts the ranked list — and the legacy single-winner dict the old
    resolver returned (duck-typed drivers in older tests speak that
    protocol), treating it as a one-entry ranking.
    """
    if isinstance(raw, dict):
        raw = [raw]
    return [_info_of(item) for item in (raw or [])
            if isinstance(item, dict)]


def resolve_candidates(page: Any, name: str,
                       limit: int = 5) -> list[dict[str, Any]]:
    """Ranked candidate controls for ``name``, best first (no pinning).

    The Stagehand-style diagnosis split: when ``resolve()`` fails, these
    candidates tell the caller *what the page actually has* so the error
    can suggest "did you mean …?" instead of a bare name. Each entry is
    ``{tag, type, by, score, name, id, label, placeholder, aria, shadow,
    in_frame}``. Returns [] when the driver cannot run JS.
    """
    name = (name or "").strip()
    if not name:
        return []
    evaluate = getattr(page, "evaluate", None)
    if evaluate is None:
        return []
    try:
        raw = evaluate(_CANDIDATES_JS, name)
    except Exception as exc:
        _log.debug("field candidates failed for %r: %s", name, exc)
        return []
    out = _ranked_list(raw)
    return out[: max(1, int(limit or 5))]


def resolve(page: Any, name: str) -> tuple[str, dict[str, Any] | None]:
    """Resolve ``name`` to ``(selector, info)`` on the page.

    ``info`` is the resolver's ``{tag, type, by, score, name, id, ...}``
    dict, or ``None`` when the driver cannot run JS (duck-typed pages) —
    the caller then uses the legacy name/id selector with untyped
    behavior. The winner is pinned with the marker attribute so the
    caller can drive it through a stable selector. Raises
    :class:`FieldNotFound` (carrying ``.candidates``) when nothing on
    the page matches.
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
        raw = evaluate(_CANDIDATES_JS, name)
    except Exception as exc:
        raise FieldNotFound(
            f"field resolver failed for {name!r}: {exc}") from exc
    ranked = _ranked_list(raw)
    if not ranked:
        candidates = resolve_candidates(page, name, limit=5)
        hint = ""
        if candidates:
            best = candidates[0]
            ident = (best.get("label") or best.get("aria")
                     or best.get("placeholder") or best.get("name")
                     or best.get("id") or "?")
            hint = f" — did you mean {ident!r}?"
        raise FieldNotFound(
            f"no form field matching {name!r} on the page{hint}",
            candidates=candidates)
    best = ranked[0]
    if best.get("ord", -1) >= 0:
        # Pin the winner by deep-DOM ordinal — exact even when ids
        # repeat across shadow roots. (Legacy single-winner infos carry
        # ord=-1: they pinned inline, so there is nothing to do.)
        try:
            pin_result = evaluate(_PIN_ORD_JS, best["ord"])
        except Exception as exc:
            raise FieldNotFound(
                f"field resolver failed for {name!r}: {exc}") from exc
        if pin_result == "not-found":
            raise FieldNotFound(
                f"field {name!r} vanished before it could be pinned — "
                "the page changed under the resolver")
    return MARKER_SELECTOR, best


def clear_marker(page: Any) -> None:
    """Remove the ``data-nm-field`` marker (best-effort, never raises)."""
    evaluate = getattr(page, "evaluate", None)
    if evaluate is None:
        return
    try:
        evaluate(_CLEAR_JS)
    except Exception as exc:
        _log.debug("clear_nm_field_marker failed (best-effort): %s", exc)


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


def describe_forms(page: Any) -> list[dict[str, Any]]:
    """The page's forms: fields grouped by owning <form> plus each
    form's submit control.

    Each entry is ``{index, id, name, action, method, fields, submit}``;
    controls outside any <form> land in an ``orphan: True`` group. Used
    to pick the right form before a multi-field fill, and to find the
    submit target without guessing. Returns [] when the driver cannot
    run JS.
    """
    evaluate = getattr(page, "evaluate", None)
    if evaluate is None:
        return []
    try:
        raw = evaluate(_DESCRIBE_FORMS_JS)
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


#: Read back the marker-pinned field's live state: typed value, checked
#: state, and attached files (for file inputs). The verification
#: counterpart to _SET_VALUE_JS — fill success is confirmed by reading
#: the DOM back, never assumed from a no-error return.
_READ_VALUE_JS = """() => {
""" + _DEEP_WALK_JS + _FIND_MARKED_JS + """
  const el = __nmFindMarked();
  if (!el) return null;
  const isFile = (el.getAttribute('type') || '').toLowerCase() === 'file';
  return {
    value: ('value' in el) ? el.value : null,
    checked: ('checked' in el) ? !!el.checked : null,
    files: (isFile && el.files) ? el.files.length : null,
    fileName: (isFile && el.files && el.files[0]) ? el.files[0].name : null,
  };
}"""


def read_value_js(page: Any) -> dict[str, Any] | None:
    """Read back the resolved (marker-pinned) field's live state.

    Returns ``{"value", "checked", "files", "fileName"}`` (fields that do
    not apply are None), or None when no field is currently pinned.
    Raises the page's own error when evaluation fails.
    """
    result = page.evaluate(_READ_VALUE_JS)
    if result is None:
        return None
    return dict(result)


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
