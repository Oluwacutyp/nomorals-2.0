"""Code-card tool: the Carbon surface for the brain.

"Screenshot this code" over the Studio — not a standalone tool.
The render action builds an :class:`ImageDocument` through
``Studio.code_card_document`` (the organ renders the code to a
PNG from Pygments' own token stream, deterministic — same card,
same render hash), saves the document beside the delivered PNG
(<same stem>.json) so Studio work can continue from the card
(more ops, undo, branch, re-render), and renders it through
``Studio.render_document``.

``action=render`` takes the code itself (``code=``) or a file
holding it (``file=``, sandbox-checked), plus ``language=`` (a
Pygments lexer name or alias — "python", "py", "js", ...),
``theme=`` (one of the ``themes`` action's names), ``chrome=``
(window chrome with traffic-light dots), ``watermark=`` (the
"Powered by peacethefirst" stamp), ``title=`` (shown in the
title bar) and ``font_size=``. ``action=themes`` lists the
themes; ``action=status`` reports what this machine can
actually render (Pygments, the stocked mono face, the themes).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..core.policy import Capability

__all__ = ["register"]


def _out_dir(context: Any) -> Path:
    base = Path(getattr(context, "workspace", None) or ".")
    d = base / "generated" / "codecards"
    d.mkdir(parents=True, exist_ok=True)
    return d


def register(registry: Any) -> None:
    """Attach the ``code_card`` tool."""
    context = registry.context

    @registry.register(
        "code_card",
        description=(
            "Render code as a presentation image (a Carbon-style "
            "code card): real syntax highlighting, a window chrome "
            "with traffic-light dots, themes, and the 'Powered by "
            "peacethefirst' stamp — as a Studio image document the "
            "owner can keep editing. Call this when the owner says "
            "'screenshot this code', 'make an image of this code' "
            "or 'turn this code into a picture', and offer it after "
            "a code fix lands ('want this as an image?'). "
            "action=render takes code=<the code text> or file=<a "
            "path holding the code> and writes the PNG (out=<optional "
            "PNG path>; default is the workspace generated/codecards "
            "dir) plus the document beside it (<same stem>.json). "
            "language= is a Pygments lexer name or alias "
            "('python', 'py', 'js', ...); theme= is one of the "
            "themes action's names (default 'dracula' under the "
            "carbon preset); preset= is one of the presets "
            "action's names — 'carbon' honours every knob as "
            "passed, a 'ray*' preset fixes the look (its theme, "
            "gradient backdrop, chrome, watermark, line numbers "
            "and font size) and names any knob it overrode; "
            "chrome= and watermark= are true/false; title= shows "
            "in the title bar. action=themes lists the themes; "
            "action=presets lists the preset pack; action=status "
            "reports what this machine renders. Returns ok, paths, "
            "document, theme, language, lexer, font, render_hash, "
            "preset, notes, or error."
        ),
        capability=Capability.FS_WRITE,
    )
    def code_card_tool(action: str = "render", code: str = "",
                       file: str = "", language: str = "",
                       theme: str = "", preset: str = "carbon",
                       chrome: bool | None = None,
                       watermark: bool | None = None, title: str = "",
                       font_size: int | None = None,
                       line_numbers: bool | None = None,
                       out: str = "") -> dict[str, Any]:
        """A code card through the Studio document flow."""
        from ..media.imggen.codecard import (
            CodeCardError,
            list_presets,
            list_themes,
        )
        from ..media.imggen.studio import Studio

        act = (action or "render").strip().lower()
        if act == "themes":
            return {"ok": True, "themes": list_themes(),
                    "default": "dracula"}
        if act == "presets":
            return {"ok": True, "presets": list_presets(),
                    "default": "carbon"}
        if act == "status":
            from ..media.imggen.codecard import PRESETS, THEMES

            try:
                from ..media.typography import (
                    get_service,
                    pil_font_for,
                )

                font = pil_font_for(get_service(), "mono", 15)
                font_name = str(font.getname()[0])
                font_note = ""
            except Exception as exc:  # noqa: BLE001 - named
                font_name = ""
                font_note = str(exc)
            return {"ok": True, "pygments": True,
                    "themes": sorted(THEMES),
                    "default_theme": "dracula",
                    "presets": sorted(PRESETS),
                    "default_preset": "carbon",
                    "font": font_name,
                    "font_note": font_note}
        if act != "render":
            return {"ok": False,
                    "error": f"unknown code_card action {act!r}; "
                             "use render|themes|presets|status"}
        text = (code or "").strip("\n")
        if file:
            from .filesystem import safe_path

            try:
                src = Path(safe_path(context, file))
            except Exception as exc:  # noqa: BLE001 - named refusal
                return {"ok": False,
                        "error": f"cannot use code file {file}: "
                                 f"{exc}"}
            if not src.exists():
                return {"ok": False,
                        "error": f"no such code file: {file}"}
            try:
                text = src.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                return {"ok": False,
                        "error": f"cannot read code file {file}: "
                                 f"{exc}"}
        if not (text or "").strip():
            return {"ok": False,
                    "error": "code_card render needs code= text "
                             "or file= a path holding the code"}
        studio = Studio()
        out_dir = _out_dir(context)
        save_to = str(out) if out else str(out_dir)
        try:
            result = studio.code_card(
                text, language=language,
                theme=(theme or None), preset=preset,
                chrome=chrome, watermark=watermark, title=title,
                font_size=font_size, line_numbers=line_numbers,
                save_to=save_to)
        except CodeCardError as exc:
            return {"ok": False, "error": str(exc)}
        result["ok"] = True
        return result
