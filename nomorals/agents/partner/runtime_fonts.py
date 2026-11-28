"""RuntimeFontsMixin: the /fonts command in the live event
path — browse and search Devon's stocked FontLibrary.

The library itself is the registry in
``media/typography/fontlib.py`` over the stocked tiers (the
pack in her data estate, the vendored core subset, owner-local
``NM_FONT_DIRS``).  This mixin only presents it: an overview
card, family search, category browsing and one family's full
record — every answer read from the real manifests, so what
/fonts says is stocked is what the renderers can resolve.

Gate: /fonts is owner-only (deliberately NOT in
PUBLIC_COMMANDS, like /mock) — the library is her estate's
inventory, and the handler repeats the operator check in the
sibling idiom.
"""

from __future__ import annotations

from .failures import record_failure

_FONTS_USAGE = (
    "usage: /fonts — the stocked library at a glance · "
    "/fonts search <words> — find families · "
    "/fonts category <name> — one shelf · "
    "/fonts info <family> — one family's record")
_LIST_CAP = 24


class RuntimeFontsMixin:
    """Font-library commands for PartnerRuntime."""

    def _fonts_library(self):  # type: ignore[no-untyped-def]
        from ...media.typography.fontlib import FontLibrary

        return FontLibrary.discover()

    def _render_fonts_card(self, card,  # type: ignore[no-untyped-def]
                           message=None) -> str:
        """Render a /fonts card for the message's platform.

        The handler has a live platform handle, so it renders
        through the layer per platform (render_card) instead of
        the canonical card_text() — the canonical dialect is for
        surfaces without a platform handle, and it reaches the
        user raw when a downstream conversion is skipped (owner
        screenshot 2026-11-15: literal ** and backticks on
        WhatsApp). No message, no platform: the canonical
        dialect is the fallback, exactly as before.
        """
        platform = ""
        chat = getattr(message, "chat", None)
        if chat is not None:
            platform = getattr(chat, "platform", "") or ""
        if platform:
            from ...social.chat.typography import render_card

            return render_card(card, platform=platform).text
        from ...social.chat.typography import card_text

        return card_text(card)

    def _control_fonts(self, kind: str, tail: str,
                       message=None) -> str:  # type: ignore[no-untyped-def]
        """One /fonts command: present the stocked library."""
        if message is not None and not self._is_operator(message):
            return "that one's just for the owner."
        from ...social.chat.typography import Card

        try:
            library = self._fonts_library()
            stats = library.stats()
        except Exception as exc:  # noqa: BLE001 - honest line
            record_failure("fonts.library", exc)
            return f"the font library could not be read: {exc}"
        tail = str(tail or "").strip()
        words = tail.split(None, 1)
        verb = words[0].lower() if words else ""
        rest = words[1].strip() if len(words) > 1 else ""

        if verb in ("", "status", "library"):
            card = Card("chat", title="Font library", icon="🔤")
            card.line("her own typefaces — stocked, offline, "
                      "hers. Every renderer resolves from here.")
            rows = []
            for tier in stats["tiers"]:
                label = {"pack": "full pack",
                         "core": "core subset",
                         "local": "owner-local"}.get(
                             tier["tier"], tier["tier"])
                if tier["stocked"]:
                    # The counts only: the source commit is an
                    # internal id — it lives in the library stats,
                    # never in user-facing text (owner rule
                    # 2026-11-03: internal ids never print).
                    value = (f"{tier['families']} families · "
                             f"{tier['files']} files")
                else:
                    value = "not stocked"
                rows.append((label, value))
            rows.append(("total", f"{stats['families']} families · "
                                  f"{stats['faces']} faces"))
            card.stats(rows, title="stocked tiers")
            if not any(tier["tier"] == "pack" and tier["stocked"]
                       for tier in stats["tiers"]):
                # The 2026-11-15 complaint was this card saying
                # "full pack: not stocked" and nothing else — no
                # progress, no reason, no way forward.  The
                # stocking facts are read from disk (never the
                # network); when they cannot be read the card
                # simply carries no progress line.
                try:
                    from ...media.typography.fontpack import stock_status
                    card.line(stock_status()["summary"])
                except Exception:  # noqa: BLE001 - note is best-effort
                    pass
            cats = stats["categories"]
            if cats:
                card.stats(
                    [(name, count) for name, count in
                     list(cats.items())[:8]],
                    title="shelves")
            card.note(_FONTS_USAGE)
            return self._render_fonts_card(card, message)

        if verb == "categories":
            card = Card("chat", title="Font shelves", icon="🔤")
            card.stats(
                [(name, count) for name, count in
                 stats["categories"].items()],
                title="families per shelf")
            card.note("/fonts category <name> opens one shelf")
            return self._render_fonts_card(card, message)

        if verb == "search":
            if not rest:
                return _FONTS_USAGE
            hits = library.search(rest)
            card = Card("chat", title=f"Fonts · {rest}", icon="🔎")
            if not hits:
                card.line(f"no stocked family matches {rest!r} — "
                          "the library answers for what it holds, "
                          "nothing invented.")
            else:
                card.block_lines(
                    [self._fonts_row(fam) for fam in
                     hits[:_LIST_CAP]],
                    title=f"{len(hits)} families")
                if len(hits) > _LIST_CAP:
                    card.note(f"first {_LIST_CAP} shown — "
                              "narrow the search to see the rest")
            return self._render_fonts_card(card, message)

        if verb == "category":
            if not rest:
                return _FONTS_USAGE
            want = rest.casefold()
            hits = [fam for fam in library.families()
                    if fam.category.casefold() == want]
            card = Card("chat", title=f"Fonts · {rest}", icon="🔤")
            if not hits:
                card.line(f"no shelf named {rest!r} — /fonts "
                          "categories lists the shelves she has.")
            else:
                card.block_lines(
                    [self._fonts_row(fam) for fam in
                     hits[:_LIST_CAP]],
                    title=f"{len(hits)} families")
                if len(hits) > _LIST_CAP:
                    card.note(f"first {_LIST_CAP} shown")
            return self._render_fonts_card(card, message)

        if verb == "info":
            if not rest:
                return _FONTS_USAGE
            fam = library.family(rest)
            card = Card("chat", title=f"Font · {rest}", icon="🔤")
            if fam is None:
                card.line(f"{rest!r} is not in the stocked "
                          "library — /fonts search finds what "
                          "she holds.")
                return self._render_fonts_card(card, message)
            rows = [("family", fam.name),
                    ("shelf", fam.category or "—"),
                    ("licence", fam.license or "—"),
                    ("tier", fam.tier),
                    ("faces", len(fam.faces))]
            if fam.license_file:
                rows.append(("licence file", fam.license_file))
            if fam.reserved_font_name:
                rows.append(("reserved name", "yes (OFL)"))
            card.stats(rows)
            face_lines = []
            for face in fam.faces[:12]:
                shape = "variable" if face.variable else \
                    str(face.weight)
                if face.weight_range:
                    shape = (f"variable "
                             f"{face.weight_range[0]}–"
                             f"{face.weight_range[1]}")
                face_lines.append(
                    f"{face.file.rsplit('/', 1)[-1]} — {shape}"
                    f"{' italic' if face.italic else ''}")
            if face_lines:
                card.block_lines(face_lines, title="faces")
            try:
                path = library.resolve(fam.name)
                card.note(f"resolves to {path}")
            except Exception as exc:  # noqa: BLE001 - named
                card.note(str(exc))
            return self._render_fonts_card(card, message)

        if verb == "prune":
            # Trim the stocked pack to the families renderers
            # actually ask for (gapfix 164): `/fonts prune`
            # lists exactly what would go (dry run);
            # `/fonts prune go` deletes it. Only unreferenced
            # pack families ever go — never the core subset,
            # never an owner-local dir, never a system font.
            from ...core import diskspace
            from ...media.typography import prune_library

            go = rest.strip().lower() in ("go", "yes", "do it")
            try:
                report = prune_library(dry_run=not go)
            except Exception as exc:  # noqa: BLE001 - honest line
                record_failure("fonts.prune", exc)
                return f"the font prune could not run: {exc}"
            if report.get("error") and not report.get(
                    "removed_families"):
                return ("the font prune could not run: "
                        f"{report['error']}")
            removed = report.get("removed_families") or []
            kept = report.get("kept_families", 0)
            card = Card("chat", title="Font prune", icon="🔤")
            if not removed:
                card.line("nothing to prune — every stocked "
                          "pack family is one a renderer asks "
                          "for.")
            else:
                names = ", ".join(removed)
                if go:
                    card.line(
                        f"pruned {len(removed)} unreferenced "
                        f"families ({names}) — "
                        f"{diskspace.human_bytes(report.get('bytes_freed', 0))} freed.")
                    if report.get("build_dir_removed"):
                        card.line(
                            "the leftover build copy went too "
                            f"({diskspace.human_bytes(report.get('build_bytes_freed', 0))}).")
                    if not report.get("verified", True):
                        card.line("the pack did not re-verify "
                                  "cleanly after the prune — "
                                  f"{report.get('error', '')}")
                    else:
                        card.line("the pack re-verified clean.")
                else:
                    card.line(
                        f"would prune {len(removed)} unreferenced "
                        f"families ({names}) — "
                        f"{diskspace.human_bytes(report.get('prunable_bytes', 0))} "
                        "would come back. Nothing deleted.")
                    card.note("`/fonts prune go` deletes them. "
                              "A pruned family comes back only by "
                              "rebuilding the pack.")
            kept_names = [name for name in
                          report.get("referenced_families") or []]
            shown = ", ".join(kept_names[:12])
            if len(kept_names) > 12:
                shown += f" and {len(kept_names) - 12} more"
            card.line(f"kept {kept} families ({shown}).")
            return self._render_fonts_card(card, message)

        return _FONTS_USAGE

    @staticmethod
    def _fonts_row(fam) -> str:  # type: ignore[no-untyped-def]
        bits = [fam.name]
        if fam.category:
            bits.append(fam.category)
        bits.append(f"{len(fam.faces)} faces")
        bits.append(fam.tier)
        return " · ".join(bits)
