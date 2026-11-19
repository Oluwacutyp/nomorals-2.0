"""DesignCraft port — layout organ tests (lane 104, census §3A.11).

Failing-first: the layout organ does not exist yet on the base, so
every import here fails until the port lands. The test pins the
public contract the port must meet:

* ``nomorals.media.layout`` — a real page/layout document model
  (spreads, pages, margins, columns, guides, layers, swatches,
  styles, stories, items) in the upstream `.designcraft` zip shape
  (document.json + assets/ + meta.json, versioned), with
  fidelity-preserving edits.
* ``flex`` — a faithful port of upstream ``crates/doc/src/flex.rs``
  ``arrange``/``layout``/``layout_all`` (row/column, wrap, justify,
  align, reverse, auto sizing, padding, gaps).
* ``compose`` — real text composition measured with the PIL font
  door: greedy wrap, columns, threading across frames, overset
  detection, vertical justification.
* ``render`` — a real page renderer to PNG through PIL/numpy.
* ``parity`` / ``attribution`` — machine-checked TOML runners;
  a missing fixture FAILS its case, never a silent skip.
* ``designcraft`` brain tool — the same registry contract the
  vectorcraft/gridcraft tools pin.
"""

from __future__ import annotations

import io
import json
import tempfile
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from nomorals.media.layout import (
    Columns,
    FlexAlign,
    FlexDirection,
    FlexJustify,
    FlexLayout,
    FlexSize,
    LayoutDocument,
    LayoutEditor,
    LayoutError,
    Margins,
    arrange,
    layout_all,
    layout_item,
    render_document,
    render_page,
    run_parity,
    verify_attribution,
)
from nomorals.media.layout.codec import parse_document, to_document_bytes
from nomorals.media.layout.compose import compose_story, measure_text
from nomorals.media.layout.model import (
    CharRun,
    Fill,
    ParaFormat,
    Story,
)
from nomorals.tools.registry import ToolRegistry

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "layout"


# ── document model ────────────────────────────────────────────────

class TestBlankDocument:
    def test_blank_has_one_spread_one_page(self) -> None:
        doc = LayoutDocument.blank()
        assert len(doc.spreads) == 1
        assert len(doc.spreads[0].pages) == 1
        page = doc.spreads[0].pages[0]
        assert (page.width, page.height) == (612.0, 792.0)
        assert page.margins == Margins(36.0, 36.0, 36.0, 36.0)
        assert len(doc.layers) == 1
        assert doc.layers[0].name == "Layer 1"

    def test_blank_round_trips_byte_identical(self) -> None:
        doc = LayoutDocument.blank()
        data = to_document_bytes(doc)
        again = parse_document(data)
        assert to_document_bytes(again) == data

    def test_title_survives(self) -> None:
        doc = LayoutDocument.blank(title="Quarterly")
        assert parse_document(to_document_bytes(doc)).title == "Quarterly"


class TestCodec:
    def test_fixture_parses(self) -> None:
        doc = LayoutDocument.from_bytes(
            (FIXTURES / "simple.designcraft").read_bytes())
        assert doc.title == "Simple Fixture"
        assert doc.page_count() == 1
        assert doc.item_count() == 3

    def test_magazine_spread(self) -> None:
        doc = LayoutDocument.from_bytes(
            (FIXTURES / "magazine.designcraft").read_bytes())
        assert doc.page_count() == 2
        assert len(doc.spreads) == 1
        assert len(doc.spreads[0].pages) == 2
        # The fixture carries three stories, not one: S1 body (threaded
        # M2→M3→M6), S2 pull quote, S3 footer.
        assert doc.story_count() == 3

    def test_broken_zip_refused(self) -> None:
        with pytest.raises(LayoutError) as exc:
            parse_document(b"not a zip at all")
        assert exc.value.kind == "parse"

    def test_corrupt_json_refused(self) -> None:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("document.json", "{not json")
            zf.writestr("meta.json", json.dumps({"format_version": 1}))
        with pytest.raises(LayoutError) as exc:
            parse_document(buf.getvalue())
        assert exc.value.kind == "parse"

    def test_unknown_format_version_refused(self) -> None:
        doc = LayoutDocument.blank()
        data = json.loads(
            zipfile.ZipFile(io.BytesIO(to_document_bytes(doc)))
            .read("document.json"))
        data["format_version"] = 99
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("document.json", json.dumps(data))
            zf.writestr("meta.json", json.dumps({"format_version": 99}))
        with pytest.raises(LayoutError) as exc:
            parse_document(buf.getvalue())
        assert exc.value.kind == "unsupported"

    def test_oversize_package_refused(self) -> None:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("document.json",
                        json.dumps({"format_version": 1, "title": "x"}))
            zf.writestr("assets/big.bin", b"x" * (33 * 1024 * 1024))
        with pytest.raises(LayoutError) as exc:
            parse_document(buf.getvalue())
        assert exc.value.kind == "parse"


# ── fidelity ──────────────────────────────────────────────────────

class TestFidelity:
    def test_no_edit_round_trip_is_byte_identical(self) -> None:
        raw = (FIXTURES / "magazine.designcraft").read_bytes()
        doc = LayoutDocument.from_bytes(raw)
        assert doc.to_bytes() == raw

    def test_edit_changes_only_the_touched_item(self) -> None:
        raw = (FIXTURES / "simple.designcraft").read_bytes()
        doc = LayoutDocument.from_bytes(raw)
        editor = LayoutEditor(doc)
        target = doc.all_items()[0]
        editor.restyle(target.id, fill=Fill(swatch="Red"))
        report = doc.fidelity_report(raw)
        assert report["untouched_changed"] == 0
        assert report["changed_ids"] == [target.id]
        assert report["added"] == 0
        assert report["removed"] == 0

    def test_add_changes_only_the_new_item(self) -> None:
        raw = (FIXTURES / "simple.designcraft").read_bytes()
        doc = LayoutDocument.from_bytes(raw)
        editor = LayoutEditor(doc)
        editor.add_rect(doc.spreads[0].pages[0].id, 10, 10, 50, 40,
                        fill=Fill(swatch="Blue"))
        report = doc.fidelity_report(raw)
        assert report["untouched_changed"] == 0
        assert report["added"] == 1


# ── page columns ──────────────────────────────────────────────────

class TestColumns:
    def test_even_edges(self) -> None:
        cols = Columns(count=2, gutter=12.0)
        assert cols.edges(540.0) == [0.0, 264.0, 276.0, 540.0]

    def test_three_columns(self) -> None:
        cols = Columns(count=3, gutter=12.0)
        edges = cols.edges(540.0)
        assert edges == [0.0, 172.0, 184.0, 356.0, 368.0, 540.0]

    def test_custom_positions_used_when_valid(self) -> None:
        cols = Columns(count=2, gutter=12.0,
                       positions=[0.0, 200.0, 212.0, 540.0])
        assert cols.edges(540.0) == [0.0, 200.0, 212.0, 540.0]

    def test_invalid_positions_fall_back_to_even(self) -> None:
        cols = Columns(count=2, gutter=12.0,
                       positions=[0.0, 100.0, 90.0, 540.0])
        assert cols.edges(540.0) == [0.0, 264.0, 276.0, 540.0]

    def test_mirrored_for_facing_page(self) -> None:
        cols = Columns(count=2, gutter=12.0,
                       positions=[0.0, 200.0, 212.0, 540.0])
        mirrored = cols.mirrored(540.0)
        assert mirrored.edges(540.0) == [0.0, 328.0, 340.0, 540.0]

    def test_gutter_move(self) -> None:
        cols = Columns(count=2, gutter=12.0)
        moved = cols.with_gutter_moved(540.0, 0, 100.0)
        assert moved == [0.0, 100.0, 112.0, 540.0]


# ── flex layout engine ────────────────────────────────────────────

def _row_bounds(widths):
    return [(0.0, 0.0, w, 50.0) for w in widths]


class TestFlexArrange:
    def test_row_flex_start(self) -> None:
        flex = FlexLayout(direction=FlexDirection.Row, gap=10.0)
        out = arrange(flex, (0.0, 0.0, 400.0, 200.0),
                      _row_bounds([100.0, 100.0, 100.0]))
        assert out.moves == [(0.0, 0.0), (110.0, 0.0), (220.0, 0.0)]

    def test_row_space_between(self) -> None:
        flex = FlexLayout(direction=FlexDirection.Row, gap=10.0,
                          justify=FlexJustify.SpaceBetween)
        out = arrange(flex, (0.0, 0.0, 400.0, 200.0),
                      _row_bounds([100.0, 100.0, 100.0]))
        assert out.moves == [(0.0, 0.0), (150.0, 0.0), (300.0, 0.0)]

    def test_row_center(self) -> None:
        flex = FlexLayout(direction=FlexDirection.Row, gap=10.0,
                          justify=FlexJustify.Center)
        out = arrange(flex, (0.0, 0.0, 400.0, 200.0),
                      _row_bounds([100.0, 100.0]))
        assert out.moves == [(95.0, 0.0), (205.0, 0.0)]

    def test_row_wrap_breaks_lines(self) -> None:
        flex = FlexLayout(direction=FlexDirection.Row, gap=10.0, wrap=True)
        out = arrange(flex, (0.0, 0.0, 250.0, 200.0),
                      _row_bounds([100.0, 100.0, 100.0]))
        assert out.moves[0] == (0.0, 0.0)
        assert out.moves[1] == (110.0, 0.0)
        assert out.moves[2] == (0.0, 60.0)

    def test_row_reverse_runs_from_the_end(self) -> None:
        flex = FlexLayout(direction=FlexDirection.Row, gap=10.0,
                          reverse=True)
        out = arrange(flex, (0.0, 0.0, 400.0, 200.0),
                      _row_bounds([100.0, 100.0]))
        # Upstream mirrors each item's main position:
        # main = main_len - at - m → 400-0-100, 400-110-100.
        assert out.moves == [(300.0, 0.0), (190.0, 0.0)]

    def test_column_direction(self) -> None:
        flex = FlexLayout(direction=FlexDirection.Column, gap=10.0)
        bounds = [(0.0, 0.0, 50.0, h) for h in (100.0, 100.0)]
        out = arrange(flex, (0.0, 0.0, 200.0, 400.0), bounds)
        assert out.moves == [(0.0, 0.0), (0.0, 110.0)]

    def test_auto_width_fits_content(self) -> None:
        flex = FlexLayout(direction=FlexDirection.Row, gap=10.0,
                          width=FlexSize.Auto)
        out = arrange(flex, (0.0, 0.0, 400.0, 200.0),
                      _row_bounds([100.0, 100.0]))
        assert out.rect == (0.0, 0.0, 210.0, 200.0)

    def test_padding_insets(self) -> None:
        flex = FlexLayout(direction=FlexDirection.Row, gap=10.0,
                          padding=(5.0, 10.0, 5.0, 10.0))
        out = arrange(flex, (0.0, 0.0, 400.0, 200.0),
                      _row_bounds([100.0]))
        assert out.moves == [(10.0, 5.0)]

    def test_align_center_cross_axis(self) -> None:
        flex = FlexLayout(direction=FlexDirection.Row, gap=10.0,
                          align=FlexAlign.Center)
        bounds = [(0.0, 0.0, 100.0, 50.0), (0.0, 0.0, 100.0, 100.0)]
        out = arrange(flex, (0.0, 0.0, 400.0, 200.0), bounds)
        # The line (100 tall) is centered in the 200-tall container
        # (offset 50); each item is centered in its line, so the
        # 50-tall item sits at 50 + 25 and the 100-tall at 50 + 0.
        assert out.moves == [(0.0, 75.0), (110.0, 50.0)]


class TestFlexOnItems:
    def test_layout_item_moves_children(self) -> None:
        doc = LayoutDocument.from_bytes(
            (FIXTURES / "flex.designcraft").read_bytes())
        container = next(i for i in doc.all_items()
                         if i.name == "flex-row")
        assert layout_item(container) is True
        kids = container.content_items()
        assert kids[0].xf.e == pytest.approx(0.0)
        assert kids[1].xf.e == pytest.approx(110.0)
        assert kids[2].xf.e == pytest.approx(220.0)

    def test_layout_all_is_idempotent(self) -> None:
        doc = LayoutDocument.from_bytes(
            (FIXTURES / "flex.designcraft").read_bytes())
        items = doc.all_items()
        layout_all(items)
        first = [(i.id, i.xf.e, i.xf.f) for i in doc.all_items()]
        layout_all(items)
        second = [(i.id, i.xf.e, i.xf.f) for i in doc.all_items()]
        assert first == second

    def test_non_flex_item_untouched(self) -> None:
        doc = LayoutDocument.blank()
        item = LayoutEditor(doc).add_rect(
            doc.spreads[0].pages[0].id, 5, 5, 20, 20)
        assert layout_item(item) is False
        assert (item.xf.e, item.xf.f) == (5.0, 5.0)


# ── text composition ──────────────────────────────────────────────

class TestCompose:
    def test_measure_is_real_and_monotonic(self) -> None:
        narrow = measure_text("hello", 12.0)
        wide = measure_text("hello world, this is longer", 12.0)
        assert 0.0 < narrow < wide
        assert measure_text("hello", 24.0) > measure_text("hello", 12.0)

    def test_greedy_wrap_fits_the_frame(self) -> None:
        story = Story(id="s1", paras=[ParaFormat(runs=[
            CharRun("The quick brown fox jumps over the lazy dog "
                    "and keeps running down the road")])])
        composed = compose_story(story, 200.0, 400.0, font_size=12.0)
        assert len(composed.lines) > 1
        for line in composed.lines:
            assert measure_text(line.text, 12.0) <= 200.0 + 1.0
        assert composed.overset is False

    def test_overset_detected(self) -> None:
        story = Story(id="s1", paras=[ParaFormat(runs=[
            CharRun("word " * 200)])])
        composed = compose_story(story, 200.0, 60.0, font_size=12.0)
        assert composed.overset is True
        assert composed.remaining

    def test_threading_flows_across_frames(self) -> None:
        # The 19-word list wraps to exactly 4 lines at 12pt in a
        # 200-wide frame, and a 200x60 frame holds exactly 4 lines
        # — so one copy fits in the first frame and proves nothing
        # about threading. Two copies (8 lines) genuinely flow
        # across both frames.
        story = Story(id="s1", paras=[ParaFormat(runs=[
            CharRun("alpha beta gamma delta epsilon zeta eta theta "
                    "iota kappa lambda mu nu xi omicron pi rho sigma "
                    "alpha beta gamma delta epsilon zeta eta theta "
                    "iota kappa lambda mu nu xi omicron pi rho sigma")])])
        frames = [(200.0, 60.0), (200.0, 60.0)]
        composed = compose_story(story, frames, font_size=12.0)
        assert len(composed.frames) == 2
        assert composed.frames[0].lines
        assert composed.frames[1].lines
        assert composed.overset is False
        joined = " ".join(
            line.text for frame in composed.frames for line in frame.lines)
        for word in ("alpha", "sigma"):
            assert word in joined

    def test_vertical_justification_centers(self) -> None:
        story = Story(id="s1", paras=[ParaFormat(runs=[
            CharRun("one line")])])
        top = compose_story(story, 200.0, 400.0, font_size=12.0,
                            vertical_justification="top")
        center = compose_story(story, 200.0, 400.0, font_size=12.0,
                               vertical_justification="center")
        assert center.lines[0].y > top.lines[0].y


# ── editor ────────────────────────────────────────────────────────

class TestEditor:
    def _doc(self):
        return LayoutDocument.from_bytes(
            (FIXTURES / "simple.designcraft").read_bytes())

    def test_add_rect(self) -> None:
        doc = self._doc()
        editor = LayoutEditor(doc)
        item = editor.add_rect(doc.spreads[0].pages[0].id,
                               10, 20, 50, 40, fill=Fill(swatch="Red"))
        assert doc.item_count() == 4
        assert (item.xf.e, item.xf.f) == (10.0, 20.0)

    def test_add_text_sets_story(self) -> None:
        doc = self._doc()
        editor = LayoutEditor(doc)
        item = editor.add_text(doc.spreads[0].pages[0].id, 10, 10,
                               200, 100, "Hello layout")
        assert "Hello layout" in doc.story_text(item.story_id)
        assert doc.item_count() == 4

    def test_move_and_resize(self) -> None:
        doc = self._doc()
        editor = LayoutEditor(doc)
        item = doc.all_items()[0]
        editor.move(item.id, 30, 40)
        assert (item.xf.e, item.xf.f) == (30.0, 40.0)
        editor.resize(item.id, 120, 80)
        assert item.bounds() == pytest.approx((30.0, 40.0, 150.0, 120.0))

    def test_group_and_ungroup(self) -> None:
        doc = self._doc()
        editor = LayoutEditor(doc)
        ids = [i.id for i in doc.all_items()[:2]]
        group = editor.group(ids)
        assert group.shape == "group"
        assert len(group.content_items()) == 2
        kids = editor.ungroup(group.id)
        assert len(kids) == 2

    def test_align_left(self) -> None:
        doc = self._doc()
        editor = LayoutEditor(doc)
        items = doc.all_items()[:2]
        editor.align([i.id for i in items], "left")
        assert items[0].xf.e == pytest.approx(items[1].xf.e)

    def test_distribute_horizontal(self) -> None:
        doc = self._doc()
        editor = LayoutEditor(doc)
        items = doc.all_items()[:3]
        editor.distribute([i.id for i in items], "horizontal")
        xs = sorted(i.xf.e for i in items)
        assert xs[1] - xs[0] == pytest.approx(xs[2] - xs[1])

    def test_snap_to_guide(self) -> None:
        doc = self._doc()
        editor = LayoutEditor(doc)
        page = doc.spreads[0].pages[0]
        editor.add_guide(page.id, "vertical", 100.0)
        item = doc.all_items()[0]
        editor.move(item.id, 103, item.xf.f)
        snapped = editor.snap(item.id, tolerance=5.0)
        assert snapped is True
        assert item.xf.e == pytest.approx(100.0)

    def test_liquid_pins_survive_page_resize(self) -> None:
        doc = self._doc()
        editor = LayoutEditor(doc)
        page = doc.spreads[0].pages[0]
        item = doc.all_items()[0]
        # Fixture rect sits at (36, 36, 200, 100) on a 612x792 page:
        # 376pt from the right edge, 656pt from the bottom edge.
        editor.set_liquid(item.id, pin_right=True, pin_bottom=True)
        editor.resize_page(page.id, 700.0, 900.0)
        # Pinned to the right/bottom edge: the item keeps its
        # distance from the edge it is pinned to.
        assert item.xf.e == pytest.approx(700.0 - 376.0 - 200.0)
        assert item.xf.f == pytest.approx(900.0 - 656.0 - 100.0)

    def test_reorder_layer(self) -> None:
        doc = self._doc()
        editor = LayoutEditor(doc)
        first = doc.all_items()[0]
        editor.bring_to_front(first.id)
        assert doc.all_items()[-1].id == first.id

    def test_remove_item(self) -> None:
        doc = self._doc()
        editor = LayoutEditor(doc)
        item = doc.all_items()[0]
        editor.remove(item.id)
        assert doc.item_count() == 2
        assert doc.find_item(item.id) is None


# ── rendering ─────────────────────────────────────────────────────

class TestRender:
    def test_render_page_png(self) -> None:
        doc = LayoutDocument.from_bytes(
            (FIXTURES / "simple.designcraft").read_bytes())
        png = render_page(doc, doc.spreads[0].pages[0].id, scale=1.0)
        assert png[:8] == b"\x89PNG\r\n\x1a\n"
        from PIL import Image
        image = Image.open(io.BytesIO(png))
        assert image.size == (612, 792)

    def test_render_red_rect_pixel(self) -> None:
        doc = LayoutDocument.from_bytes(
            (FIXTURES / "simple.designcraft").read_bytes())
        png = render_page(doc, doc.spreads[0].pages[0].id, scale=1.0)
        from PIL import Image
        image = Image.open(io.BytesIO(png)).convert("RGB")
        # The fixture's first item is a red rect at (36, 36, 200, 100).
        assert image.getpixel((100, 80)) == (255, 0, 0)

    def test_render_text_has_ink(self) -> None:
        doc = LayoutDocument.from_bytes(
            (FIXTURES / "simple.designcraft").read_bytes())
        png = render_page(doc, doc.spreads[0].pages[0].id, scale=1.0)
        from PIL import Image
        image = Image.open(io.BytesIO(png)).convert("L")
        # Text frame at (36, 160, 300, 100): some pixels are not white.
        region = image.crop((36, 160, 336, 260))
        assert min(region.getdata()) < 200

    def test_render_document_renders_every_page(self) -> None:
        doc = LayoutDocument.from_bytes(
            (FIXTURES / "magazine.designcraft").read_bytes())
        pages = render_document(doc, scale=0.5)
        assert len(pages) == 2
        assert all(p[:8] == b"\x89PNG\r\n\x1a\n" for p in pages)

    def test_hidden_layer_not_rendered(self) -> None:
        doc = LayoutDocument.from_bytes(
            (FIXTURES / "simple.designcraft").read_bytes())
        layer = doc.layers[0]
        layer.visible = False
        png = render_page(doc, doc.spreads[0].pages[0].id, scale=1.0)
        from PIL import Image
        image = Image.open(io.BytesIO(png)).convert("RGB")
        assert image.getpixel((100, 80)) == (255, 255, 255)

    def test_render_scale_rejects_zero(self) -> None:
        doc = LayoutDocument.blank()
        with pytest.raises(LayoutError) as exc:
            render_page(doc, doc.spreads[0].pages[0].id, scale=0)
        assert exc.value.kind == "invalid_argument"


# ── parity + attribution ──────────────────────────────────────────

class TestParity:
    def test_parity_passes(self) -> None:
        result = run_parity()
        assert result["failed"] == []
        assert result["passed"] == 10

    def test_missing_fixture_fails(self, tmp_path) -> None:
        toml = tmp_path / "missing.toml"
        toml.write_text(
            'schema_version = 1\n'
            '[[cases]]\nid = "ghost"\nfixture = "no-such-file.designcraft"\n'
            'ops = ["inspect"]\n[cases.expect]\nitems = 1\n')
        result = run_parity(toml)
        assert result["failed"] == ["ghost"]
        assert result["passed"] == 0


class TestAttribution:
    def test_attribution_verifies(self) -> None:
        result = verify_attribution()
        assert result["failed"] == []
        assert result["skipped"] == []
        assert result["checked"] >= 10


# ── brain tool ────────────────────────────────────────────────────

class TestTool:
    def _registry(self, home):
        registry = ToolRegistry(context=SimpleNamespace(
            settings=SimpleNamespace(), home=Path(home)))
        registry.register_builtins()
        return registry

    def test_tool_registered(self) -> None:
        with tempfile.TemporaryDirectory() as home:
            registry = self._registry(home)
            assert "designcraft" in registry._tools

    def test_tool_inspect_and_render(self) -> None:
        with tempfile.TemporaryDirectory() as home:
            registry = self._registry(home)
            src = str(FIXTURES / "simple.designcraft")
            outcome = registry.call("designcraft", action="inspect",
                                    sources=[src])
            assert outcome.ok is True
            assert outcome.value["items"] == 3
            out = Path(home) / "page.png"
            outcome = registry.call(
                "designcraft", action="render", sources=[src],
                output=str(out))
            assert outcome.ok is True
            assert out.exists()

    def test_tool_unknown_action(self) -> None:
        with tempfile.TemporaryDirectory() as home:
            registry = self._registry(home)
            outcome = registry.call("designcraft", action="nope")
            assert outcome.ok is False
            assert "unknown" in str(outcome.error).lower()
