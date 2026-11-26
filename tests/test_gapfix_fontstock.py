"""Pins for the font-stock lane (audit D3): the full pack stocks
itself in bounded, resumable batches; a partial pack is never
stocked as the library; /fonts and `nm fonts` report the truth.

Fixtures are small synthetic google/fonts checkouts (local
source dirs) built over host DejaVu faces — the suite never
depends on the real ~2.5 GB pack or the network.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pytest

from nomorals.media.typography import fontlib, fontpack
from nomorals.media.typography.fontlib import FontLibrary

_DEJAVU = Path("/usr/share/fonts/truetype/dejavu")
pytestmark = pytest.mark.skipif(
    not (_DEJAVU / "DejaVuSans.ttf").is_file(),
    reason="host DejaVu faces are the fixture source")

# (bucket, dir, family name, licence id, licence file, faces)
_FAMILIES = (
    ("ofl", "alpha", "Alpha Sans", "OFL", "OFL.txt",
     [("AlphaSans-Regular.ttf", "DejaVuSans.ttf", 400, "normal"),
      ("AlphaSans-Bold.ttf", "DejaVuSans-Bold.ttf", 700, "normal")]),
    ("ofl", "beta", "Beta Serif", "OFL", "OFL.txt",
     [("BetaSerif-Regular.ttf", "DejaVuSerif.ttf", 400, "normal")]),
    ("ofl", "gamma", "Gamma Mono", "OFL", "OFL.txt",
     [("GammaMono-Regular.ttf", "DejaVuSansMono.ttf", 400, "normal"),
      ("GammaMono-Italic.ttf", "DejaVuSansMono.ttf", 400, "italic")]),
    ("apache", "delta", "Delta Sans", "APACHE2", "LICENSE.txt",
     [("DeltaSans-Regular.ttf", "DejaVuSans.ttf", 400, "normal")]),
    ("ofl", "zeta", "Zeta Display", "OFL", "OFL.txt",
     [("ZetaDisplay-Regular.ttf", "DejaVuSans-Bold.ttf", 400,
       "normal")]),
)
# No TTF faces at all: a NAMED exclusion, never a silent drop.
_EXCLUDED = ("ofl", "epsilon", "Epsilon Empty", "OFL", "OFL.txt")


def _write_source(root: Path) -> Path:
    """A fake google/fonts checkout: 5 stockable families + 1
    family with no faces (named exclusion)."""
    root.mkdir(parents=True, exist_ok=True)
    for bucket, dirname, name, licence, licence_file, faces in \
            _FAMILIES:
        fam_dir = root / bucket / dirname
        fam_dir.mkdir(parents=True)
        lines = [f'name: "{name}"', f'license: "{licence}"',
                 'category: "SANS_SERIF"', 'subsets: "latin"']
        for filename, _src, weight, style in faces:
            lines.append("fonts {")
            lines.append(f'  filename: "{filename}"')
            lines.append(f'  style: "{style}"')
            lines.append(f"  weight: {weight}")
            lines.append("}")
            shutil.copyfile(_DEJAVU / _src, fam_dir / filename)
        (fam_dir / "METADATA.pb").write_text(
            "\n".join(lines) + "\n", encoding="utf-8")
        (fam_dir / licence_file).write_text(
            "SIL Open Font License 1.1\n", encoding="utf-8")
    bucket, dirname, name, licence, licence_file = _EXCLUDED
    fam_dir = root / bucket / dirname
    fam_dir.mkdir(parents=True)
    (fam_dir / "METADATA.pb").write_text(
        f'name: "{name}"\nlicense: "{licence}"\n'
        'category: "SANS_SERIF"\nsubsets: "latin"\n',
        encoding="utf-8")
    (fam_dir / licence_file).write_text(
        "SIL Open Font License 1.1\n", encoding="utf-8")
    return root


@pytest.fixture()
def source_dir(tmp_path):
    return _write_source(tmp_path / "google-fonts")


@pytest.fixture()
def build_dir(tmp_path):
    return tmp_path / "build"


def _manifest(build_dir):
    return json.loads(
        (build_dir / "MANIFEST.json").read_text(encoding="utf-8"))


def _state(build_dir):
    return json.loads(
        (build_dir / "stock-state.json").read_text(encoding="utf-8"))


# ── resumable batches ────────────────────────────────────────

class TestResumableBatches:
    def test_batches_land_everything_and_resume(
            self, source_dir, build_dir):
        first = fontpack.build_pack_resumable(
            build_dir, source_dir=source_dir, batch_families=2,
            max_batches=1, workers=2)
        assert first["complete"] is False
        assert first["families_done"] == 2
        assert first["families_pending"] == 4
        assert first["manifest_kind"] == "partial"
        manifest = _manifest(build_dir)
        assert manifest["kind"] == "partial"
        assert manifest["pending_families"] == 4
        assert manifest["family_count"] == 2
        assert _state(build_dir)["done"] == [
            "apache/delta", "ofl/alpha"]

        second = fontpack.build_pack_resumable(
            build_dir, source_dir=source_dir, batch_families=2,
            max_batches=1, workers=2)
        assert second["complete"] is False
        assert second["families_done"] == 3
        assert second["families_pending"] == 2
        excluded = _state(build_dir)["excluded"]
        assert [e["dir"] for e in excluded] == ["ofl/epsilon"]
        assert "no TTF faces" in excluded[0]["reason"]

        third = fontpack.build_pack_resumable(
            build_dir, source_dir=source_dir, batch_families=2,
            workers=2)
        assert third["complete"] is True
        assert third["families_done"] == 5
        assert third["families_pending"] == 0
        assert third["families_this_run"] == 2
        manifest = _manifest(build_dir)
        assert manifest["kind"] == "full"
        assert manifest["pending_families"] == 0
        assert manifest["family_count"] == 5
        assert manifest["file_count"] == 7
        assert len(manifest["exclusions"]) >= 1
        assert fontpack.verify_pack(build_dir, full=True)["ok"]

        again = fontpack.build_pack_resumable(
            build_dir, source_dir=source_dir, workers=2)
        assert again["complete"] is True
        assert again["families_this_run"] == 0

    def test_missing_files_are_refetched_not_claimed(
            self, source_dir, build_dir):
        fontpack.build_pack_resumable(
            build_dir, source_dir=source_dir, batch_families=2,
            max_batches=1, workers=2)
        assert (build_dir / "ofl/alpha").is_dir()
        shutil.rmtree(build_dir / "ofl/alpha")
        rerun = fontpack.build_pack_resumable(
            build_dir, source_dir=source_dir, batch_families=2,
            max_batches=1, workers=2)
        assert "ofl/alpha" in _state(build_dir)["done"]
        assert (build_dir / "ofl/alpha"
                / "AlphaSans-Regular.ttf").is_file()
        # alpha re-fetched plus beta: 3 done, not the 2 the
        # stale state claimed.
        assert rerun["families_done"] == 3
        assert rerun["families_this_run"] == 2

    def test_different_commit_is_refused(
            self, source_dir, build_dir):
        fontpack.build_pack_resumable(
            build_dir, source_dir=source_dir, batch_families=2,
            max_batches=1, workers=2)
        with pytest.raises(fontpack.FontPackError) as caught:
            fontpack.build_pack_resumable(
                build_dir, source_dir=source_dir, commit="deadbeef",
                workers=2)
        assert "different" in str(caught.value) or \
            "fresh" in str(caught.value)


# ── the stock gate ───────────────────────────────────────────

class TestStockGate:
    def test_stock_pack_refuses_partial(
            self, source_dir, build_dir, tmp_path):
        fontpack.build_pack_resumable(
            build_dir, source_dir=source_dir, batch_families=2,
            max_batches=1, workers=2)
        with pytest.raises(fontpack.FontPackError) as caught:
            fontpack.stock_pack(build_dir, home=tmp_path / "home")
        assert "partial" in str(caught.value)
        assert "4 families still pending" in str(caught.value)
        assert not (tmp_path / "home" / "fonts"
                    / "library").exists()

    def test_stock_pack_stocks_the_whole_pack(
            self, source_dir, build_dir, tmp_path):
        fontpack.build_pack_resumable(
            build_dir, source_dir=source_dir, workers=2)
        home = tmp_path / "home"
        report = fontpack.stock_pack(build_dir, home=home)
        assert report["families"] == 5
        dest = fontlib.estate_library_dir(home)
        assert (dest / "MANIFEST.json").is_file()
        assert not (dest / "stock-state.json").exists()
        lib = FontLibrary.discover(home=home)
        assert lib.family("Alpha Sans").tier == "pack"
        assert lib.resolve("Gamma Mono").is_file()


# ── ensure: the auto-stock driver ────────────────────────────

class TestEnsure:
    def test_ensure_stocks_local_pack_end_to_end(
            self, source_dir, tmp_path, monkeypatch):
        home = tmp_path / "home"
        monkeypatch.setenv("NM_HOME", str(home))
        monkeypatch.setenv("NM_FONT_DIRS", "")
        report = fontpack.ensure_font_pack(
            home=home, source_dir=source_dir, batch_families=2,
            workers=2)
        assert report["status"] == "stocked"
        assert report["families"] == 5
        lib = FontLibrary.discover(home=home)
        assert lib.family("Delta Sans").tier == "pack"
        assert lib.resolve("Zeta Display").is_file()
        again = fontpack.ensure_font_pack(
            home=home, source_dir=source_dir, workers=2)
        assert again["status"] == "stocked"
        assert "already stocked" in again["detail"]

    def test_ensure_failure_is_honest_and_cools_down(
            self, tmp_path, monkeypatch):
        home = tmp_path / "home"
        monkeypatch.setenv("NM_HOME", str(home))
        monkeypatch.setenv("NM_FONT_DIRS", "")
        missing = tmp_path / "no-such-checkout"
        first = fontpack.ensure_font_pack(
            home=home, source_dir=missing, workers=2)
        assert first["status"] == "failed"
        assert "no-such-checkout" in first["detail"]
        assert not fontlib.estate_library_dir(home).exists()
        status = fontpack.stock_status(home=home)
        assert status["pack"]["stocked"] is False
        assert status["build"]["last_error"]
        second = fontpack.ensure_font_pack(
            home=home, source_dir=missing, workers=2)
        assert second["status"] == "deferred"
        third = fontpack.ensure_font_pack(
            home=home, source_dir=missing, force=True, workers=2)
        assert third["status"] == "failed"

    def test_ensure_blocked_when_disk_short(
            self, source_dir, tmp_path, monkeypatch):
        home = tmp_path / "home"
        monkeypatch.setattr(fontpack, "_free_bytes",
                            lambda _path: 0)
        report = fontpack.ensure_font_pack(
            home=home, source_dir=source_dir, workers=2)
        assert report["status"] == "blocked"
        assert "disk short" in report["detail"]
        assert not fontlib.estate_library_dir(home).exists()

    def test_max_batches_leaves_honest_partial(
            self, source_dir, tmp_path):
        home = tmp_path / "home"
        report = fontpack.ensure_font_pack(
            home=home, source_dir=source_dir, batch_families=2,
            max_batches=1, workers=2)
        assert report["status"] == "partial"
        assert report["families_done"] == 2
        assert report["families_pending"] == 4
        assert "resumes" in report["detail"]
        assert not fontlib.estate_library_dir(home).exists()


# ── stock status: the truth, read from disk ──────────────────

class TestStockStatus:
    def test_status_reports_progress_then_stocked(
            self, source_dir, tmp_path, monkeypatch):
        home = tmp_path / "home"
        monkeypatch.setenv("NM_HOME", str(home))
        monkeypatch.setenv("NM_FONT_DIRS", "")
        fresh = fontpack.stock_status(home=home)
        assert fresh["pack"]["stocked"] is False
        assert fresh["build"]["started"] is False
        assert "not stocked" in fresh["summary"]
        fontpack.build_pack_resumable(
            fontlib.estate_library_dir(home).parent
            / "library.build", source_dir=source_dir,
            batch_families=2, max_batches=1, workers=2)
        mid = fontpack.stock_status(home=home)
        assert mid["build"]["started"] is True
        assert mid["build"]["families_done"] == 2
        assert mid["build"]["families_total"] == 6
        assert mid["build"]["families_pending"] == 4
        assert "2 of 6 families fetched so far" in mid["summary"]
        fontpack.ensure_font_pack(
            home=home, source_dir=source_dir, workers=2)
        done = fontpack.stock_status(home=home)
        assert done["pack"]["stocked"] is True
        assert done["pack"]["families"] == 5
        assert "full pack stocked: 5 families" in done["summary"]


# ── /fonts + nm fonts surfaces ───────────────────────────────

class TestSurfaces:
    def _runtime(self):
        from nomorals.agents.partner.runtime_fonts import (
            RuntimeFontsMixin,
        )

        class _Stub(RuntimeFontsMixin):
            def _is_operator(self, message):
                return True

        return _Stub()

    def test_fonts_overview_names_partial_progress(
            self, source_dir, tmp_path, monkeypatch):
        home = tmp_path / "home"
        monkeypatch.setenv("NM_HOME", str(home))
        monkeypatch.setenv("NM_FONT_DIRS", "")
        fontpack.build_pack_resumable(
            fontlib.estate_library_dir(home).parent
            / "library.build", source_dir=source_dir,
            batch_families=2, max_batches=1, workers=2)
        overview = self._runtime()._control_fonts("fonts", "", None)
        assert "full pack" in overview
        assert "not stocked" in overview
        assert "2 of 6 families fetched so far" in overview

    def test_fonts_overview_after_stock_has_no_progress_line(
            self, source_dir, tmp_path, monkeypatch):
        home = tmp_path / "home"
        monkeypatch.setenv("NM_HOME", str(home))
        monkeypatch.setenv("NM_FONT_DIRS", "")
        fontpack.ensure_font_pack(
            home=home, source_dir=source_dir, workers=2)
        overview = self._runtime()._control_fonts("fonts", "", None)
        assert "5 families" in overview
        assert "fetched so far" not in overview

    def test_cli_status_and_stock_auto(
            self, source_dir, tmp_path, monkeypatch, capsys):
        from nomorals.cmdline.commands.fonts import _cmd_fonts

        home = tmp_path / "home"
        monkeypatch.setenv("NM_HOME", str(home))
        monkeypatch.setenv("NM_FONT_DIRS", "")
        rc = _cmd_fonts(
            argparse.Namespace(fonts_action="status"), None)
        assert rc == 0
        out = json.loads(capsys.readouterr().out)
        assert out["pack"]["stocked"] is False
        rc = _cmd_fonts(argparse.Namespace(
            fonts_action="stock", auto=True, batch=2,
            source_dir=str(source_dir), force=False,
            pack_dir="", build=False, commit=""), None)
        assert rc == 0
        report = json.loads(capsys.readouterr().out)
        assert report["status"] == "stocked"
        assert report["families"] == 5
        rc = _cmd_fonts(
            argparse.Namespace(fonts_action="status"), None)
        assert rc == 0
        out = json.loads(capsys.readouterr().out)
        assert out["pack"]["stocked"] is True
        assert out["pack"]["families"] == 5

    def test_boot_hook_is_quiet_when_stocked_and_never_raises(
            self, source_dir, tmp_path, monkeypatch):
        from nomorals.agents.partner.runtime_provision import (
            RuntimeProvisionMixin,
        )

        home = tmp_path / "home"
        monkeypatch.setenv("NM_HOME", str(home))
        fontpack.ensure_font_pack(
            home=home, source_dir=source_dir, workers=2)

        home_arg = str(home)

        class _Settings:
            home = home_arg

        class _Stub(RuntimeProvisionMixin):
            settings = _Settings()

        result = _Stub()._ensure_font_pack_stocked()
        assert result["status"] == "stocked"

    def test_boot_hook_never_raises_and_opt_out_is_honoured(
            self, tmp_path, monkeypatch):
        from nomorals.agents.partner.runtime_provision import (
            RuntimeProvisionMixin,
        )

        home = tmp_path / "home"
        monkeypatch.setenv("NM_HOME", str(home))

        home_arg = str(home)

        class _Settings:
            home = home_arg

        class _Stub(RuntimeProvisionMixin):
            settings = _Settings()

        # The driver itself blowing up must still be a status,
        # never an exception out of the boot hook.
        def _boom(*_args, **_kwargs):
            raise RuntimeError("boom in the driver")

        monkeypatch.setattr(
            "nomorals.media.typography.fontpack.ensure_font_pack",
            _boom)
        result = _Stub()._ensure_font_pack_stocked()
        assert result["status"] == "skipped"
        # And the owner opt-out is honoured without touching
        # the driver at all.
        monkeypatch.setenv("NM_FONT_AUTOSTOCK", "0")
        result = _Stub()._ensure_font_pack_stocked()
        assert result["status"] == "skipped"
        assert result["detail"] == "NM_FONT_AUTOSTOCK=0"
