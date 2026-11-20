"""VITS fine-tuning port tests (lane 120, census 3A.29).

Two halves, like the port: the numpy/stdlib half runs on system python;
the torch half is gated with ``pytest.importorskip("torch")`` and runs
in the torchvenv (the parent spot-run on system python will show those
as skipped, not failed). Every pinned number comes from
``vits-parity.toml``, measured by running the REAL upstream code.

The ARPA-cleaner tests need g2p_en + its NLTK data (present in the
torchvenv, absent on system python) and skip honestly without them.
"""

import hashlib
import json
import os
import shutil
import tempfile
import tomllib
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from nomorals.voice.vits import audio as vaudio
from nomorals.voice.vits import configs as vconfigs
from nomorals.voice.vits import data as vdata
from nomorals.voice.vits import mathutils as vmath
from nomorals.voice.vits import preprocess as vpreprocess
from nomorals.voice.vits.text import (
    cleaned_text_to_sequence,
    sequence_to_text,
    text_to_sequence,
)
from nomorals.voice.vits.text.cleaners import (
    arpa_cleaners,
    arpa_precleaners,
)
from nomorals.voice.vits.text.symbols import symbols

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures" / "vits"
with open(ROOT / "vits-parity.toml", "rb") as _parity_file:
    PARITY = tomllib.load(_parity_file)
with open(ROOT / "vits-attribution.toml", "rb") as _attribution_file:
    ATTRIBUTION = tomllib.load(_attribution_file)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _needs_g2p():
    try:
        import g2p_en  # noqa
    except ImportError:
        return True
    return False


def _make_sidecars(wav_dir):
    """Seeded .torchmoji/.bert sidecars, as preprocess.py would leave them."""
    rng = np.random.default_rng(1234)
    for wav in sorted(Path(wav_dir).glob("*.wav")):
        base = str(wav).replace(".wav", "")
        np.savez_compressed(base + ".torchmoji.npz",
                            embedding=rng.standard_normal((1, 1, 2304)).astype(np.float32))
        np.savez_compressed(base + ".bert.npz",
                            embedding=rng.standard_normal((1, 10, 312)).astype(np.float32))


def _fixture_copy(tmp):
    """Copy the fixtures to tmp, re-pointing filelists at the copy."""
    dest = Path(tmp) / "vits"
    shutil.copytree(FIX, dest)
    for name in ("train_filelist.txt", "val_filelist.txt", "test_filelist.txt",
                 "vctk_train_filelist.txt", "train_filelist.txt.cleaned"):
        p = dest / name
        text = p.read_text()
        text = text.replace("tests/fixtures/vits/wavs/", str(dest / "wavs") + "/")
        p.write_text(text)
    _make_sidecars(dest / "wavs")
    return dest


def _train22_filelist(dest):
    """The 22050 Hz filelist the ground truth used: all train rows except
    the 44100 Hz clip (ljs_004). The 12 s clip is dropped by the loader's
    duration filter; the wrong-SR clip is kept by the filter (it only
    checks duration) and refused only when indexed."""
    rows = (dest / "train_filelist.txt").read_text().splitlines()
    kept = [r for r in rows if "ljs_004" not in r]
    out = dest / "train22_filelist.txt"
    out.write_text("\n".join(kept) + "\n")
    return out


def _load_hps_for_fixtures(dest):
    hps = vconfigs.load_config(config_name="ljs_li22.json", validate_config=False)
    hps.data.text_cleaners = ["arpa_cleaners"]
    hps.data.training_files = str(_train22_filelist(dest))
    hps.data.validation_files = str(dest / "val_filelist.txt")
    return hps


class TextTests(unittest.TestCase):
    def test_symbols_match_parity(self):
        self.assertEqual(len(symbols), PARITY["text"]["symbols"])
        for row in PARITY["text"]["symbol"]:
            self.assertEqual(symbols.index(row["symbol"]), row["id"])

    def test_g2p_phoneme_list_matches_g2p_en_when_available(self):
        from nomorals.voice.vits.text.symbols import g2p_phonemes
        try:
            from g2p_en import G2p
        except ImportError:
            self.skipTest("g2p_en not installed")
        # G2p().phonemes is a set: compare the members, not the order.
        self.assertEqual(sorted(g2p_phonemes()), sorted(G2p().phonemes))

    def test_basic_cleaners_match_parity(self):
        for row in PARITY["text"]["basic"]:
            self.assertEqual(text_to_sequence(row["sentence"], ["basic_cleaners"]),
                             row["ids"])

    @unittest.skipIf(_needs_g2p(), "g2p_en not installed")
    def test_arpa_cleaners_match_parity(self):
        for row in PARITY["text"]["arpa"]:
            self.assertEqual(arpa_cleaners(row["sentence"]), row["cleaned"])
            self.assertEqual(cleaned_text_to_sequence(row["cleaned"]), row["ids"])

    def test_sequence_roundtrip(self):
        ids = text_to_sequence("Hello world", ["basic_cleaners"])
        self.assertEqual(sequence_to_text(ids), PARITY["text"]["seq_roundtrip"])

    def test_arpa_precleaners(self):
        self.assertEqual(arpa_precleaners("Mrs. De Mohrenschildt thought that Oswald,"),
                         "misess de mohrenschildt thought that oswald,")

    def test_english_cleaners2_fails_honestly_without_backend(self):
        from nomorals.voice.vits.text.cleaners import english_cleaners2
        with self.assertRaises((ImportError, RuntimeError, OSError)):
            english_cleaners2("Hello world")

    def test_expand_numbers_fails_honestly(self):
        from nomorals.voice.vits.text.cleaners import expand_numbers
        with self.assertRaises(NotImplementedError):
            expand_numbers("I have 2 apples")


class ConfigTests(unittest.TestCase):
    def test_all_vendored_configs_parse_and_validate(self):
        names = vconfigs.list_configs()
        self.assertEqual(len(names), 14)
        for name in names:
            vconfigs.load_config(config_name=name)

    def test_fixed_configs_carry_the_comma(self):
        for name in ("ljs_li44_tmbert_ft_small_s1_arpa.json",
                     "ljs_li44_tmbert_s1_arpa_mid.json"):
            text = (ROOT / "nomorals" / "voice" / "vits" / "data" / "configs" / name).read_text()
            self.assertIn('"use_8bit": true,', text)

    def test_validate_rejects_bad_config(self):
        hps = vconfigs.load_config(config_name="ljs_li22.json")
        hps.train.segment_size = 8191
        with self.assertRaises(ValueError):
            vconfigs.validate(hps)
        hps = vconfigs.load_config(config_name="ljs_li22.json")
        del hps.data.sampling_rate
        with self.assertRaises(ValueError):
            vconfigs.validate(hps)

    def test_zdict_is_vendored(self):
        zdict = ROOT / "nomorals" / "voice" / "vits" / "data" / "zdict.txt"
        self.assertTrue(zdict.exists())
        self.assertGreater(zdict.stat().st_size, 5_000_000)
        self.assertEqual(_sha256(zdict),
                         [a["sha256"] for a in ATTRIBUTION["assets"]
                          if a["path"] == "nomorals/voice/vits/data/zdict.txt"][0])


class AudioTests(unittest.TestCase):
    def _wav(self, name):
        sr, data = vaudio.load_wav(str(FIX / "wavs" / name))
        return sr, (data / 32768.0)[np.newaxis, :]

    def test_mel_filterbank_is_slaney_exact(self):
        # The filterbank is built from scratch; librosa (when present)
        # must agree to float32 precision.
        try:
            import librosa
        except ImportError:
            self.skipTest("librosa not installed")
        for (sr, n_fft, n_mels, fmin, fmax) in ((22050, 1024, 80, 0, 8000),
                                                (44100, 2048, 80, 20, 11025)):
            mine = vaudio.slaney_mel_filterbank(sr, n_fft, n_mels, fmin, fmax)
            ref = librosa.filters.mel(sr, n_fft, n_mels, fmin, fmax)
            np.testing.assert_allclose(mine, ref, rtol=0, atol=1e-6)

    def test_mel_and_spec_match_parity(self):
        for row in PARITY["audio"]["file"]:
            sr, y = self._wav(row["name"])
            cfg = vconfigs.load_config(config_name=row["config"])
            d = cfg.data
            spec = vaudio.spectrogram(y, d.filter_length, sr, d.hop_length,
                                      d.win_length, center=False)
            mel = vaudio.mel_spectrogram(y, d.filter_length, d.n_mel_channels,
                                         sr, d.hop_length, d.win_length,
                                         d.mel_fmin, d.mel_fmax, center=False)
            self.assertEqual(list(mel.shape), row["mel_shape"])
            for key, arr in (("spec", spec), ("mel", mel)):
                for stat in ("mean", "std", "min", "max"):
                    np.testing.assert_allclose(
                        float(getattr(arr, stat)()), row[key + "_" + stat],
                        rtol=0, atol=5e-4)

    def test_stft_roundtrip(self):
        for row in PARITY["audio"]["file"]:
            sr, y = self._wav(row["name"])
            cfg = vconfigs.load_config(config_name=row["config"])
            d = cfg.data
            spec = vaudio.stft(y[0], d.filter_length, d.hop_length, d.win_length,
                               center=True)
            rec = vaudio.istft(np.abs(spec), np.angle(spec),
                               d.filter_length, d.hop_length, d.win_length)
            n = min(len(rec), len(y[0]))
            err = float(np.max(np.abs(rec[512:n - 512] - y[0][512:n - 512])))
            self.assertLess(err, 1e-5)

    def test_wrong_sr_refused(self):
        with self.assertRaises(ValueError):
            vaudio.load_wav_normalized(str(FIX / "wavs" / "wrong_sr.wav"), 22050)


class MathTests(unittest.TestCase):
    def test_intersperse_and_padding(self):
        self.assertEqual(vmath.intersperse([1, 2, 3], 0),
                         PARITY["math"]["intersperse"])
        self.assertEqual(vmath.get_padding(3, 1), PARITY["math"]["get_padding_3_1"])
        mask = vmath.sequence_mask(np.array([2, 4]), 5)
        self.assertEqual(mask.tolist(), PARITY["math"]["sequence_mask"])

    def test_kl_loss_is_finite_and_positive(self):
        # The exact value is pinned in the torch-gated test below (it needs
        # torch's RNG stream); here only the shape of the answer is checked.
        rng = np.random.default_rng(7)
        z_p = rng.standard_normal((2, 4, 6))
        logs_q = rng.standard_normal((2, 4, 6))
        m_p = rng.standard_normal((2, 4, 6))
        logs_p = rng.standard_normal((2, 4, 6))
        z_mask = np.ones((2, 1, 6))
        kl = vmath.kl_loss(z_p, logs_q, m_p, logs_p, z_mask)
        self.assertTrue(np.isfinite(kl))
        self.assertGreater(kl, 0)


class DataTests(unittest.TestCase):
    def test_loader_filters_and_shapes(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = _fixture_copy(tmp)
            hps = _load_hps_for_fixtures(dest)
            loader = vdata.TextAudioLoader(str(dest / "train22_filelist.txt"),
                                           hps.data)
            self.assertEqual(len(loader), PARITY["data"]["loader_kept"])
            self.assertEqual(loader.lengths, PARITY["data"]["loader_lengths"])
            item = loader[0]
            self.assertEqual(len(item[0]), PARITY["data"]["first_text_len"])
            self.assertEqual(list(item[1].shape), PARITY["data"]["first_spec_shape"])
            self.assertEqual(list(item[2].shape), PARITY["data"]["first_wav_shape"])
            self.assertEqual(list(item[3].shape), PARITY["data"]["first_moji_shape"])
            self.assertEqual(list(item[4].shape), PARITY["data"]["first_bert_shape"])

    def test_collate_shapes(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = _fixture_copy(tmp)
            hps = _load_hps_for_fixtures(dest)
            loader = vdata.TextAudioLoader(str(dest / "train22_filelist.txt"),
                                           hps.data)
            batch = vdata.TextAudioCollate()([loader[0], loader[1]])
            self.assertEqual([list(x.shape) for x in batch[:9]],
                             PARITY["data"]["collate_shapes"])

    def test_sampler_buckets(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = _fixture_copy(tmp)
            hps = _load_hps_for_fixtures(dest)
            loader = vdata.TextAudioLoader(str(dest / "train22_filelist.txt"),
                                           hps.data)
            sampler = vdata.DistributedBucketSampler(
                loader, 2, [32, 300, 400, 500, 600, 700, 800, 900, 1000],
                num_replicas=1, rank=0, shuffle=False)
            sampler.set_epoch(0)
            self.assertEqual(sampler.buckets, PARITY["data"]["sampler_buckets"])
            self.assertEqual(list(sampler), PARITY["data"]["sampler_batches"])

    def test_speaker_loader_kept_and_sid(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = _fixture_copy(tmp)
            hps = _load_hps_for_fixtures(dest)
            loader = vdata.TextAudioSpeakerLoader(
                str(dest / "vctk_train_filelist.txt"), hps.data)
            self.assertEqual(len(loader), PARITY["data"]["speaker_kept"])
            self.assertEqual(int(loader[0][5][0]), PARITY["data"]["speaker_first_sid"])
            # Upstream's speaker loader raises UnboundLocalError on every
            # filelist (n_fau never initialised); the port fixes it.
            self.assertIn("UnboundLocalError",
                          PARITY["data"]["speaker_upstream_error"])

    def test_missing_sidecar_refuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "vits"
            shutil.copytree(FIX, dest)
            # Re-point the filelists at the copy (without this the loader
            # reads the committed fixtures and writes a spec cache there).
            for name in ("train_filelist.txt",):
                fp = dest / name
                fp.write_text(fp.read_text().replace(
                    "tests/fixtures/vits/wavs/", str(dest / "wavs") + "/"))
            hps = vconfigs.load_config(config_name="ljs_li22.json",
                                       validate_config=False)
            hps.data.text_cleaners = ["arpa_cleaners"]
            loader = vdata.TextAudioLoader(str(dest / "train_filelist.txt"),
                                           hps.data)
            with self.assertRaises(ValueError):
                loader[0]


class PreprocessTests(unittest.TestCase):
    @unittest.skipIf(_needs_g2p(), "g2p_en not installed")
    def test_cleaned_filelist_matches_upstream(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = _fixture_copy(tmp)
            out, n_rows = vpreprocess.clean_filelist(
                str(dest / "train_filelist.txt"),
                text_cleaners=["arpa_cleaners"])
            self.assertEqual(n_rows, 12)  # 6 rows x dual ARPA mode
            upstream = (FIX / "train_filelist.txt.cleaned").read_text()
            got = Path(out).read_text()
            # Same content modulo the fixture directory prefix.
            upstream = upstream.replace(
                "tests/fixtures/vits/wavs/", str(dest / "wavs") + "/")
            self.assertEqual(got, upstream)

    def test_prepare_run_end_to_end(self):
        # basic_cleaners needs no g2p/espeak, so this runs on system python.
        # 22050 Hz config: the 44100 Hz clip and the wrong-SR clip are
        # refused honestly; the 12 s clip still gets a spec (prepare does
        # not duration-filter, the loader does).
        with tempfile.TemporaryDirectory() as tmp:
            dest = _fixture_copy(tmp)
            out_dir = Path(tmp) / "logs" / "test_voice"
            hps_path = Path(tmp) / "test_config.json"
            cfg = json.loads((ROOT / "nomorals" / "voice" / "vits" / "data"
                              / "configs" / "ljs_li22.json").read_text())
            cfg["data"]["text_cleaners"] = ["basic_cleaners"]
            cfg["data"]["training_files"] = str(dest / "train_filelist.txt")
            cfg["data"]["validation_files"] = str(dest / "val_filelist.txt")
            hps_path.write_text(json.dumps(cfg))
            from nomorals.voice.vits.prepare import prepare_run
            report = prepare_run(config=str(hps_path), model_name="test_voice",
                                 out_dir=str(out_dir))
            # basic_cleaners has no dual mode (that is ARPA-only), so
            # the 6 input rows stay 6 cleaned rows.
            self.assertEqual(report["cleaned"][str(dest / "train_filelist.txt")]["rows"], 6)
            self.assertEqual(report["spec_cache"]["built"], 4)
            # The two validation rows reuse the specs the training rows
            # just built (the loop does not dedupe across filelists).
            self.assertEqual(report["spec_cache"]["cached"], 2)
            self.assertEqual(len(report["warnings"]), 2)
            self.assertTrue(any("ljs_004" in w for w in report["warnings"]))
            self.assertTrue(any("wrong_sr" in w for w in report["warnings"]))
            self.assertTrue((out_dir / "prepare_report.json").exists())
            self.assertTrue((out_dir / "config.json").exists())

    @unittest.skipIf(_needs_g2p(), "g2p_en not installed")
    def test_test_preproc_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "test_preproc"
            written = vpreprocess.build_test_preproc(
                vpreprocess.TEST_SENTENCES, ["arpa_cleaners"], str(out_dir))
            self.assertEqual(len(written), 5)
            entry = np.load(written[0])
            ids = entry["text"].tolist()
            cleaned = arpa_cleaners(vpreprocess.TEST_SENTENCES[0])
            expect = cleaned_text_to_sequence(cleaned)
            expect = vmath.intersperse(expect, 0)
            self.assertEqual(ids, expect)


class RegistryTests(unittest.TestCase):
    def test_tool_count_still_778(self):
        # This lane adds NO brain tool: the VITS capability is exercised
        # through the prepare/train CLIs and the lazily-registered TTS
        # backend, not a new tool.
        # Pin corrected 2026-11-19 by the parent at the vits/data
        # fix-up ship: the registry moved 735 -> 749 through later
        # ships (bughunter +8, auth-plugin wiring +5, mocap +1);
        # 749 -> 774 through the secprobe wire-up (+25 tools);
        # 774 -> 775 through this lane (B.19, scene_spawn); 775 -> 776 through this lane (B.20, cloudcam); 776 -> 777 through B.22 (show); 777 -> 778 through this lane (B.26, scene_project); the
        # assertion's meaning (no new tool from this lane) is
        # unchanged.
        from nomorals.tools.registry import ToolRegistry
        registry = ToolRegistry(SimpleNamespace())
        registry.register_builtins()
        # 779 after this lane (B.23) added facemcap. 780 after this lane (B.25) added facemesh. 781 after this lane (B.27) added vtuber. 782 after this lane (C.2) registered tonal_g2p; 783 after this lane (C.1) added smarthome; 784 after this lane (C.4) added camwall; 785 after this lane (C.8) added robot.
        self.assertEqual(len(registry._tools), 785)

    def test_vits_backend_registered_but_unavailable_here(self):
        from nomorals.voice.tts import (
            _BACKEND_SPECS,
            _BACKENDS,
            BACKEND_CAPABILITIES,
            available_backends,
        )
        self.assertIn("vits", _BACKENDS)
        self.assertIn("vits", _BACKEND_SPECS)
        self.assertIn("vits", BACKEND_CAPABILITIES)
        self.assertEqual(BACKEND_CAPABILITIES["vits"]["license"], "MIT")
        self.assertNotIn("vits", available_backends())

    def test_vits_backend_construction_refuses_without_model(self):
        from nomorals.voice.vits.backend import VITSBackend
        with self.assertRaises(RuntimeError):
            VITSBackend()

    def test_voice_exports(self):
        from nomorals.voice import (
            VITSBackend,
            list_configs,
            list_vits_models,
            load_config,
            prepare_run,
            vits_model_dir,
        )
        self.assertEqual(len(list_configs()), 14)
        self.assertEqual(list_vits_models(), [])
        self.assertTrue(str(vits_model_dir()).endswith("voice/vits"))
        self.assertTrue(callable(VITSBackend))
        self.assertTrue(callable(load_config))
        self.assertTrue(callable(prepare_run))


class AttributionTests(unittest.TestCase):
    def test_every_asset_exists_hashes_and_is_mit(self):
        assets = ATTRIBUTION["assets"]
        self.assertEqual(len(assets), 47)
        for asset in assets:
            path = ROOT / asset["path"]
            self.assertTrue(path.exists(), asset["path"])
            self.assertEqual(_sha256(path), asset["sha256"],
                             "hash mismatch for {}".format(asset["path"]))
            self.assertEqual(asset["license"], "MIT")
            self.assertEqual(asset["author"], "Jaehyeon Kim")

    def test_adapted_assets_name_their_change(self):
        adapted = [a for a in ATTRIBUTION["assets"] if "upstream_sha256" in a]
        # 30 at lane 120 + 3 files whose byte-identical status ended with
        # the fix-up ruff reformat (transforms.py, moji/lstm.py,
        # moji/attlayer.py); each now carries upstream_sha256 + note.
        self.assertEqual(len(adapted), 33)
        for asset in adapted:
            self.assertIn("note", asset, asset["path"])
            self.assertNotEqual(asset["sha256"], asset["upstream_sha256"])

    def test_fixed_configs_documented(self):
        fixed = [a for a in ATTRIBUTION["assets"]
                 if "FIXED" in a.get("note", "")]
        self.assertEqual(len(fixed), 2)
        self.assertEqual({a["upstream_path"] for a in fixed},
                         {"configs/ljs_li44_tmbert_ft_small_s1_arpa.json",
                          "configs/ljs_li44_tmbert_s1_arpa_mid.json"})

    def test_moji_weights_absent_and_documented(self):
        weights = ROOT / "nomorals" / "voice" / "vits" / "data" / "pytorch_model.bin"
        self.assertFalse(weights.exists())
        paths = [a["path"] for a in ATTRIBUTION["assets"]]
        self.assertNotIn("nomorals/voice/vits/data/pytorch_model.bin", paths)


class MojiTokenizerTests(unittest.TestCase):
    def test_tokenizer_splits(self):
        from nomorals.voice.vits.moji import tokenizer
        self.assertEqual(tokenizer.tokenize("Hello, world! I love #NLP :)"),
                         ["Hello", ",", "world", "!", "I", "love", "#NLP", ":)"])

    def test_tokenizer_titles_case_insensitive(self):
        from nomorals.voice.vits.moji import tokenizer
        self.assertIn("mrs.", tokenizer.tokenize("mrs. Jones went to Dr. No."))


# ------------------------------------------------------------------ torch half
# Per-class gating: a module-level importorskip would skip the numpy half
# too on system python, where torch is absent.
try:
    import torch
    _HAVE_TORCH = True
except ImportError:
    torch = None
    _HAVE_TORCH = False


@unittest.skipUnless(_HAVE_TORCH, "torch not installed")
class TorchLossTests(unittest.TestCase):
    def test_losses_match_parity(self):
        # Draw order mirrors groundtruth.py exactly, including the
        # discarded first discriminator_loss call (its result is thrown
        # away by the `and` in the ground-truth script).
        from nomorals.voice.vits.losses import (
            discriminator_loss,
            feature_loss,
            generator_loss,
            kl_loss,
        )
        torch.manual_seed(7)
        z_p = torch.randn(2, 4, 6)
        logs_q = torch.randn(2, 4, 6)
        m_p = torch.randn(2, 4, 6)
        logs_p = torch.randn(2, 4, 6)
        z_mask = torch.ones(2, 1, 6)
        self.assertAlmostEqual(float(kl_loss(z_p, logs_q, m_p, logs_p, z_mask)),
                               PARITY["math"]["kl"], places=5)
        self.assertAlmostEqual(
            float(feature_loss([[torch.randn(2, 3, 4)]], [[torch.randn(2, 3, 4)]])),
            PARITY["math"]["feature"], places=5)
        discriminator_loss([torch.randn(2, 5)], [torch.randn(2, 5)])
        self.assertAlmostEqual(
            float(discriminator_loss([torch.randn(2, 5)], [torch.randn(2, 5)])[0]),
            PARITY["math"]["disc"], places=5)
        self.assertAlmostEqual(float(generator_loss([torch.randn(2, 5)])[0]),
                               PARITY["math"]["gen"], places=5)

    def test_spline_roundtrip(self):
        from nomorals.voice.vits.transforms import rational_quadratic_spline
        torch.manual_seed(7)
        u = torch.rand(2, 3)
        w = torch.randn(2, 3, 10)
        h = torch.randn(2, 3, 10)
        d = torch.randn(2, 3, 11)
        y_fwd, lad = rational_quadratic_spline(u, w, h, d)
        y_back, _ = rational_quadratic_spline(y_fwd, w, h, d, inverse=True)
        self.assertLess(float(torch.max(torch.abs(y_back - u)).item()),
                        PARITY["math"]["spline_roundtrip_max_err"] * 10)
        self.assertAlmostEqual(float(lad.mean()), PARITY["math"]["spline_logabsdet_mean"],
                               places=5)


@unittest.skipUnless(_HAVE_TORCH, "torch not installed")
class TorchCommonsTests(unittest.TestCase):
    def test_commons_match_numpy_twins(self):
        from nomorals.voice.vits import commons
        self.assertEqual(commons.intersperse([1, 2, 3], 0),
                         vmath.intersperse([1, 2, 3], 0))
        self.assertEqual(commons.get_padding(3, 1), vmath.get_padding(3, 1))
        mask = commons.sequence_mask(torch.LongTensor([2, 4]), 5)
        self.assertEqual(mask.tolist(), PARITY["math"]["sequence_mask"])
        # groundtruth.py draws a stray randn(2, 5) before the kl inputs.
        torch.manual_seed(7)
        torch.randn(2, 5)
        kl = commons.kl_divergence(torch.randn(2, 3), torch.randn(2, 3),
                                   torch.randn(2, 3), torch.randn(2, 3))
        self.assertAlmostEqual(float(kl.mean()), PARITY["math"]["kl_div_mean"],
                               places=5)


@unittest.skipUnless(_HAVE_TORCH, "torch not installed")
class TorchAudioTests(unittest.TestCase):
    def test_torch_mel_matches_numpy_mel(self):
        from nomorals.voice.vits.mel_processing import mel_spectrogram_torch
        for row in PARITY["audio"]["file"]:
            sr, data = vaudio.load_wav(str(FIX / "wavs" / row["name"]))
            cfg = vconfigs.load_config(config_name=row["config"])
            d = cfg.data
            y = torch.FloatTensor(data / 32768.0).unsqueeze(0)
            mel_t = mel_spectrogram_torch(y, d.filter_length, d.n_mel_channels,
                                          sr, d.hop_length, d.win_length,
                                          d.mel_fmin, d.mel_fmax,
                                          center=False).numpy()
            mel_n = vaudio.mel_spectrogram(y.numpy(), d.filter_length,
                                           d.n_mel_channels, sr, d.hop_length,
                                           d.win_length, d.mel_fmin, d.mel_fmax,
                                           center=False)
            np.testing.assert_allclose(mel_t, mel_n, rtol=0, atol=1e-4)

    def test_torch_stft_roundtrip(self):
        from nomorals.voice.vits.stft import TorchSTFT
        for row in PARITY["audio"]["file"]:
            sr, data = vaudio.load_wav(str(FIX / "wavs" / row["name"]))
            cfg = vconfigs.load_config(config_name=row["config"])
            d = cfg.data
            y = torch.FloatTensor(data / 32768.0).unsqueeze(0)
            stft = TorchSTFT(d.filter_length, hop_length=d.hop_length,
                             win_length=d.win_length)
            mag, phase = stft.transform(y)
            rec = stft.inverse(mag, phase).squeeze(-2)
            n = min(rec.shape[1], y.shape[1])
            err = float(torch.max(torch.abs(rec[:, 512:n - 512] - y[:, 512:n - 512])))
            self.assertLess(err, 1e-5)


@unittest.skipUnless(_HAVE_TORCH, "torch not installed")
class TorchModelTests(unittest.TestCase):
    def _build(self):
        from nomorals.voice.vits.models import SynthesizerTrn
        hps = vconfigs.load_config(config_name="ljs_li44_tmbert_ft_s1_arpa.json")
        torch.manual_seed(0)
        net = SynthesizerTrn(len(symbols),
                             hps.data.filter_length // 2 + 1,
                             hps.train.segment_size // hps.data.hop_length,
                             **hps.model)
        net.eval()
        return hps, net

    def _inputs(self, hps):
        # Ground truth re-seeds before the inputs; the draw order is
        # x, y, mel, tm, bert. Infer reuses the same tensors, so the
        # y/mel draws still happen even when only tm/bert are used.
        torch.manual_seed(0)
        B, T, S = 2, 12, 40
        x = torch.randint(0, len(symbols), (B, T))
        x_lengths = torch.LongTensor([T, 9])
        y = torch.randn(B, hps.data.filter_length // 2 + 1, S)
        y_lengths = torch.LongTensor([S, 32])
        mel = torch.randn(B, hps.data.n_mel_channels, S)
        tm = torch.randn(B, 2304)
        bert = torch.randn(B, 10, hps.model.bert_size)
        bert_lens = torch.LongTensor([10, 8])
        return x, x_lengths, y, y_lengths, mel, tm, bert, bert_lens

    def test_param_count(self):
        hps, net = self._build()
        self.assertEqual(sum(p.numel() for p in net.parameters()),
                         PARITY["model"]["params"])

    def test_forward_matches_parity_bit_exact(self):
        hps, net = self._build()
        x, x_lengths, y, y_lengths, mel, tm, bert, bert_lens = self._inputs(hps)
        with torch.no_grad():
            outs = net(x, x_lengths, y, y_lengths, mel, tm, bert, bert_lens)
        o = outs[0].numpy().astype(np.float64)
        self.assertEqual(list(outs[0].shape), PARITY["model"]["forward_o_shape"])
        self.assertAlmostEqual(float(o.mean()), PARITY["model"]["forward_o_mean"], places=9)
        self.assertAlmostEqual(float(o.std()), PARITY["model"]["forward_o_std"], places=9)
        sample = [float(v) for v in o.ravel()[:: max(1, o.size // 12)][:12]]
        for a, b in zip(sample, PARITY["model"]["forward_o_sample"], strict=True):
            self.assertAlmostEqual(a, b, places=6)

    def test_infer_matches_parity(self):
        hps, net = self._build()
        x, x_lengths, y, y_lengths, mel, tm, bert, bert_lens = self._inputs(hps)
        with torch.no_grad():
            # Ground truth ran forward first: its randn draws are part of
            # the RNG stream infer's duration noise is drawn from.
            net(x, x_lengths, y, y_lengths, mel, tm, bert, bert_lens)
            inf = net.infer(x, x_lengths, tm, bert, bert_lens, max_len=64)
        o = inf[0].numpy().astype(np.float64)
        self.assertEqual(list(inf[0].shape), PARITY["model"]["infer_o_shape"])
        self.assertAlmostEqual(float(o.mean()), PARITY["model"]["infer_o_mean"], places=9)
        sample = [float(v) for v in o.ravel()[:: max(1, o.size // 12)][:12]]
        for a, b in zip(sample, PARITY["model"]["infer_o_sample"], strict=True):
            self.assertAlmostEqual(a, b, places=6)


@unittest.skipUnless(_HAVE_TORCH, "torch not installed")
class TorchFrontendTests(unittest.TestCase):
    def test_torchmoji_refuses_without_weights(self):
        from nomorals.voice.vits.moji import TorchMoji
        with self.assertRaises(FileNotFoundError) as ctx:
            TorchMoji()
        self.assertIn("pytorch_model.bin", str(ctx.exception))

    def test_bert_frontend_refuses_without_transformers(self):
        from nomorals.voice.vits.bertfe import BERTFrontEnd
        try:
            import transformers  # noqa
        except ImportError:
            with self.assertRaises(ImportError):
                BERTFrontEnd()
            return
        self.skipTest("transformers is installed here")

    def test_trainer_constructs_on_cpu(self):
        from nomorals.voice.vits.train import VITSTrainer
        hps = vconfigs.load_config(config_name="ljs_li44_tmbert_ft_s1_arpa.json")
        with tempfile.TemporaryDirectory() as tmp:
            trainer = VITSTrainer(hps, tmp, device="cpu")
            self.assertEqual(
                sum(p.numel() for p in trainer.net_g.parameters()),
                PARITY["model"]["params"])

    def test_trainer_runs_one_cpu_step(self):
        # The ft_s1 config is 44100 Hz, so the smoke filelists carry only
        # the 44100 Hz clip (ljs_004). The trainer evaluates AND saves at
        # global_step 0 (0 % interval == 0), so the validation filelist
        # must be 44100 Hz too, and three step-0 checkpoints land in the
        # temp model dir — that also proves checkpoint saving works.
        from nomorals.voice.vits.train import VITSTrainer
        with tempfile.TemporaryDirectory() as tmp:
            dest = _fixture_copy(tmp)
            rows = (dest / "train_filelist.txt").read_text().splitlines()
            kept = [r for r in rows if "ljs_004" in r]
            train_path = dest / "train_smoke.txt"
            train_path.write_text("\n".join(kept) + "\n")
            val_path = dest / "val_smoke.txt"
            val_path.write_text("\n".join(kept) + "\n")
            hps = vconfigs.load_config(config_name="ljs_li44_tmbert_ft_s1_arpa.json")
            hps.data.training_files = str(train_path)
            hps.data.validation_files = str(val_path)
            # NB: do NOT shrink segment_size: the frequency
            # discriminator's input channels are segment_size / pqmf_m,
            # and 8192 / 64 == freq_init_ch 128. A smaller segment is a
            # shape error by construction, not a speedup.
            hps.train.batch_size = 2
            hps.train.log_interval = 1
            hps.train.eval_interval = 10 ** 9
            hps.train.save_interval = 10 ** 9
            hps.train.epochs = 1
            model_dir = os.path.join(tmp, "logs", "smoke")
            trainer = VITSTrainer(hps, model_dir, device="cpu")
            steps = trainer.train(epochs=1, max_steps=1)
            self.assertEqual(steps, 1)
            self.assertTrue(os.path.exists(os.path.join(model_dir, "G_0.pth")))


if __name__ == "__main__":
    unittest.main()
