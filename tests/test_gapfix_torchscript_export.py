"""TorchScript export (census §3B.16, torchscript-models) — Devon's
own export, mined pattern-only from the upstream's three wrap
scripts + porting guide.

The upstream turns Glow-TTS / MelGAN checkpoints into ``.torchjit``
files with a hardcoded ``trace_model = True`` and a silent
``check_trace=False`` fallback after the sanity check fails, and
never verifies the export against its source. Devon's version:
the caller names trace or script and the receipt records it; the
trace check runs first and a failure names its cause (the
unchecked retry needs an explicit flag and is recorded); a
scripting failure names the upstream pitfall it matches; the
export is verified against the source on a fixed probe and the
max deviation is a number in the receipt; dynamic axes are a
script promise proven by a resized run; a long-input probe proves
the exported module survives longer input (the upstream's
finding); the artifact is always runnable on CPU; and loading
refuses a mismatched or corrupt artifact naming the cause.

Covers: the four scripting-pitfall diagnoses (+ unmatched ->
None), the torch gate, the receipt round-trip, script + trace
round-trips on a small real model (deviation asserted small),
dynamic axes (proven + refused on trace), the trace-check
failure path (simulated — this torch only warns), the reseed
proof (a stochastic module still verifies at 0.0), the fixed
branch a trace bakes in (caught by verification with the number
named), the long-input probe (enforced for script, advisory for
trace), device handling, load refusals (missing receipt, bad
stamp, sha mismatch, missing file, bad entry), and bad arguments.
Torch tests skip without torch (system python).
"""

import importlib.util
import json
import os
import tempfile
import unittest
from unittest import mock

import pytest

from nomorals.training import export as ex

TORCH = importlib.util.find_spec("torch") is not None

torch_only = pytest.mark.skipif(not TORCH, reason="needs torch")


def _tiny():
    import torch
    from torch import nn

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(4, 2, bias=False)

        def forward(self, x):
            return self.lin(x) * 2.0

    torch.manual_seed(0)
    return Tiny()


def _example():
    import torch

    return {"x": torch.tensor([[1.0, 2.0, 3.0, 4.0]])}


class DiagnosisTests(unittest.TestCase):
    def test_four_pitfalls_named(self):
        cases = [
            ("Expected an integer literal for index",
             "enumerate"),
            ("lhs of assignment must be a variable",
             "Break it up"),
            ("multiple comprehension generators not supported",
             "Unroll"),
            ("Expected a value of type 'Tensor' for argument 'length' "
             "but instead found type 'int'",
             "Type-hint"),
        ]
        for message, fragment in cases:
            diagnosis = ex.diagnose_script_error(ValueError(message))
            self.assertIsNotNone(diagnosis, message)
            self.assertIn(fragment, diagnosis)

    def test_unmatched_error_is_none(self):
        self.assertIsNone(
            ex.diagnose_script_error(ValueError("something else broke")))

    def test_receipt_path(self):
        self.assertEqual(ex.receipt_path_for("/a/b/model.torchjit"),
                         "/a/b/model.json")
        self.assertEqual(ex.receipt_path_for("/a/b/model.pt"),
                         "/a/b/model.json")

    def test_receipt_round_trip(self):
        verification = ex.Verification(
            probe_shapes={"x": [1, 4]}, max_abs_deviation=0.0,
            mean_abs_deviation=0.0, per_output=[0.0], tolerance=1e-4,
            passed=True, long_probe={"shapes": {"x": [8, 4]}, "ok": True,
                                     "error": None})
        receipt = ex.ExportReceipt(
            name="tiny", version="0.1.0", task="probe",
            format="torchscript", path="/tmp/tiny.torchjit",
            sha256="ab" * 32, inputs=[{"name": "x", "dtype": "float32",
                                       "shape": [1, 4]}],
            outputs=[{"name": "output_0", "dtype": "float32",
                      "shape": [1, 2]}],
            entry="forward", method="script", traced=False,
            trace_checked=False, trace_check_error=None,
            trace_check_warnings=[], dynamic_axes={"x": [0]},
            example_shapes={"x": [1, 4]}, export_device="cpu",
            runnable_on="cpu", torch_version="2.5.1",
            exporter_version=ex.EXPORT_FORMAT_VERSION,
            source="test", verification=verification)
        back = ex.ExportReceipt.from_dict(receipt.to_dict())
        self.assertEqual(back.to_dict(), receipt.to_dict())
        self.assertEqual(back.verification.max_abs_deviation, 0.0)
        self.assertTrue(back.verification.passed)


class NoTorchTests(unittest.TestCase):
    def test_export_refuses_without_torch(self):
        with mock.patch("nomorals.training.export._TORCH_AVAILABLE",
                        False), self.assertRaises(ImportError) as ctx:
            ex.export_module(object(), {"x": None}, "/tmp/x.torchjit",
                             name="t", task="probe")
        self.assertIn("torch", str(ctx.exception))

    def test_load_refuses_without_torch(self):
        with mock.patch("nomorals.training.export._TORCH_AVAILABLE",
                        False), self.assertRaises(ImportError) as ctx:
            ex.load_exported("/tmp/x.torchjit")
        self.assertIn("torch", str(ctx.exception))


@torch_only
class ExportTests(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.tmp = self._td.name

    def tearDown(self):
        self._td.cleanup()

    def _out(self, name):
        return os.path.join(self.tmp, name)

    def test_script_round_trip(self):
        import torch
        module = _tiny()
        out = self._out("tiny-script.torchjit")
        receipt = ex.export_module(
            module, _example(), out, method="script", name="tiny",
            task="probe", dynamic_axes={"x": [0]})
        self.assertEqual(receipt.method, "script")
        self.assertFalse(receipt.traced)
        self.assertEqual(receipt.export_device, "cpu")
        self.assertEqual(receipt.runnable_on, "cpu")
        self.assertEqual(receipt.verification.max_abs_deviation, 0.0)
        self.assertTrue(receipt.verification.passed)
        self.assertEqual(receipt.inputs[0]["shape"], [None, None])
        self.assertEqual(receipt.dynamic_axes, {"x": [0]})
        self.assertTrue(os.path.isfile(ex.receipt_path_for(out)))
        loaded = ex.load_exported(out)
        self.assertEqual(loaded.receipt.name, "tiny")
        self.assertIsNone(loaded.torch_version_warning)
        with torch.no_grad():
            got = loaded.module(torch.tensor([[1.0, 2.0, 3.0, 4.0]]))
            want = module(torch.tensor([[1.0, 2.0, 3.0, 4.0]]))
        self.assertEqual(got.shape, want.shape)
        self.assertAlmostEqual(
            float((got - want).abs().max()), 0.0, places=6)

    def test_trace_round_trip(self):
        module = _tiny()
        out = self._out("tiny-trace.torchjit")
        receipt = ex.export_module(
            module, _example(), out, method="trace", name="tiny",
            task="probe")
        self.assertEqual(receipt.method, "trace")
        self.assertTrue(receipt.traced)
        self.assertTrue(receipt.trace_checked)
        self.assertIsNone(receipt.trace_check_error)
        self.assertEqual(receipt.inputs[0]["shape"], [1, 4])
        self.assertEqual(receipt.verification.max_abs_deviation, 0.0)
        loaded = ex.load_exported(out)
        self.assertEqual(loaded.receipt.method, "trace")

    def test_trace_refuses_dynamic_axes(self):
        with self.assertRaises(ex.ExportError) as ctx:
            ex.export_module(_tiny(), _example(),
                             self._out("bad.torchjit"), method="trace",
                             name="tiny", task="probe",
                             dynamic_axes={"x": [0]})
        self.assertIn("dynamic_axes", str(ctx.exception))
        self.assertIn("script", str(ctx.exception))

    def test_trace_check_failure_path(self):
        # This torch only warns on a failed trace check, so the
        # upstream failure ("Graphs differed across invocations!")
        # is simulated by making trace_module raise once.
        import torch
        real_trace = torch.jit.trace_module
        calls = {"n": 0}

        def _flaky(module, inputs, **kwargs):
            calls["n"] += 1
            if kwargs.get("check_trace") and calls["n"] == 1:
                raise RuntimeError(
                    "Tracing failed sanity checks! Graphs differed "
                    "across invocations!")
            return real_trace(module, inputs, **kwargs)

        with mock.patch.object(torch.jit, "trace_module",
                               _flaky), self.assertRaises(
                                   ex.ExportError) as ctx:
            ex.export_module(_tiny(), _example(),
                             self._out("flaky.torchjit"),
                             method="trace", name="tiny", task="probe")
        self.assertIn("allow_unchecked_trace", str(ctx.exception))
        calls["n"] = 0
        with mock.patch.object(torch.jit, "trace_module", _flaky):
            receipt = ex.export_module(
                _tiny(), _example(), self._out("flaky.torchjit"),
                method="trace", name="tiny", task="probe",
                allow_unchecked_trace=True)
        self.assertFalse(receipt.trace_checked)
        self.assertIn("Graphs differed", receipt.trace_check_error or "")
        self.assertTrue(receipt.verification.passed)

    def test_stochastic_module_verifies_at_zero(self):
        # A module that draws noise in forward: without the reseed
        # before each verification run the deviation would be noise.
        import torch
        from torch import nn

        class Noisy(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(4, 2, bias=False)

            def forward(self, x):
                return self.lin(x) + torch.randn(1, 2)

        torch.manual_seed(0)
        receipt = ex.export_module(
            Noisy(), _example(), self._out("noisy.torchjit"),
            method="script", name="noisy", task="probe")
        self.assertEqual(receipt.verification.max_abs_deviation, 0.0)

    def test_trace_bakes_the_branch_verification_catches_it(self):
        import torch
        from torch import nn

        class Branched(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = nn.Linear(4, 4, bias=False)
                self.b = nn.Linear(4, 4, bias=False)

            def forward(self, x):
                if x.sum() > 0:
                    return self.a(x)
                return self.b(x)

        torch.manual_seed(0)
        branched = Branched()
        negative = {"x": torch.tensor([[-1.0, -2.0, -3.0, -4.0]])}
        # Script keeps the branch: the other side verifies at 0.0.
        receipt = ex.export_module(
            branched, _example(), self._out("br-script.torchjit"),
            method="script", name="br", task="probe",
            probe_inputs=negative)
        self.assertEqual(receipt.verification.max_abs_deviation, 0.0)
        # Trace fixes the branch the example took: the probe on the
        # other side deviates, and the refusal names the number.
        torch.manual_seed(0)
        with self.assertRaises(ex.ExportError) as ctx:
            ex.export_module(
                Branched(), _example(), self._out("br-trace.torchjit"),
                method="trace", name="br", task="probe",
                probe_inputs=negative)
        self.assertIn("deviates", str(ctx.exception))
        self.assertIn("tolerance", str(ctx.exception))

    def test_long_probe_script_enforced_trace_advisory(self):
        import torch
        from torch import nn

        class Flatten(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(4, 2, bias=False)

            def forward(self, x):
                return self.lin(x.view(-1))

        long_in = {"x": torch.randn(8, 4)}
        receipt = ex.export_module(
            _tiny(), _example(), self._out("long-script.torchjit"),
            method="script", name="tiny", task="probe",
            long_probe_inputs=long_in)
        self.assertTrue(receipt.verification.long_probe["ok"])
        # The flatten module's traced graph only ever saw 4
        # elements: the long probe fails, and for a trace that is
        # recorded — not fatal.
        torch.manual_seed(0)
        receipt = ex.export_module(
            Flatten(), _example(), self._out("long-trace.torchjit"),
            method="trace", name="flat", task="probe",
            long_probe_inputs=long_in)
        self.assertFalse(receipt.verification.long_probe["ok"])
        self.assertIn("shape", receipt.verification.long_probe["error"])

    def test_device_handling(self):
        self.assertEqual(ex.resolve_device("auto"), "cpu")
        self.assertEqual(ex.resolve_device("cpu"), "cpu")
        with self.assertRaises(ex.ExportError) as ctx:
            ex.resolve_device("cuda")
        self.assertIn("no CUDA", str(ctx.exception))
        with self.assertRaises(ex.ExportError) as ctx:
            ex.resolve_device("tpu")
        self.assertIn("unknown export device", str(ctx.exception))

    def test_bad_arguments(self):
        bad = [
            (dict(method="onnx"), "unknown export method"),
            (dict(version="1.0"), "bad version"),
            (dict(task=""), "export needs a task"),
            (dict(name=""), "export needs a service name"),
            (dict(example_inputs={}), "example inputs"),
            (dict(example_inputs={"x": object()}),
             "is not a tensor"),
            (dict(out_path=os.path.join(self.tmp, "x.bin")),
             "must end .torchjit or .pt"),
        ]
        for kwargs, fragment in bad:
            call = dict(method="script", name="tiny", task="probe",
                        example_inputs=_example(),
                        out_path=self._out("bad.torchjit"))
            call.update(kwargs)
            module = _tiny()
            inputs = call.pop("example_inputs")
            out = call.pop("out_path")
            with self.assertRaises(ex.ExportError) as ctx:
                ex.export_module(module, inputs, out, **call)
            self.assertIn(fragment, str(ctx.exception), kwargs)

    def test_load_refusals_name_the_cause(self):
        out = self._out("tiny.torchjit")
        ex.export_module(_tiny(), _example(), out, method="script",
                         name="tiny", task="probe")
        receipt_path = ex.receipt_path_for(out)
        # Missing receipt.
        os.remove(receipt_path)
        with self.assertRaises(ex.ExportError) as ctx:
            ex.load_exported(out)
        self.assertIn("no export receipt", str(ctx.exception))
        ex.export_module(_tiny(), _example(), out, method="script",
                         name="tiny", task="probe")
        # Bad stamp.
        with open(receipt_path, encoding="utf-8") as fh:
            raw = json.load(fh)
        raw["format_version"] = "something.else/9.9"
        with open(receipt_path, "w", encoding="utf-8") as fh:
            json.dump(raw, fh)
        with self.assertRaises(ex.ExportError) as ctx:
            ex.load_exported(out)
        self.assertIn("format stamp", str(ctx.exception))
        ex.export_module(_tiny(), _example(), out, method="script",
                         name="tiny", task="probe")
        # sha mismatch (a byte flipped in the artifact).
        with open(out, "r+b") as fh:
            fh.seek(-10, os.SEEK_END)
            fh.write(b"\x00")
        with self.assertRaises(ex.ExportError) as ctx:
            ex.load_exported(out)
        self.assertIn("sha256 mismatch", str(ctx.exception))
        # Missing file.
        with self.assertRaises(ex.ExportError) as ctx:
            ex.load_exported(self._out("nope.torchjit"))
        self.assertIn("missing", str(ctx.exception))
        # Bad entry.
        ex.export_module(_tiny(), _example(), out, method="script",
                         name="tiny", task="probe")
        with self.assertRaises(ex.ExportError) as ctx:
            ex.load_exported(out, entry="nope")
        self.assertIn("no callable entry", str(ctx.exception))
        # Torch version drift is a warning, not a refusal.
        with open(receipt_path, encoding="utf-8") as fh:
            raw = json.load(fh)
        raw["torch_version"] = "0.0.0-drift"
        with open(receipt_path, "w", encoding="utf-8") as fh:
            json.dump(raw, fh)
        loaded = ex.load_exported(out)
        self.assertIsNotNone(loaded.torch_version_warning)
        self.assertIn("0.0.0-drift", loaded.torch_version_warning)


@torch_only
class CliTests(unittest.TestCase):
    def test_voice_service_export_probe_parses(self):
        from nomorals.cmdline.parser import _parser
        args = _parser().parse_args(
            ["voice", "service", "export-vits-probe",
             "--method", "trace", "--device", "cpu"])
        self.assertEqual(args.voice_service_action, "export-vits-probe")
        self.assertEqual(args.method, "trace")
        self.assertEqual(args.device, "cpu")
        args = _parser().parse_args(
            ["voice", "service", "export-vits-probe"])
        self.assertEqual(args.method, "trace")
        self.assertEqual(args.device, "auto")


if __name__ == "__main__":
    unittest.main()
