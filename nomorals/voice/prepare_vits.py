"""Fine-tune preparation CLI: clean filelists, build sidecars, cache specs.

Usage:
    python -m nomorals.voice.prepare_vits --config-name ljs_li44_tmbert_ft_small_s1_arpa.json \\
        --model my_voice --data-root /path/to/data [--build-sidecars] [--test-dir test_preproc]

Runs with no torch installed. The training step itself is
``python -m nomorals.voice.vits.train -m my_voice`` (needs torch).
"""

import argparse
import json
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Prepare a VITS fine-tune run (clean filelists, sidecars, spec cache)")
    parser.add_argument("-c", "--config", type=str, default=None,
                        help="config JSON path (overrides --config-name)")
    parser.add_argument("--config-name", type=str,
                        default="ljs_li44_tmbert_ft_small_s1_arpa.json",
                        help="vendored config name under nomorals/voice/vits/data/configs/")
    parser.add_argument("-m", "--model", type=str, required=True,
                        help="model/run name (output goes to logs/<model>)")
    parser.add_argument("--data-root", type=str, default=None,
                        help="directory holding the config's filelists")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="output directory (default logs/<model>)")
    parser.add_argument("--build-sidecars", action="store_true",
                        help="build .torchmoji/.bert sidecars (needs TorchMoji + TinyBERT)")
    parser.add_argument("--test-dir", type=str, default=None,
                        help="write test_preproc entries here")
    args = parser.parse_args(argv)

    from nomorals.voice.vits.prepare import prepare_run
    report = prepare_run(
        config=args.config, config_name=args.config_name,
        model_name=args.model, data_root=args.data_root,
        out_dir=args.out_dir, build_sidecars=args.build_sidecars,
        test_dir=args.test_dir)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
