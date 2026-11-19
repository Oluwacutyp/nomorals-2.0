"""Fine-tune preparation: the runnable end-to-end path.

``prepare_run`` takes a config, a model name and a data root and produces
everything training needs: cleaned filelists, context sidecars when the
frontends are available, cached spectrograms, and a prepare report. It
runs with no torch installed; the training step itself (``train.py``)
needs torch and refuses honestly without it.
"""

import json
import os

from . import audio as _audio
from . import data as _data
from . import preprocess as _preprocess
from .configs import load_config


def prepare_run(config=None, config_name=None, model_name="vits_finetune",
                data_root=None, out_dir=None, build_sidecars=False,
                test_dir=None):
    """Prepare a fine-tune run. Returns the report dict (also written out).

    ``data_root`` holds the raw filelists (``train_filelist.txt`` etc.);
    when omitted the config's own ``training_files`` paths are used as-is.
    """
    hps = load_config(config, config_name)
    if out_dir is None:
        out_dir = os.path.join("logs", model_name)
    os.makedirs(out_dir, exist_ok=True)

    filelists = []
    for key in ("training_files", "validation_files"):
        path = getattr(hps.data, key, None)
        if not path:
            continue
        if data_root is not None and not os.path.isabs(path):
            path = os.path.join(data_root, path)
        filelists.append(path)
    for path in filelists:
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"filelist {path} (from config data.{key}) does not exist; "
                "point the config at real filelists first"
            )

    report = {
        "model": model_name,
        "config": config_name or config,
        "out_dir": out_dir,
        "sampling_rate": hps.data.sampling_rate,
        "text_cleaners": list(hps.data.text_cleaners),
        "cleaned": {},
        "sidecars": {},
        "spec_cache": {"built": 0, "cached": 0},
        "test": [],
        "warnings": [],
    }

    prep = _preprocess.preprocess_filelists(
        filelists, list(hps.data.text_cleaners),
        build_sidecars_flag=build_sidecars, test_dir=test_dir)
    report["cleaned"] = prep["cleaned"]
    report["sidecars"] = prep["sidecars"]
    report["test"] = prep["test"]

    # Spectrogram cache for every row of every cleaned filelist.
    for filelist in filelists:
        cleaned = report["cleaned"][filelist]["out"]
        for row in _data.load_filepaths_and_text(cleaned):
            wav = row[0]
            if not os.path.isfile(wav):
                report["warnings"].append(f"missing audio file: {wav}")
                continue
            cache = _audio.spec_cache_path(wav)
            if os.path.exists(cache):
                report["spec_cache"]["cached"] += 1
            else:
                try:
                    _audio.load_or_compute_spec(wav, hps.data)
                    report["spec_cache"]["built"] += 1
                except ValueError as exc:
                    report["warnings"].append(f"{wav}: {exc}")

    # The config as actually used, frozen next to the report.
    from .configs import save_config
    save_config(hps, os.path.join(out_dir, "config.json"))
    report_path = os.path.join(out_dir, "prepare_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    report["report_path"] = report_path
    return report
