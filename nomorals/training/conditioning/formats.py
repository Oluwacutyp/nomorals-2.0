"""Captioned-video dataset formats — a faithful port of upstream ``dataset.py``'s loading half.

Upstream ``ImageOrVideoDataset`` (finetrainers/dataset.py) supports four
ways of naming (caption, video) pairs; this module ports exactly those
four, with the same validation and the same error wording:

* **two-file** — ``--caption_column`` / ``--video_column`` name two
  line-separated files inside ``--data_root`` (prompts + relative video
  paths).
* **CSV** — ``--dataset_file`` is a ``.csv``; the two columns name the
  caption and video columns. Upstream reads it with ``pandas.read_csv``;
  this port reads it with the stdlib :mod:`csv` module instead — the
  only behavioural difference is that a row with a missing value raises
  a named :class:`ConditioningError` here, where upstream would crash
  inside pandas with an ``AttributeError`` on ``NaN.strip()``.
* **JSON** — ``--dataset_file`` is a ``.json`` list of dicts.
* **JSONL** — ``--dataset_file`` is a ``.jsonl`` file, one dict per line
  (blank lines are skipped; upstream would crash on them).

What is deliberately NOT ported here: the video/image tensor
preprocessing (``decord`` + ``torchvision``), which needs a GPU stack
this box does not have. Frame extraction and resizing live in
:mod:`nomorals.training.conditioning.buckets` as pure shape logic; the
actual pixel work is done by the encoder callables the caller injects
into :mod:`nomorals.training.conditioning.precompute`.

Also ported here: the two prompt cleanups upstream applies at load
time — ``remove_common_llm_caption_prefixes`` (strip
:data:`COMMON_LLM_START_PHRASES`) and the ``id_token`` prefix that
LoRA training appends to every prompt.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

from ...core.logging_setup import get_logger
from .constants import COMMON_LLM_START_PHRASES
from .errors import ConditioningError

__all__ = [
    "CaptionedVideos",
    "apply_id_token",
    "clean_llm_prefixes",
    "load_dataset",
]

_log = get_logger(__name__)

#: Upstream's exact error text for a dataset file with an unsupported
#: suffix (finetrainers/dataset.py, ``ImageOrVideoDataset.__init__``).
_UNSUPPORTED_DATASET_FILE = (
    "Expected `--dataset_file` to be a path to a CSV file or a directory "
    "containing line-separated text prompts and video paths."
)


@dataclass
class CaptionedVideos:
    """The loaded pairs: ``prompts[i]`` captions ``video_paths[i]``.

    ``source_format`` is one of ``"two-file"``, ``"csv"``, ``"json"``,
    ``"jsonl"`` — the four formats upstream documents in
    ``docs/dataset/README.md``. Paths are absolute (resolved under
    ``data_root``); every one of them is a real file, checked at load
    time, never at first use.
    """

    prompts: list[str]
    video_paths: list[Path]
    source_format: str
    data_root: Path = field(default_factory=lambda: Path("."))
    caption_column: str = ""
    video_column: str = ""

    def __len__(self) -> int:
        return len(self.video_paths)

    def prompt_for(self, index: int, *, id_token: str | None = None) -> str:
        """The prompt for ``index``, with ``id_token`` prepended if given."""
        return apply_id_token(self.prompts[index], id_token)


def clean_llm_prefixes(prompt: str) -> str:
    """Strip upstream's LLM-caption opening phrases from one prompt.

    Verbatim port of the ``remove_llm_prefixes`` loop in
    ``ImageOrVideoDataset.__init__``: strip, then for each phrase in
    :data:`COMMON_LLM_START_PHRASES` in order, remove it if the prompt
    starts with it and strip again.
    """
    text = prompt.strip()
    for phrase in COMMON_LLM_START_PHRASES:
        if text.startswith(phrase):
            text = text.removeprefix(phrase).strip()
    return text


def apply_id_token(prompt: str, id_token: str | None) -> str:
    """Prepend the LoRA identifier token exactly as upstream does.

    Upstream stores ``f"{id_token.strip()} "`` and concatenates it in
    ``__getitem__``; ``None`` (or empty) means no prefix at all.
    """
    if not id_token:
        return prompt
    return f"{id_token.strip()} {prompt}"


def _check_counts(prompts: list[str], video_paths: list[Path]) -> None:
    if len(video_paths) != len(prompts):
        raise ConditioningError(
            f"Expected length of prompts and videos to be the same but found "
            f"len(self.prompts)={len(prompts)} and "
            f"len(self.video_paths)={len(video_paths)}. Please ensure that "
            f"the number of caption prompts and videos match in your dataset."
        )


def _check_video_files(
    video_paths: list[Path], *, video_column: str, data_root: Path
) -> None:
    if any(not path.is_file() for path in video_paths):
        raise ConditioningError(
            f"Expected `self.video_column='{video_column}'` to be a path to "
            f"a file in `self.data_root={data_root!r}` containing "
            f"line-separated paths to video data but found atleast one path "
            f"that is not a valid file."
        )


def _load_two_file(
    data_root: Path, caption_column: str, video_column: str
) -> tuple[list[str], list[Path]]:
    """Two-file format: line-separated prompts + line-separated paths."""
    if not data_root.exists():
        raise ConditioningError("Root folder for videos does not exist")
    prompt_path = data_root / caption_column
    video_path = data_root / video_column
    if not prompt_path.exists() or not prompt_path.is_file():
        raise ConditioningError(
            "Expected `--caption_column` to be path to a file in "
            "`--data_root` containing line-separated text prompts."
        )
    if not video_path.exists() or not video_path.is_file():
        raise ConditioningError(
            "Expected `--video_column` to be path to a file in "
            "`--data_root` containing line-separated paths to video data "
            "in the same directory."
        )
    with open(prompt_path, encoding="utf-8") as handle:
        prompts = [line.strip() for line in handle if line.strip()]
    with open(video_path, encoding="utf-8") as handle:
        video_paths = [
            data_root / line.strip() for line in handle if line.strip()
        ]
    return prompts, video_paths


def _load_csv(
    data_root: Path, dataset_file: Path, caption_column: str, video_column: str
) -> tuple[list[str], list[Path]]:
    """CSV format: one row per pair, columns named by the caller."""
    with open(dataset_file, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ConditioningError(
                f"CSV dataset {dataset_file} has no header row."
            )
        for column in (caption_column, video_column):
            if column not in reader.fieldnames:
                raise ConditioningError(
                    f"CSV dataset {dataset_file} has no column "
                    f"'{column}' (columns: {', '.join(reader.fieldnames)})."
                )
        prompts: list[str] = []
        video_paths: list[Path] = []
        for row_number, row in enumerate(reader, start=2):
            caption = row.get(caption_column)
            video = row.get(video_column)
            if caption is None or not str(caption).strip():
                raise ConditioningError(
                    f"CSV dataset {dataset_file} row {row_number} has no "
                    f"'{caption_column}' value."
                )
            if video is None or not str(video).strip():
                raise ConditioningError(
                    f"CSV dataset {dataset_file} row {row_number} has no "
                    f"'{video_column}' value."
                )
            prompts.append(str(caption))
            video_paths.append(data_root / str(video).strip())
    return prompts, video_paths


def _load_json(
    data_root: Path, dataset_file: Path, caption_column: str, video_column: str
) -> tuple[list[str], list[Path]]:
    """JSON format: a list of dicts, one pair per entry."""
    with open(dataset_file, encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ConditioningError(
            f"JSON dataset {dataset_file} must hold a list of entries, "
            f"got {type(data).__name__}."
        )
    prompts = [str(entry[caption_column]) for entry in data]
    video_paths = [
        data_root / str(entry[video_column]).strip() for entry in data
    ]
    return prompts, video_paths


def _load_jsonl(
    data_root: Path, dataset_file: Path, caption_column: str, video_column: str
) -> tuple[list[str], list[Path]]:
    """JSONL format: one dict per line (blank lines skipped)."""
    prompts: list[str] = []
    video_paths: list[Path] = []
    with open(dataset_file, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ConditioningError(
                    f"JSONL dataset {dataset_file} line {line_number} is "
                    f"not valid JSON: {exc}."
                ) from exc
            if not isinstance(entry, dict):
                raise ConditioningError(
                    f"JSONL dataset {dataset_file} line {line_number} must "
                    f"hold an object, got {type(entry).__name__}."
                )
            prompts.append(str(entry[caption_column]))
            video_paths.append(
                data_root / str(entry[video_column]).strip()
            )
    return prompts, video_paths


def load_dataset(
    data_root: str | Path,
    caption_column: str,
    video_column: str,
    *,
    dataset_file: str | Path | None = None,
    id_token: str | None = None,
    remove_llm_prefixes: bool = False,
) -> CaptionedVideos:
    """Load (caption, video) pairs in one of upstream's four formats.

    Mirrors ``ImageOrVideoDataset.__init__``'s dispatch: no
    ``dataset_file`` means the two-file format; ``.csv`` / ``.json`` /
    ``.jsonl`` select the others; anything else raises upstream's
    unsupported-file error. ``id_token`` is stored for
    :meth:`CaptionedVideos.prompt_for`, not baked into the prompts;
    ``remove_llm_prefixes`` strips the LLM opening phrases at load time.
    """
    root = Path(data_root)
    if dataset_file is None:
        prompts, video_paths = _load_two_file(root, caption_column, video_column)
        source_format = "two-file"
    else:
        dataset_path = Path(dataset_file)
        suffix = dataset_path.suffix.lower()
        if suffix == ".csv":
            prompts, video_paths = _load_csv(
                root, dataset_path, caption_column, video_column
            )
            source_format = "csv"
        elif suffix == ".json":
            prompts, video_paths = _load_json(
                root, dataset_path, caption_column, video_column
            )
            source_format = "json"
        elif suffix == ".jsonl":
            prompts, video_paths = _load_jsonl(
                root, dataset_path, caption_column, video_column
            )
            source_format = "jsonl"
        else:
            raise ConditioningError(_UNSUPPORTED_DATASET_FILE)
    _check_video_files(video_paths, video_column=video_column, data_root=root)
    _check_counts(prompts, video_paths)
    if remove_llm_prefixes:
        prompts = [clean_llm_prefixes(p) for p in prompts]
    _log.info(
        "loaded %d captioned videos (%s) from %s",
        len(video_paths), source_format, root,
    )
    return CaptionedVideos(
        prompts=prompts,
        video_paths=video_paths,
        source_format=source_format,
        data_root=root,
        caption_column=caption_column,
        video_column=video_column,
    )
