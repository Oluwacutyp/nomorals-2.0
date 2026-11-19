# finetrainers-conditioning fixtures

Real sample data vendored from the upstream FineTrainers snapshot
(storytold/FineTrainers-Conditioning, commit 9f0fdb5, Apache-2.0),
`assets/tests/` — the same files upstream's own `tests/test_dataset.py`
loads. Nothing here is synthetic except the two derived files.

- `metadata.csv` — CSV-format dataset: columns `video,caption`, 1 row.
- `prompts.txt` / `videos.txt` — two-file format: 1 caption, 1 video.
- `prompts_multi.txt` / `videos_multi.txt` — two-file format: 2 rows
  (hiker.mp4 + hiker_tiny.mp4, same caption).
- `videos/hiker.mp4` (265,988 bytes), `videos/hiker_tiny.mp4` (70,973
  bytes) — real H.264 clips of a hiker on a mountain.
- `dataset.json`, `dataset.jsonl` — DERIVED (this lane): the same two
  real rows re-expressed in upstream's JSON / JSONL dataset formats
  (docs/dataset/README.md). Same caption, same video paths, no invented
  content — they exist so the JSON/JSONL loaders are tested against
  real rows, not invented ones.
- `prompts_llm.txt` / `videos_llm.txt` — DERIVED (this lane): one real
  video row whose caption starts with an upstream LLM opening phrase
  ("In this video, ..."), so `--remove_common_llm_caption_prefixes`
  is tested against a real row instead of an invented string.
