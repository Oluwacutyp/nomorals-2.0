# WisdomKeeper

The esoteric study companion and practice organ (L5). Wraps `nomorals/books`
for corpus storage and `nomorals/documents` for parsing — it does not
rebuild either.

## Classes

- **`WisdomKeeper`** (`keeper.py`) — the façade. Owns one of each below.
- **`CanonCorpus`** (`corpus.py`) — curated manifest of texts (versioned JSON),
  ingest with sha256 idempotency, `ask()` with provenance on every hit.
- **`ArchiveIngestor`** (`ingestor.py`) — fetches texts from public libraries
  (archive.org, sacred-texts.com, gnosis.org…), respects robots.txt, caches
  blobs by sha256, actively searches archives for new candidates.
- **`HistoryEngine`** (`history.py`) — 52-event timeline dataset
  (`data/timeline.json`, every entry sourced), `timeline()` / `compare()` /
  `lineage()` queries.
- **`PracticeGuide`** (`practice.py`) — breathing session scripts
  (`sessions/*.json`), injectable-clock pacer, practice log, post-session
  journaling. Every session prints the honest safety framing: breathing aids
  relaxation via the parasympathetic system; no verified link to astral
  projection or kundalini; stop if lightheaded/dizzy; clinician note for
  lung/heart/pregnancy/BP conditions; not medical care.

## CLI (operator)

```
nm wisdom status [--json]
nm wisdom ask <query> [--limit N] [--json]
nm wisdom ingest <slug> | --all
nm wisdom seed
nm wisdom search <query> [--limit N] [--json]
nm wisdom timeline [--tradition T] [--start Y] [--end Y] [--json]
nm wisdom compare <topic> [--json]
nm wisdom practice list [--json]
nm wisdom practice <session-id> [--rounds N]
```

## Corpus seed

`data/manifest.json` ships 47 texts across 14 traditions (canon, apocrypha,
pseudepigrapha, gnostic, dss, eastern, esoteric, secondary). Run
`nm wisdom seed` to register them, `nm wisdom ingest --all` to fetch.
Only public sources; no credentials; robots.txt respected.
