# Third-party notices — nomorals/skills/library/bughunter/

Everything under this directory is vendored from upstream projects,
not written by Devon. Each file's SHA-256 is pinned in
`nomorals/security/bughunter/bughunter-attribution.toml` at upstream
commit `0e70a02`.

## Claude-BugHunter — elementalsouls/Claude-BugHunter v2.1.0

- Source: https://github.com/elementalsouls/Claude-BugHunter
- Copyright (c) 2026 Sachin Sharma
- Code (`*.py`, `*.sh`): **MIT License** — see `LICENSE` in this
  directory. Permission is hereby granted, free of charge, to any
  person obtaining a copy of this software and associated
  documentation files (the "Software"), to deal in the Software
  without restriction, including without limitation the rights to
  use, copy, modify, merge, publish, distribute, sublicense, and/or
  sell copies of the Software, subject to including the copyright
  notice and this permission notice in all copies.
- Skill and docs content (`skills/*/SKILL.md`, `commands/*.md`,
  `docs/disclosed-reports/*`, `cbh/data/skill_index.json`):
  **Creative Commons Attribution 4.0 International (CC BY 4.0)** —
  see `LICENSE-CONTENT` in this directory. Reuse is permitted
  including for commercial purposes, with attribution and a link
  back to the original project.
- `NOTICE` in this directory is upstream's own notice file,
  reproduced verbatim.

## claude-bug-bounty — shuvonsec/claude-bug-bounty (MIT)

The 8 foundation skills and 12 commands below were vendored into
Claude-BugHunter from shuvonsec/claude-bug-bounty (MIT) and are
vendored here through that second hop, with the same MIT terms:

- Skills: `bb-methodology`, `bug-bounty`, `triage-validation`,
  `report-writing`, `security-arsenal`, `web2-recon`, `web3-audit`,
  `meme-coin-audit`
- Commands: `/hunt`, `/recon`, `/scope`, `/triage`, `/validate`,
  `/report`, `/autopilot`, `/chain`, `/intel`, `/surface`,
  `/token-scan`, `/web3-audit`

## Counts on disk (auditable units)

- 83 skill packs under `skills/` (58 per-vuln-class `hunt-*`
  skills + enterprise-platform, reporting and methodology skills)
- 36 pattern-library files under `disclosed-reports/` (`hunt-*.md`)
  plus a `README.md` = 37 files in the directory. The bundled
  `skill_index.json` lists exactly those 36 names — the earlier
  "37 pattern files" count was the directory listing, not the
  pattern count, and there is no index/disk discrepancy.
- 15 slash commands under `commands/`
- Upstream's own counts disagree with each other (README 681,
  credits 574+, SKILL.md frontmatter `report_count:` sums to 1,122
  across 70 skills); none of those numbers is reproduced as fact
  here — the auditable unit is the 36 pattern files on disk.
