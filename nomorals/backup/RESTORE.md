# Restore — from zero (template)

This is the from-zero restore recipe for a `nomorals-backup` generation.
Each Release carries its own copy, rendered from that generation's
`manifest.json` (see `nomorals.backup.manifest.render_restore_recipe`).

## You need

1. The backup key: vault `github-backup` / `backup-key`, or the offline
   copy (paper / password manager).
   **Lose the key, lose the backup. There is no reset.**
2. This repo (manifests) + the Release assets for the generation.
3. A clean machine with Python + this codebase.

## Steps

1. Download every `*.gz.enc` part from the Release.
2. Reassemble split parts in filename order (`.part00of03`, …).
3. SHA-256 the reassembled ciphertext; it MUST equal the `sha256` in
   `manifest.json`. Mismatch = FAILED, do not decrypt.
4. Decrypt with the backup key (AES-256, `nmc1` blob), then gunzip.
5. For each `.db` file: open with sqlite3, run
   `PRAGMA integrity_check` (must be `ok`), confirm the table count
   matches the manifest.
6. Put files back at their `rel` paths relative to the project root.
   Vault files go back encrypted-only — never plaintext.
7. Start Devon; run `nm backup verify` as a second opinion.

## 3-2-1 reminder

- Copy 1: live DB + continuous local replication.
- Copy 2: local / external archive (staging dir + outbox).
- Copy 3: GitHub encrypted (this repo + Releases).
- Optional copy 4: S3-compatible store (B2 / R2 / Wasabi).
