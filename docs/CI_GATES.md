# CI gates (Echelon Wave 8)

Two checks that used to be advisory are gates now. Both live in
`nomorals.tools` — CI, the pre-commit hooks, and the tool registry all
call the same modules; there is one implementation of each gate.

**The one local gate command** (run before pushing; mirrors CI):

```sh
make gates
# = python3 -m nomorals.tools.ruff_ratchet
#   python3 -m nomorals.tools.secret_scan
```

Optional: `pre-commit install` once per clone (`.pre-commit-config.yaml`
wires both gates as local hooks).

## 1. The ruff ratchet — lint debt only ever decreases

The repo carries a large standing ruff debt, so a hard `ruff check`
gate would be red forever and teach everyone to ignore CI. Instead,
`ruff-ratchet.json` (repo root) records the current findings **per rule
code**, measured with the ruff version pinned in
`.github/workflows/ci.yml`. The gate (`nomorals.tools.ruff_ratchet`)
fails when:

- any code's live count rises **above** its recorded ceiling (new debt);
- a rule code appears that the ratchet does not record (new debt class);
- a recorded ceiling is **higher** than the live count — debt was paid
  down but the ratchet was not tightened. The record must only ever
  shrink, in the same idiom as the layering test's grandfathered list.

Counts are only comparable for one toolchain (newer ruff adds rules),
so the ratchet records its ruff version; a run under a different
version reports `toolchain_drift` instead of a verdict. If ruff is not
installed the check reports `ruff_installed: false` — an explicit
"cannot measure", never a silent pass.

**Paying debt down:** fix findings, then re-record and commit the
tightened ratchet with the fixes:

```sh
python3 -m nomorals.tools.ruff_ratchet --update
```

**Bumping the pinned ruff:** install the new version, run the ratchet —
it will report drift; re-measure deliberately (`--update` refuses under
drift against an existing record: edit the pin and delete/replace the
record in the same commit), and expect counts to move because the rule
set changed, not because the code did.

## 2. The committed-secret gate — and the disclosure playbook

`nomorals.tools.secret_scan` runs the repo's one secret-pattern battery
(`nomorals.security.audit_patterns` — the same patterns the security
auditor and the privacy scanner use) over every tracked file and exits
non-zero on any hit not in `.secret-scan-allowlist.json`.

An allowlist entry excuses **one exact hit** — path + pattern + SHA-256
of the matched value — with a written reason (documentation example,
test fixture, planted canary, format marker). A different value at the
same path still fails. Gate reports carry only the battery's redacted
form of a value, never the value itself.

**If the gate finds a REAL secret, allowlisting is not the response.**
The playbook (from the echelon post-mortem: their repos carried
production secrets for years because the scan was a warning):

1. **Rotate first.** Revoke/reissue the credential at its provider.
   Rotation outranks cleanup — history rewriting does not un-leak a
   value that was ever pushed.
2. **Purge.** Remove the value from the tree; if it reached a shared
   remote, purge it from history (e.g. `git filter-repo`) and
   force-push, knowing step 1 is what actually protects you.
3. **Publish the explanation.** A short note in the repo (and to anyone
   with repo access): what the credential was, when it was exposed,
   what was rotated, and when. Quiet fixes are how the next leak hides.
4. Only then, if the *shape* must remain in the tree (a fixture, an
   example), make it a non-secret — a documentation example value, or a
   probe assembled at runtime from concatenated parts. Test fixtures
   never carry full-format secret literals.

## 3. Landability scoring (advisory tool, comparable data)

`nomorals.tools.landability` scores a diff (`--base/--head`, or
`--patch-file`): 100 minus named deductions (no test file in the diff,
silent-failure shapes in added lines, oversized diff), with **hard
blocks** for touching the frozen `nomorals/storage/migrations.py`, the
owner-owned `nomorals/memory/persona.py`, or adding secret-shaped
lines. Reviewer annotations are echoed in the report and never change
the score — the tool measures the diff, not the commentary.

## 4. Verify-then-install

`UpdateManager` verifies what a pull actually landed before the update
proceeds (`nomorals/os/verify_install.py`): HEAD equals the claimed
sha, the landed sha descends from the previous one, every object of
the landed tree is present, every changed file on disk hashes to its
HEAD blob, and no new blob exceeds 64 MiB. A failed verification undoes
the pull and aborts the update — a transport's success line is never
taken as proof of the resulting state. The verifier follows the
health-gate idiom: the production construction gets it by default;
scripted constructions (injected health gates, stubbed git seams) get
one only when they pass it explicitly.
