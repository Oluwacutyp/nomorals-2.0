# i18n parity — Echelon Wave 7

The wave in one line: Devon's first i18n system — craft-i18n's
shape (TSV catalogs keyed by English source string, `@plural`
rows, fallback chains, a checker with per-locale completeness)
ported to Python at L3, first consumer the community group
surfaces.

Regenerate the completeness table:
`python -c "from nomorals.i18n import check_catalogs, render_parity; print(render_parity(check_catalogs()))"`

## Completeness (measured, checker run on the shipped catalogs)

| locale | translated | total | completeness |
|---|---|---|---|
| ar | 6 | 6 | 100.0% |
| de | 6 | 6 | 100.0% |
| en | 6 | 6 | 100.0% |
| es | 6 | 6 | 100.0% |
| fr | 6 | 6 | 100.0% |
| ha | 6 | 6 | 100.0% |
| hi | 6 | 6 | 100.0% |
| ig | 6 | 6 | 100.0% |
| it | 6 | 6 | 100.0% |
| ja | 6 | 6 | 100.0% |
| nl | 6 | 6 | 100.0% |
| pcm | 6 | 6 | 100.0% |
| pt | 6 | 6 | 100.0% |
| ru | 6 | 6 | 100.0% |
| sw | 6 | 6 | 100.0% |
| yo | 6 | 6 | 100.0% |

The catalog key set is the wired surface's set (below) — small
on purpose: completeness is 100% *of the community membership
flow*, not of Devon's UI. Every other user-facing string in the
codebase is still hardcoded English; that is the honest state,
and growing the key set surface by surface is the follow-up work,
not a claim made here. Locales with no catalog (e.g. `uk`, `id`)
are absent from the table by design: lookups for them fall back
through the chain to English and the `Translation` record says
so — absence is a recorded fact, not a faked 0%.

## Rows

| piece | state | evidence |
|---|---|---|
| Catalogs as data (`catalogs/*.tsv`, key = English source string, `@plural` rows) | shipped | `i18n/catalog.py`; checker test |
| Fallback chains (`pt-BR → pt → en`) with the served locale recorded on every render | shipped | `i18n/locale.py`, `i18n/runtime.py` (`Translation.served_locale/exact/source`) |
| CLDR plural rules per locale (integer cardinals; ar 6 categories, ru 4, ja 1) | shipped | `i18n/plural.py`; plural tests |
| Checker: row validity, placeholder parity, chain resolution, completeness | shipped | `i18n/check.py`; `tests/test_i18n_check.py` (CI-gate form) |
| First consumer: community group membership flow (`/cgroup join/answer/leave`, `/cgroup lang`) | shipped | `community/groups.py::_group_message`; surface tests render one join in es + ru |
| `family/locales.py` region packs | mapped, not replaced | `i18n/bridge.pack_locales` reads pack languages as i18n locales; packs stay region data |
| `llm/languages.py` code table | mapped, not replaced | `i18n/bridge.mt_language_code` (locale → ladder code, `""` when unknown) |
| `llm/translation_quality.py` as the MT-contribution quality bridge | kept, wired | `i18n/bridge.score_contribution` scores a drafted row by back-translation before it lands in a catalog |
| CLI consumer (`nm i18n …`) | not started — the plan's sequencing is group surfaces first, then CLI | — |

## Named limits and decisions

- **No ICU in the stdlib.** Number/date formatting is plain
  interpolation (`{count}` renders the integer as given, no
  locale grouping, no digit shaping). Recorded, not hidden:
  message selection, plurals and fallback — the parts that
  change meaning — are rule-driven; formatting is the rung a
  future ICU dependency would have to earn.
- **Plural rules are integer cardinals.** Categories that exist
  only for decimal counts in CLDR (Czech/Lithuanian `many`) are
  not listed; non-integral counts land in `other`.
- **`ga` is not given Irish plural rules.** ISO `ga` is Irish,
  but the family packs use `ga` for the Ga language of Ghana;
  until that collision is resolved upstream, `ga` uses the
  visible other-only default rather than the wrong language's
  grammar.
- **Regional catalogs are chain-served, not duplicated.** There
  is no `pt-BR.tsv`: `pt-BR` resolves through `pt`, and the
  render record shows `served_locale="pt"`, `exact=False`.
- Catalog translations for the membership flow are hand-seeded.
  Machine-translated contributions must pass
  `bridge.score_contribution` (back-translation F1 + judge)
  before they are written into a catalog file.
