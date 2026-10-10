# DATASCI sweep — external mining (2026-10-10)

Module: `nomorals/datasci/` — 4 files: `__init__.py`, `errors.py`, `plots.py`,
`workspace.py`. A named-dataset pandas workspace with provenance + matplotlib
charts. Method: mine best-in-class OUTSIDE implementations first, then build
the gold in.

## Sources mined

1. **ydata-profiling** (`data-centric-ai-community/ydata-profiling`,
   https://github.com/data-centric-ai-community/ydata-profiling) — the
   de-facto automated EDA standard. Mined feature list from its README:
   - *Overview* section: record count, variable count, overall missingness,
     duplicate rows, **memory footprint**.
   - *Alerts*: automatic data-quality warnings — high correlation, skewness,
     uniformity, zeros, missing values, **constant values**.
   - *Compare datasets*: one-line report comparing two datasets.
   - Univariate analysis: descriptive stats (mean/median/mode) + per-column
     type inference (categorical / numerical / date).
   → Gold taken: `validate()` quality report with ydata-style alerts
   (CONSTANT, HIGH_CARDINALITY, SKEWED, HIGH_MISSING, ZEROS, HIGH_CORRELATION,
   DUPLICATES); `describe()` gains an `overview` block (memory, dup rows,
   missing cells/pct) + Pearson `correlations` + `alerts`; new `compare(a, b)`.

2. **matplotlib plot-types gallery**
   (https://matplotlib.org/3.8.3/plot_types/index.html) — canonical taxonomy:
   pairwise (plot, scatter, bar, barh, stem, stackplot), statistical
   distributions (hist, **boxplot, violinplot**, ecdf, pie), gridded
   (imshow/pcolormesh — the basis of a correlation heatmap).
   → Gold taken: new kinds `box`, `violin`, `barh`, `heatmap` (annotated
   correlation via imshow), `kde`, `area`, `pie`. Previously only
   line/bar/scatter/hist existed.

3. **paper-figure-codegen** (`yuyuan12138/paper-figure-codegen`,
   https://github.com/yuyuan12138/paper-figure-codegen) — 17 publication
   plot types; the useful patterns for us: **scatter+regression**
   (scatter with line of best fit), histogram+KDE, heatmap for correlation,
   horizontal bar for rankings, paper-style low-saturation palettes +
   per-figure export (PNG with dpi control).
   → Gold taken: `scatter(..., trend=True)` best-fit line via
   `numpy.polyfit`; `kde` via `scipy.stats.gaussian_kde`; `bins`, `dpi`,
   `figsize`, `grid` knobs on `render_plot`; capped bar categories (top-N)
   with rotated tick labels so charts stay readable.

4. **pandas IO docs / community guides**
   (e.g. https://github.com/sujalkarbhari1/pythonprogramming/blob/HEAD/Numpy_Pandas/14_pandas_reading_writing_data.md,
   https://www.plus2net.com/python/pandas-to_string.php):
   - `pd.read_excel(..., engine="openpyxl")` for `.xlsx`; `sheet_name=None`
     reads all sheets.
   - SQLite round-trip pattern: `df.to_sql("t", sqlite3.connect(":memory:"))`
     then `pd.read_sql_query("SELECT ...")` — stdlib-only SQL over a
     DataFrame (no DuckDB dependency needed).
   - Parquet (`to_parquet`/`read_parquet`, needs pyarrow): preserves exact
     dtypes, columnar, compressed — the production persistence format; CSV
     loses dtypes (the module's current CSV+sidecar workaround proves it).
   → Gold taken: `.xlsx`/`.xls` loading (openpyxl is installed);
   `sql()` running SELECT/WITH statements via in-memory sqlite3 (validated
   to read-only statements); `export()` to csv/json/jsonl/xlsx; persistence
   upgraded to **parquet-when-available with CSV+sidecar fallback** (the
   class docstring already promised parquet — now it's true); restore reads
   both formats.

5. **pyjanitor** (known from the ecosystem; cleaning-verb conventions:
   `remove_empty`, `coalesce`, type coercion helpers) and the Great
   Expectations expectation vocabulary (expect_column_values_to_not_be_null,
   expect_column_values_to_be_unique) — data-cleaning + data-quality
   vocabulary.
   → Gold taken: `clean()` (drop duplicates, drop-na modes, per-column
   coercion to numeric/datetime/str) with `as_name` registration;
   `validate()` per-column missing %/unique counts/zero %/outlier counts.

6. **Datasette** (datasette.io) — explore-by-facets UX: list → head/tail →
   value-counts → filter. A data workspace should answer "what's in column
   X" in one call.
   → Gold taken: `tail()`, `sample()`, `value_counts()`, `aggregate()`
   (groupby), `merge()` (two datasets), `rename()`.

7. **Seaborn / plotly.express guidance**
   (https://github.com/anwai98/prompt-to-plot/blob/HEAD/docs/plotting_libraries.md):
   seaborn for static group/distribution work, dark themes for dashboards.
   → Gold taken: composable `THEMES` presets on `render_plot`
   (`light`/`dark`/`ninja`/`minimal`) — styling lives in presets, never
   hardcoded into the render path (matches the user's style-agnostic
   architecture rule). The `ninja` theme uses electric-blue accents matching
   the user's Termux theme.

## Gaps found in current implementation (what the classes SHOULD have)

- `describe()`: only raw pandas `describe()` — no memory, no duplicates, no
  missing %, no correlations, no alerts, no skew. (Fixed: overview +
  correlations + alerts.)
- No SQL access at all — pandas `query()` only. (Fixed: `sql()`.)
- No Excel support despite openpyxl installed. (Fixed.)
- No export path — data goes in, never out except PNG. (Fixed: `export()`.)
- No data-quality story. (Fixed: `validate()` + `clean()`.)
- No dataset algebra: no merge, no groupby, no rename, no compare.
  (Fixed: `merge`, `aggregate`, `rename`, `compare`.)
- `plots.py`: 4 kinds, zero styling, unreadable bars past ~10 categories,
  no size/dpi control. (Fixed: 11 kinds, themes, caps, knobs.)
- Docstring claimed parquet persistence; implementation used CSV.
  (Fixed: parquet when pyarrow present, CSV fallback; restore both.)

## What was NOT taken (and why)

- ydata-profiling's full HTML report — out of scope; this module emits
  JSON-safe dicts + PNG bytes for the agent/CLI, not a notebook widget.
- DuckDB — not installed; in-memory sqlite3 covers the SELECT use-case
  with zero new deps.
- IsolationForest / statsmodels outlier models — scipy/sklearn gaps;
  IQR-based outlier counts are dependency-free and explainable.
- plotly interactive charts — matplotlib/Agg PNG stays the artifact
  contract (no display, chat-friendly).
