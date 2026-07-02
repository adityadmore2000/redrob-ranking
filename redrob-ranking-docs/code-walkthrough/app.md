# `app.py` — Deep Line-by-Line Teaching Notes

> Streamlit UI layer for the Redrob candidate ranking demo. This file is a **thin
> presentation/orchestration layer**; all ML/ranking logic lives in `demo_pipeline.py`
> (imported as `dp`).

**The one mental model to hold onto:** `streamlit run app.py` re-executes this entire
script top-to-bottom on **every** user interaction. Almost every design choice below
(`@st.cache_resource`, `st.session_state`, the transient button return value) exists to
cope with that "re-run everything" model.

---

## Part 1 — Module docstring

```python
"""
app.py — Streamlit demo for the Redrob candidate ranking system.
...
Run locally with:  streamlit run app.py
"""
```

- **What:** module-level docstring; first statement in the file, stored as `__doc__`.
- **Why:** orients a reader (what the app does + how to run it) and documents the
  architecture: Track 1 (structured signals), Track 2 (semantic embedding match),
  Phase 2 (multiplicative final score + reasoning). The "multiplicative" detail is the
  single most important design decision in the system.
- **Internals:** evaluated then bound to `__doc__`; negligible cost.
- **Interview:** docstring (real runtime string via `__doc__`, seen by `help()`) vs
  comment (stripped by tokenizer). Must be the *first* statement to count as `__doc__`.
- **Hidden:** `streamlit run app.py` ≠ `python app.py` — Streamlit runs the script
  through its own runner and re-runs it on each interaction.

---

## Part 2 — Imports

```python
import json
import os

import pandas as pd
import streamlit as st

import demo_pipeline as dp
from demo_pipeline import InputError, MAX_CANDIDATES
```

- **What:** three PEP 8 import groups — stdlib, third-party, local.
- `json` — stdlib JSON. **Imported but not actually used in this file** (parsing happens
  in `demo_pipeline`); a linter would flag `F401`.
- `os` — used to build a CWD-independent path to the sample data.
- `pandas as pd` — data-analysis library; central object is the **DataFrame** (labeled
  2-D table). `pd` is the universal alias.
- `streamlit as st` — framework that turns a Python script into an interactive web app
  with no HTML/JS. `st` is the universal alias.
- `demo_pipeline as dp` — local module holding all real logic. UI (`app.py`) and business
  logic (`dp`) are **decoupled** — deliberate, testable architecture.
- `from demo_pipeline import InputError, MAX_CANDIDATES`:
  - `InputError` — custom exception distinguishing *expected bad user input* from
    *unexpected bugs*; drives the two-tier `except` handling in `run_ranking`.
  - `MAX_CANDIDATES` — the ≤100 cap. Imported (not hard-coded) so UI text is a **single
    source of truth** with the pipeline.
- **Interview:** why aliases (readability/convention); why import the constant (DRY); what
  the import grouping means (PEP 8 ordering).

---

## Part 3 — Path and constants

```python
_HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE_PATH = os.path.join(_HERE, 'data', 'sample_candidates.json')
```

- **What:** absolute path to bundled sample data, anchored to *this file's* location.
- **Evaluation (inside-out):** `__file__` (this script's path) → `abspath` (absolute) →
  `dirname` (its directory = `_HERE`) → `join(...)` (correct OS separators).
- **Why:** a relative `open('data/...')` would resolve against the **CWD**, which is
  unpredictable (esp. on Hugging Face Spaces). Anchoring to `__file__` is CWD-robust.
- **Interview:** why not a relative path (CWD dependence); why `os.path.join` (cross-platform
  separators); what `__file__` is. Modern alternative: `pathlib.Path(__file__).resolve().parent`.
- Leading underscore in `_HERE` = "module-private."

```python
SCORE_COLS = [
    'final_score', 'semantic_score_norm', 'hard_filter_score',
    'availability_score', 'credibility_score',
]
```

- **What:** constant list of the 5 columns rendered as **progress bars** later.
- **Why:** single editable list; consumed by the loop at the results section. Order = bar
  order (`final_score` first = headline). Implicitly a **schema contract** with the
  DataFrame `dp.rank_candidates` returns.

---

## Part 4 — Page configuration

```python
st.set_page_config(
    page_title='Candidate Ranking Demo',
    page_icon='🏆',
    layout='wide',
)
```

- **What:** page-level settings (browser tab title, favicon, content width).
- **Why here:** **must be the first Streamlit command** or Streamlit raises
  `StreamlitAPIException`. That's why it sits above all rendering.
- **Args:**
  - `page_title` — browser tab text (default is generic app name).
  - `page_icon='🏆'` — favicon; emoji = zero asset files; matches ranking theme.
  - `layout='wide'` — content spans full width. Default is `'centered'` (narrow). `'wide'`
    chosen because the centerpiece is a wide multi-column data table.
- **Internals:** stores page settings, frontend sets `document.title`, favicon link, and
  container max-width CSS. Returns `None` (side-effect only).
- **Interview:** why it must be first; what layouts exist (`centered`/`wide`); icon can be
  emoji/image/URL.
- **Relationship:** `layout='wide'` affects everything rendered afterward, esp. the big table.

---

## Part 5 — Cached model loader

```python
@st.cache_resource(show_spinner='Loading the embedding model (first run only)…')
def load_model():
    # CPU is the safe default on Hugging Face Spaces free tier.
    return dp.get_model(device='cpu')
```

- **What:** function returning the BGE-base-en-v1.5 embedding model, decorated to cache it.
- **Why:** the script reruns on every interaction; loading a transformer is slow/heavy.
  `@st.cache_resource` runs the body **once** and returns the **same object** on every later
  call (across reruns and users, for the life of the server process).
- **`cache_resource` vs `cache_data` (key interview point):**
  - `cache_data` — serializable data (DataFrames, dicts); returns a **copy** each time.
  - `cache_resource` — global unserializable **shared resources** (models, DB/network
    connections); returns the **same object**. Correct choice for a model.
- **Arg `show_spinner=...`** — custom spinner text while the (uncached) body runs; honestly
  notes this only happens on the first run. Alternatives: `True`/`False`/custom string.
- **Arg `device='cpu'`** — forces CPU; HF Spaces free tier has no GPU. `'cuda'` for GPU.
- **Internals:** decorator wraps the function; body runs lazily on the first `load_model()`
  call (in `run_ranking`), result cached; later calls return instantly.
- **Return:** the loaded model, same instance every time (no copy).
- **Hidden:** cache key = function + args; no args here ⇒ effectively a singleton. Not copied,
  so mutation would be shared (fine for read-only inference).

---

## Part 6 — Sidebar

```python
with st.sidebar:
    st.header('About')
    st.markdown(...)
    st.subheader('Expected input')
    st.markdown(f'... up to **{MAX_CANDIDATES}** candidate objects ...')
    st.code('{ ... }', language='json')
    st.caption('Missing fields fall back to neutral defaults ...')
```

- **`st.sidebar`** — the collapsible left panel, used here as a **context manager**: every
  `st.*` inside the `with` renders into the sidebar (cleaner than `st.sidebar.foo(...)`).
- **Why:** persistent reference material (what it does, expected schema, JSON example) stays
  out of the way of the main action area.
- **Widgets:** `st.header` (h2) → `st.subheader` (h3) size hierarchy; `st.markdown` renders
  rich Markdown (bold, bullets, inline code); adjacent string literals auto-concatenate;
  `\n\n` = paragraph breaks.
- **f-string** `{MAX_CANDIDATES}` — interpolates the imported constant (UI ↔ pipeline in sync).
- **`st.code(..., language='json')`** — monospace highlighted box with copy button; devs
  copy-paste schemas, so a literal example beats prose.
- **`st.caption(...)`** — small muted footnote text.
- **Interview:** two ways to fill the sidebar; difference between text/markdown/code/caption.

---

## Part 7 — Header

```python
st.title('🏆 Candidate Ranking Demo')
st.markdown('Upload a candidate pool or try the sample, ...')
```

- `st.title` — largest heading (h1) in the **main** area (outside the sidebar `with`, so it
  lands in the main flow). `st.markdown` below = one-line subtitle.

---

## Part 8 — Input controls

```python
col_upload, col_sample = st.columns([3, 1])

with col_upload:
    uploaded = st.file_uploader(
        f'Upload candidates (JSON or JSONL, ≤{MAX_CANDIDATES})',
        type=['json', 'jsonl'],
        help='A JSON array of candidate objects, or one object per line (JSONL).',
    )

with col_sample:
    st.write('')  # vertical spacers to align button with uploader
    st.write('')
    use_sample = st.button('🎲 Use Sample Data', use_container_width=True)
```

- **`st.columns([3, 1])`** — two side-by-side columns in a **3:1 width ratio**; returns a
  list of column containers, tuple-unpacked into `col_upload`, `col_sample`. (`st.columns(2)`
  = equal widths.)
- **`st.file_uploader`** — drag/drop file input.
  - label (f-string with the cap); `type=['json','jsonl']` filters extensions (UX guardrail,
    **not** real validation); `help=` tooltip.
  - **Returns `None`** when empty, or an **`UploadedFile`** once selected (`.getvalue()` →
    bytes, `.name` → filename).
- **`st.write('')` ×2** — empty spacers to vertically align the button with the uploader.
- **`st.button(...)`** — returns a **bool that is `True` only on the single rerun caused by
  the click**, `False` otherwise. This transience is *why* results must be saved to
  `session_state`. `use_container_width=True` stretches it to fill the narrow column.
- **Interview:** button return semantics; uploader return before/after; ratio vs equal columns.

---

## Part 9 — Resolve active input source

```python
raw_bytes = None
source_name = None
if use_sample:
    try:
        with open(SAMPLE_PATH, 'rb') as f:
            raw_bytes = f.read()
        source_name = 'sample_candidates.json'
    except OSError as exc:
        st.error(f'Could not load the bundled sample data: {exc}')
elif uploaded is not None:
    raw_bytes = uploaded.getvalue()
    source_name = uploaded.name
```

- **What:** normalizes both input sources into the same two variables (`raw_bytes`,
  `source_name`) so downstream code is source-agnostic.
- Init to `None` — the later `if raw_bytes is not None:` guard depends on it.
- `if use_sample:` — **sample button wins** on its click. Opens sample in **binary mode
  (`'rb'`)** to match the uploaded-file bytes path; `with` auto-closes; `except OSError`
  catches file-not-found/permission and shows a red banner (leaving `raw_bytes` `None`).
- `elif uploaded is not None:` — else, if a file was uploaded, use its bytes + name.
- **`elif`** encodes mutual exclusivity ("sample wins").
- **Interview:** why `'rb'` (pipeline wants bytes; parity with upload); why `OSError`; why
  normalize both sources.

---

## Part 10 — `run_ranking` (orchestration + two-tier errors)

```python
def run_ranking(raw, name):
    """Validate input, run the pipeline with a progress bar, return a DataFrame."""
    try:
        candidates = dp.load_candidates_from_bytes(raw, name)
    except InputError as exc:
        st.error(f'⚠️ {exc}')
        return None
    except Exception as exc:  # noqa: BLE001 — surface anything unexpected cleanly
        st.error(f'⚠️ Could not read the file: {exc}')
        return None

    st.success(f'Loaded **{len(candidates)}** candidates from `{name}`.')
    model = load_model()
    progress_bar = st.progress(0.0, text='Starting…')

    def on_progress(frac, msg):
        progress_bar.progress(min(1.0, frac), text=msg)

    try:
        with st.spinner('Ranking candidates…'):
            df = dp.rank_candidates(candidates, model, progress=on_progress)
    except Exception as exc:  # noqa: BLE001
        progress_bar.empty()
        st.error(
            'Ranking failed while processing this file. The records may not '
            f'match the expected candidate schema.\n\nDetails: `{exc}`'
        )
        return None

    progress_bar.empty()
    return df
```

- **Two-tier exception strategy (most interview-worthy):**
  1. `except InputError` — pipeline's own exception for malformed/oversized input; message is
     already user-friendly. *Expected, recoverable.*
  2. `except Exception` — catch-all for genuine bugs/unexpected formats; generic message; app
     never crashes to a raw traceback. Both `return None`.
- **`# noqa: BLE001`** — documents that the blind `except Exception` is *intentional* in UI
  code (suppresses the "blind except" linter rule).
- `st.success(...)` — green confirmation banner with count + filename.
- `model = load_model()` — first call to the cached loader; lazy (only when there's data).
- **Progress bar + callback:** `st.progress(0.0, ...)` returns a handle; `on_progress` is a
  nested callback passed *into* the pipeline so `dp` can report progress **without knowing
  about Streamlit** (dependency injection / UI-agnostic pipeline). `min(1.0, frac)` guards
  float drift past 1.0.
- `with st.spinner(...)` — animated spinner during the unknown-length ranking step.
- `progress_bar.empty()` — clears the bar on completion or error.
- **Return:** DataFrame on success, `None` on any failure (caller's "don't render" signal).
- **Interview:** why two excepts; why a progress callback (decoupling/testability); what `# noqa` does.

---

## Part 11 — Trigger run + persist to session_state

```python
if raw_bytes is not None:
    df = run_ranking(raw_bytes, source_name)
    if df is not None:
        st.session_state['ranked_df'] = df
        st.session_state['ranked_source'] = source_name
```

- **`st.session_state`** — dict-like store that **persists across reruns** for one user
  session (until the tab/session closes). Normal locals reset every rerun; this doesn't.
- **Why:** the button's `True` and uploader value are per-run. Any later interaction (e.g.
  clicking Download) triggers a rerun where `raw_bytes` is `None` — so results are stashed in
  `session_state` to **persist without recomputation**.
- Guards: run only with input this run; persist only on success.
- **Interview:** why `session_state` exists (rerun model); its scope/lifetime (per session,
  *not* shared across users — unlike `cache_resource`).

---

## Part 12 — Rendering results

```python
df = st.session_state.get('ranked_df')
if df is not None:
    st.divider()
    st.subheader(f'Ranked results — {len(df)} candidates')
```

- `.get('ranked_df')` — reads the persisted DataFrame, `None` if never set (avoids
  `KeyError`). Rendering is thus **decoupled from computing** — results show from
  `session_state` regardless of whether the pipeline ran this rerun.
- `st.divider()` — horizontal rule; `st.subheader` with `len(df)` row count.

### Metric row

```python
top = df.iloc[0]
m1, m2, m3 = st.columns(3)
m1.metric('Top candidate', top['name'] or top['candidate_id'])
m2.metric('Top final score', f"{top['final_score']:.3f}")
m3.metric('Candidates ranked', len(df))
```

- `df.iloc[0]` — Pandas **integer-location** indexing: first row by position = the winner
  (DataFrame is pre-sorted best-first). Returns a Series.
- `st.columns(3)` — three equal columns; `st.metric(label, value)` = big KPI number.
- `top['name'] or top['candidate_id']` — short-circuit fallback (never show a blank name).
- `f"{...:.3f}"` — 3-decimal f-string format.

### Column config

```python
column_config = {
    'rank': st.column_config.NumberColumn('Rank', width='small'),
    'candidate_id': st.column_config.TextColumn('Candidate ID'),
    'name': st.column_config.TextColumn('Name'),
    'reasoning': st.column_config.TextColumn('Reasoning', width='large'),
}
for col in SCORE_COLS:
    label = col.replace('_', ' ').replace('score', '').strip().title()
    column_config[col] = st.column_config.ProgressColumn(
        label or col, min_value=0.0, max_value=1.0, format='%.3f',
    )
```

- **`st.column_config`** — per-column render control: dict of *column name → config object*.
  First string arg = display header. `reasoning` gets `width='large'` for long text.
- **Loop over `SCORE_COLS`** (payoff for the top-of-file constant) → each score column
  becomes a **`ProgressColumn`** (horizontal bar) so the spread is scannable.
  - `label = col.replace('_',' ').replace('score','').strip().title()` — prettifies
    snake_case → Title Case header (`final_score` → `Final`). `label or col` = fallback.
  - `min_value/max_value` = bar scale (scores normalized to [0,1]); `format='%.3f'` uses
    **C/printf-style** formatting (contrast the f-string `:.3f` elsewhere).

### Dataframe render

```python
st.dataframe(df, column_config=column_config, hide_index=True,
             use_container_width=True, height=560)
```

- **`st.dataframe`** — interactive (sortable/scrollable) table (vs static `st.table`).
- `hide_index=True` — hide the meaningless Pandas index (`rank` already numbers rows).
- `use_container_width=True` — full width (pairs with `layout='wide'`).
- `height=560` — fixed height; the table scrolls internally.

### CSV download

```python
csv_bytes = df[['candidate_id', 'rank', 'final_score', 'reasoning']] \
    .rename(columns={'final_score': 'score'}) \
    .to_csv(index=False).encode('utf-8')
st.download_button('⬇️ Download Ranked CSV', data=csv_bytes,
                   file_name='ranked_candidates.csv', mime='text/csv',
                   use_container_width=False)
```

- `df[[...]]` — column selection → leaner export DataFrame.
- `.rename(columns={'final_score': 'score'})` — export header rename (matches the
  "rename final_score to score" commit).
- `.to_csv(index=False)` → CSV string; `.encode('utf-8')` → **bytes** (required by
  `download_button`).
- **`st.download_button`** — client-side file download; `data`/`file_name`/`mime` set the
  payload, name, and content type.

### Expander + empty state

```python
    with st.expander('How to read the scores'):
        st.markdown('- **Final** — ... ')
else:
    st.info('Upload a file or click **Use Sample Data** to see the ranking.')
```

- **`st.expander`** — collapsible accordion (collapsed by default) tucking away the score
  legend.
- **`else`** of `if df is not None:` — the **empty state**: a blue `st.info` banner guiding a
  first-time user.

### Interview questions (results)

- `st.dataframe` vs `st.table` — interactive vs static.
- Why `.iloc[0]` — positional first row of a pre-sorted DataFrame.
- Why `.get()` on session_state — avoids `KeyError` before it's set.
- Why encode CSV to bytes — `download_button` requires bytes.
- Why `%.3f` (ProgressColumn) vs `:.3f` (f-string) — different formatting APIs.

---

## Big-picture takeaways (say these in an interview)

1. **Separation of concerns** — `app.py` is pure UI/orchestration; `demo_pipeline.py` holds
   all ML/ranking logic and is Streamlit-agnostic (progress reported via an injected callback).
2. **Streamlit rerun model** drives everything: `@st.cache_resource` (load the heavy model
   once), `st.session_state` (persist results across reruns), and awareness that
   `st.button` is `True` only on its click.
3. **Single source of truth** — `MAX_CANDIDATES` and `SCORE_COLS` imported/defined once and
   reused, so UI text and rendering can't drift from the pipeline.
4. **Graceful failure** — two-tier exception handling separates expected user-input errors
   from unexpected bugs; the app never shows a raw traceback.
5. **CWD-robust paths** via `__file__` for deployment portability (Hugging Face Spaces).
