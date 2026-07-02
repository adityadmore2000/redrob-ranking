# `demo_pipeline.py` — Deep Line-by-Line Teaching Notes

> The **in-memory, end-to-end ranking engine** behind the Streamlit demo. `app.py`
> calls exactly three things here: `load_candidates_from_bytes`, `get_model`, and
> `rank_candidates`. This module reuses the *real* production scoring logic (the
> `compute_*` functions in `phase1/`, the text builder, and the Phase-2 reasoning/
> ranking) but runs it all in memory on an uploaded pool instead of reading
> pre-computed artifacts from disk.

---

## Part 1 — Docstring: file-based production vs in-memory demo

```python
"""
demo_pipeline.py
================
In-memory, end-to-end ranking pipeline for the Streamlit demo.
...
"""
```

- **The key architectural idea:** production (`phase1/`, `phase2/rank.py`) is **file-based**
  — Phase 1 computes six artifacts once for a *fixed* candidate population and writes them to
  `artifacts/`; Phase 2 reads them back. That split exists so the heavy embedding pass is
  computed once and reused.
- The demo can't reuse artifacts because the user uploads a **fresh, arbitrary** pool — so
  every signal must be recomputed in memory for that exact pool.
- **Critical promise:** the semantic normalization, multiplicative final score, sort order,
  and reasoning text **mirror `phase2/rank.py` exactly** — the demo output matches the real
  submission. This is why the module *imports and reuses* production functions rather than
  reimplementing them.
- Documents the **Public API**: `load_candidates_from_bytes`, `rank_candidates`, `get_model`,
  `MAX_CANDIDATES`.
- **Interview:** "Why a separate demo pipeline?" → production is batch/file-based for a fixed
  pool; the demo is interactive over arbitrary input, so it recomputes in memory but reuses
  identical scoring code to stay consistent.

---

## Part 2 — Imports + `sys.path` manipulation

```python
import io
import json
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_PHASE1 = os.path.join(_HERE, 'phase1')
for _p in (_HERE, _PHASE1):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from field_map import get_candidate_id
import track1_hard_filter
import track1_availability
import track1_credibility
from track2_text_builder import build_candidate_text, build_jd_text
from phase2.rank import build_reasoning, normalize_semantic, rank_top_n
```

- `io` — imported but effectively unused here (a linter would flag it, like `json` in `app.py`).
- `numpy as np` — array/linear-algebra library; the numeric heart of the scoring (vectors,
  dot products, float64 arrays). `np` is the universal alias.
- `pandas as pd` — the ranked result is a DataFrame.
- **`sys.path` manipulation (the interview-worthy trick):**
  - `_HERE` = this file's directory (CWD-robust via `__file__`, same pattern as `app.py`).
  - `_PHASE1` = the `phase1/` subfolder.
  - The loop **prepends** both to `sys.path` (Python's module search path) *if not already
    present*. Why? So that `import track1_hard_filter` (a module that lives inside `phase1/`)
    works as a **flat top-level import** — matching how the production scripts import each
    other from *within* `phase1/`. Without this, you'd need `from phase1 import ...`, which
    might break the phase1 modules' own sibling imports.
  - `sys.path.insert(0, ...)` puts them at the **front** so these take priority over any
    same-named installed package.
- **Imports of real logic** — this is the "reuse, don't reimplement" contract:
  - `get_candidate_id` from `field_map` — canonical way to pull an ID from a candidate dict.
  - `track1_hard_filter/availability/credibility` — the three structured-signal scorers, each
    exposing `compute_all(candidates)`.
  - `build_candidate_text`, `build_jd_text` — turn a candidate/JD into the text that gets embedded.
  - `build_reasoning`, `normalize_semantic`, `rank_top_n` from `phase2.rank` — the exact
    Phase-2 functions, imported so the demo's normalization/ranking/reasoning are identical.
- **Interview:** what `sys.path` is and why insert at 0; why reuse production functions
  (consistency + single source of truth); note the mixed import styles
  (`import track1_*` flat vs `from phase2.rank import ...` packaged) reflect the path hack.

```python
MODEL_NAME = 'BAAI/bge-base-en-v1.5'
MAX_CANDIDATES = 100
```

- `MODEL_NAME` — the Hugging Face model id for the embedding model (BGE base, English, v1.5).
- `MAX_CANDIDATES` — the demo cap (also imported by `app.py` for its labels — single source
  of truth).

---

## Part 3 — `InputError`

```python
class InputError(ValueError):
    """Raised for user-facing input problems (bad format, too many rows, …)."""
```

- **What:** a custom exception **subclassing `ValueError`**. Subclassing `ValueError` (not
  bare `Exception`) means it's still caught by generic `except ValueError` and semantically
  reads as "the value/input was wrong."
- **Why:** lets `app.py` distinguish *expected, user-facing* input problems (show a friendly
  message) from *unexpected bugs* (generic message). This is the other half of the two-tier
  exception strategy documented in `app.md`.
- **Interview:** why subclass `ValueError` instead of `Exception` — narrower, semantically
  correct, still catchable by callers expecting value errors.

---

## Part 4 — `load_candidates_from_bytes` (parsing + validation)

```python
def load_candidates_from_bytes(raw: bytes, filename: str) -> list:
    name = (filename or '').lower()
    try:
        text = raw.decode('utf-8')
    except UnicodeDecodeError as exc:
        raise InputError('File is not valid UTF-8 text. ...') from exc

    text = text.strip()
    if not text:
        raise InputError('The uploaded file is empty.')
    ...
```

- **Type hints** `raw: bytes, filename: str -> list` — documentation + tooling aid (not
  enforced at runtime).
- `name = (filename or '').lower()` — defensive: handle `filename` being `None`, then
  lowercase so extension checks are case-insensitive.
- `raw.decode('utf-8')` — bytes → string. On failure, re-raise as a friendly `InputError`
  **`from exc`** (exception chaining: preserves the original traceback for debugging while
  showing a clean message). Empty-after-strip → `InputError`.

### Format detection (extension + content sniffing)

```python
    candidates = None
    if name.endswith('.jsonl') or (not name.endswith('.json') and text[0] != '['):
        candidates = _parse_jsonl(text)
    else:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            candidates = _parse_jsonl(text)   # maybe JSONL with a .json name
        else:
            if isinstance(parsed, list):
                candidates = parsed
            elif isinstance(parsed, dict):
                candidates = [parsed]
            else:
                raise InputError('JSON must be an array of candidate objects ...')
```

- **Robust, forgiving parsing** — prefers the file extension but **sniffs the first non-space
  char** so a mislabeled file still loads:
  - `.jsonl`, or *not* `.json* and doesn't start with `[` → treat as **JSON Lines**.
  - Otherwise try `json.loads` (a single JSON document).
    - `try/except/else`: the `else` runs only if no exception — good practice, keeps the
      `try` body minimal.
    - On `JSONDecodeError`, fall back to JSONL (handles JSONL saved with a `.json` name).
    - A parsed **list** → use directly; a single **dict** → wrap in a one-element list; anything
      else (number/string/bool) → `InputError`.
- **Interview:** why sniff content instead of trusting the extension (real-world files are
  mislabeled); why `try/except/else` (the `else` clarifies "success path").

### Structural validation

```python
    if not candidates:
        raise InputError('No candidate records found in the file.')
    if any(not isinstance(c, dict) for c in candidates):
        raise InputError('Every record must be a JSON object ...')
    missing_profile = sum(1 for c in candidates if 'profile' not in c)
    if missing_profile == len(candidates):
        raise InputError('No record has a "profile" field. ...')
    if len(candidates) > MAX_CANDIDATES:
        raise InputError(f'Too many candidates: {len(candidates)}. ... at most {MAX_CANDIDATES}. ...')
```

- Layered checks, each with a **specific** message:
  1. empty list;
  2. `any(not isinstance(c, dict) ...)` — every record must be an object;
  3. `sum(1 for ... if 'profile' not in c)` — if **all** records lack `profile`, this isn't a
     candidate dataset (note: only fails if *every* record is missing it — partial profiles are
     tolerated, matching the "neutral defaults" philosophy);
  4. cap enforcement against `MAX_CANDIDATES`.
- **Interview:** the `sum(1 for ...)` generator counts matches without building a list;
  `any(...)` short-circuits.

### ID backfill

```python
    for i, c in enumerate(candidates):
        if not c.get('candidate_id'):
            c['candidate_id'] = f'CAND_{i + 1:07d}'
    return candidates
```

- Guarantees every record has a **stable `candidate_id`** so downstream output/reasoning never
  breaks on a record that forgot one. `f'CAND_{i + 1:07d}'` = zero-padded 7-digit id
  (`CAND_0000001`), matching the dataset's native format.
- **Hidden:** mutates the input dicts in place (adds a key). Acceptable here since the list is
  freshly parsed and owned by this function.

---

## Part 5 — `_parse_jsonl`

```python
def _parse_jsonl(text: str) -> list:
    out = []
    for n, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise InputError(f'Line {n} is not valid JSON: {exc.msg}.') from exc
    return out
```

- Parses **one JSON object per line**. `enumerate(..., start=1)` gives **1-based line numbers**
  so the error message (`Line {n} is not valid JSON`) points at the human line number. Blank
  lines skipped. `exc.msg` extracts just the reason from the decode error.
- Leading `_` = module-private helper.

---

## Part 6 — `get_model`

```python
def get_model(device: str = 'cpu'):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_NAME, device=device)
```

- **`sentence-transformers`** is a library that wraps transformer models to produce
  **sentence/document embeddings** (fixed-length vectors capturing meaning). `SentenceTransformer(name)`
  downloads + loads the model.
- **The `import` is inside the function (lazy import)** — deliberate: the heavy library is only
  imported when a model is actually needed, keeping module import (and app startup) fast, and
  avoiding the cost in code paths that never rank (e.g. just validating input).
- `device='cpu'` default — CPU is safe everywhere; `app.py` wraps this call in
  `@st.cache_resource` so it runs once.
- **Interview:** what an embedding is; why lazy-import a heavy dependency.

---

## Part 7 — `rank_candidates` (the full pipeline)

```python
OUTPUT_COLUMNS = [
    'rank', 'candidate_id', 'name', 'final_score', 'semantic_score_raw',
    'semantic_score_norm', 'hard_filter_score', 'availability_score',
    'credibility_score', 'reasoning',
]
```

- Fixes the **column order** of the returned DataFrame, identical to what `phase2/rank.py`
  writes — so the demo table and the real submission line up.

```python
def rank_candidates(candidates: list, model, progress=None) -> pd.DataFrame:
    def _tick(frac, msg):
        if progress:
            progress(frac, msg)
    n = len(candidates)
```

- `progress` is the **optional callback** injected by `app.py` (`on_progress`). `_tick` is a
  tiny wrapper that only calls it if provided — so the pipeline runs headless too (e.g. in a
  notebook) with no UI coupling. This is the decoupling described in `app.md`.

### Track 1 — structured signals

```python
    _tick(0.05, 'Scoring structured signals (Track 1)…')
    hard = track1_hard_filter.compute_all(candidates)
    avail = track1_availability.compute_all(candidates)
    cred = track1_credibility.compute_all(candidates)

    track1_details = []
    for h, a, cr in zip(hard, avail, cred):
        row = {'candidate_id': h['candidate_id']}
        row.update(h); row.update(a); row.update(cr)
        track1_details.append(row)
```

- Each `compute_all` returns a **list of per-candidate detail dicts** (one per candidate,
  aligned by position). Fast, pure-Python.
- `zip(hard, avail, cred)` iterates the three lists **in lockstep**; each row is merged into
  one dict (`row.update(...)`) — the exact merged shape `rank.py`'s reasoning expects. The
  merged `track1_details[i]` later feeds `build_reasoning`.

```python
    candidate_ids = np.array([get_candidate_id(c) for c in candidates], dtype=object)
    hard_filter  = np.array([r['hard_filter_score']  for r in hard],  dtype=np.float64)
    availability = np.array([r['availability_score'] for r in avail], dtype=np.float64)
    credibility  = np.array([r['credibility_score']  for r in cred],  dtype=np.float64)
```

- Extract the three **score arrays** as `float64` NumPy arrays (fast vectorized math later).
  `candidate_ids` is `dtype=object` because ids are strings. All arrays are **position-aligned**
  with `candidates`.

### Track 2 — semantic embedding

```python
    _tick(0.20, 'Building candidate text representations…')
    texts = [build_candidate_text(c) for c in candidates]

    _tick(0.35, 'Embedding candidates against the job description…')
    from jd_parser import JD
    jd_text = build_jd_text(JD)

    all_vecs = model.encode(
        [jd_text] + texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    jd_vec = all_vecs[0]
    cand_vecs = all_vecs[1:]
    semantic = (cand_vecs @ jd_vec).astype(np.float64)
```

- `build_candidate_text(c)` — flattens each candidate profile into one text blob to embed.
- `JD` is imported lazily from `jd_parser` (the fixed job description object); `build_jd_text`
  renders it to text.
- **`model.encode([jd_text] + texts, ...)`** — the single most important line. It embeds the
  **JD prepended to all candidate texts in one batched call** (efficient; the JD is index 0).
  - `batch_size=32` — process 32 texts per forward pass (throughput vs memory tradeoff).
  - `convert_to_numpy=True` — return a NumPy array, not torch tensors.
  - `normalize_embeddings=True` — **L2-normalize** each vector to unit length. This is the
    crucial trick: once vectors are unit-length, their **dot product equals cosine similarity**.
  - `show_progress_bar=False` — suppress the library's own bar (the app shows its own).
- `jd_vec = all_vecs[0]`, `cand_vecs = all_vecs[1:]` — split JD from candidates.
- **`semantic = cand_vecs @ jd_vec`** — the `@` operator is **matrix multiplication**;
  `(N×D) @ (D,)` → `(N,)` = one similarity score per candidate. Because vectors are normalized,
  this dot product **is cosine similarity** (range ~[-1, 1], usually [0, 1] for related text).
- **Interview gold:** why normalize + dot product = cosine similarity; why embed JD in the same
  batch; what `@` does; batch_size tradeoff.

### Phase 2 — normalize, score, rank

```python
    _tick(0.75, 'Normalizing and computing final scores…')
    semantic_norm = normalize_semantic(semantic)
    final_score = semantic_norm * hard_filter * availability * credibility

    top_n = min(MAX_CANDIDATES, n)
    top_idx = rank_top_n(final_score, semantic_norm, hard_filter, top_n)
```

- `normalize_semantic(semantic)` — **min-max normalizes** raw similarities across *this pool*
  to [0, 1] (so scores are relative to the uploaded set). Reused from `phase2/rank.py`.
- **`final_score = semantic_norm * hard_filter * availability * credibility`** — the headline
  design decision: a **multiplicative** score. Because each factor is in [0, 1], **any near-zero
  signal drags the whole score toward zero** — a disqualifying signal (fails hard filter, not
  available) naturally sinks the candidate. Contrast a weighted *sum*, where a strong signal
  could mask a fatal weakness. This is element-wise NumPy multiplication over the aligned arrays.
- `rank_top_n(...)` — returns the **indices** of the top-N by `final_score`, with
  `semantic_norm` and `hard_filter` as **tie-breakers** (a stable lexsort, identical to
  production). `top_idx` is positions into the original arrays.
- **Interview gold:** why multiplicative not additive; what min-max normalization does and why
  it's pool-relative.

### Reasoning + row assembly

```python
    _tick(0.88, 'Generating per-candidate reasoning…')
    by_id = {str(get_candidate_id(c)): c for c in candidates}

    rows = []
    prev_desc_key = None
    for r, i in enumerate(top_idx, start=1):
        cid = str(candidate_ids[i])
        candidate = by_id.get(cid, {})
        profile = candidate.get('profile', {}) if candidate else {}
        reasoning, prev_desc_key = build_reasoning(
            r, profile, candidate, track1_details[i], semantic_norm[i], prev_desc_key,
        )
        if not reasoning:
            reasoning = f'Ranked #{r}; profile unavailable for detailed reasoning.'
        rows.append({
            'rank': r,
            'candidate_id': cid,
            'name': (profile.get('anonymized_name') or '').strip(),
            'final_score': round(float(final_score[i]), 6),
            'semantic_score_raw': round(float(semantic[i]), 6),
            'semantic_score_norm': round(float(semantic_norm[i]), 6),
            'hard_filter_score': round(float(hard_filter[i]), 6),
            'availability_score': round(float(availability[i]), 6),
            'credibility_score': round(float(credibility[i]), 6),
            'reasoning': reasoning,
        })

    _tick(1.0, 'Done.')
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
```

- `by_id` — a **dict comprehension** mapping id → candidate for O(1) lookup by id.
- `enumerate(top_idx, start=1)` — walk the ranked indices; `r` = 1-based rank, `i` = position
  in the original arrays.
- `build_reasoning(...)` returns `(reasoning_text, prev_desc_key)`. **`prev_desc_key` is threaded
  through the loop** — it lets the reasoning generator detect/avoid repeating identical
  descriptions across consecutive candidates (state carried between iterations). Reused verbatim
  from production so the demo's reasoning matches the submission.
- Fallback text if reasoning is empty.
- Each row: `float(...)` converts NumPy scalars to native Python floats, `round(..., 6)` to 6
  decimals (clean, deterministic output). Name defaults to empty string if missing.
- `pd.DataFrame(rows, columns=OUTPUT_COLUMNS)` — build the final table with the fixed column
  order. This is what `app.py` renders and downloads.
- **Interview:** why a dict comprehension for lookup (O(1) vs scanning); why thread
  `prev_desc_key`; why convert NumPy scalars to `float` (JSON/DataFrame cleanliness).

---

## Big-picture takeaways

1. **Reuse over reimplementation** — the demo imports the *exact* production `compute_*`,
   `normalize_semantic`, `rank_top_n`, and `build_reasoning`, guaranteeing the interactive
   output matches the real submission.
2. **Two tracks → one multiplicative score.** Track 1 (structured, pure-Python) and Track 2
   (semantic embeddings) combine as `semantic_norm × hard_filter × availability × credibility`,
   so any weak signal sinks the candidate.
3. **Cosine-similarity via normalized dot product** — `normalize_embeddings=True` + `@` is the
   efficient, standard way to score JD↔candidate match; the JD is embedded in the same batch.
4. **Robust, forgiving input parsing** — extension preference + content sniffing + layered
   validation, each failure surfaced as a friendly `InputError`.
5. **UI-agnostic** — progress via an optional callback; no Streamlit imports here.
6. **`sys.path` hack** lets it reuse the flat-imported `phase1/` modules unchanged.
