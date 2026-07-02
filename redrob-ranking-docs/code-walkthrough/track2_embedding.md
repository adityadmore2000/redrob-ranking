# `phase1/track2_embedding.py` — Deep Line-by-Line Teaching Notes

> **The Phase 1 master runner (production, file-based).** It orchestrates *everything*
> Phase 2 needs: loads candidates, computes all three Track 1 scores, builds candidate
> texts, embeds them + the JD with BGE, computes cosine similarity, and writes seven
> artifacts to `artifacts/`. This is the batch counterpart to `demo_pipeline.py`'s
> in-memory flow — same math, but persisted to disk for a fixed 100k-candidate pool.

**Public API:** `embed_texts`, `cosine_similarity_matrix`, `run`, `verify_artifacts`.

---

## Part 1 — Docstring: the artifact contract

The docstring lists the **seven parallel artifacts** and — critically — states:

> *"All parallel arrays (candidate_ids, semantic_scores, the three Track 1 score arrays, and
> track1_details) share the same candidate order."*

- **This shared ordering is the whole contract.** Phase 2 loads these arrays independently and
  assumes `semantic_scores[i]`, `hard_filter_scores[i]`, `candidate_ids[i]`, etc. all refer to the
  *same* candidate. If the order diverged, rankings would be silently wrong. (This is why the hard
  filter refuses to clobber `candidate_ids.npy`.)
- `.npy` for numeric arrays, `.pkl` for the list-of-dicts `track1_details`.
- **Interview:** what "parallel arrays" means and why a shared index order is a fragile-but-common
  design that must be guarded.

---

## Part 2 — Imports + `sys.path` bootstrap

```python
import json, os, pickle, sys, time
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
for _p in (_PROJECT_ROOT, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import numpy as np
from field_map import get_candidate_id
from track2_text_builder import build_candidate_text, build_jd_text
import track1_hard_filter, track1_availability, track1_credibility
MODEL_NAME = 'BAAI/bge-base-en-v1.5'
```

- Same dual-launch `sys.path` trick as the hard filter (works whether run as a file or `-m`).
- Imports the three Track 1 scorers + the text builder — this module *is* the composition point.

---

## Part 3 — `load_candidates`

```python
def load_candidates(input_path: str) -> list:
    with open(input_path) as f:
        head = f.read(1); f.seek(0)
        if head == '[':
            return json.load(f)
        return [json.loads(line) for line in f if line.strip()]
```

- Same first-char sniff (`[` → JSON array, else JSONL) seen across the project. Peek-and-rewind
  (`read(1)` then `seek(0)`).

---

## Part 4 — `embed_texts` (batched embedding)

```python
def embed_texts(texts: list, model, batch_size: int = 64) -> np.ndarray:
    from tqdm import tqdm   # lazy: Track-1-only runs need no tqdm
    embeddings = []
    for start in tqdm(range(0, len(texts), batch_size), desc='Embedding', unit='batch'):
        batch = texts[start:start + batch_size]
        vecs = model.encode(batch, batch_size=batch_size, show_progress_bar=False,
                            convert_to_numpy=True, normalize_embeddings=False)
        embeddings.append(np.asarray(vecs, dtype=np.float32))
    if not embeddings:
        return np.zeros((0, 0), dtype=np.float32)
    return np.vstack(embeddings)
```

- **Manual batching** over `range(0, len(texts), batch_size)` with slice `texts[start:start+bs]` —
  processes chunks to avoid running out of memory on 100k texts. `tqdm(...)` wraps the loop for a
  progress bar (`desc`/`unit` label it).
- **`normalize_embeddings=False` here** — note the contrast with `demo_pipeline`, which normalized
  inside `encode`. This module normalizes *later*, inside `cosine_similarity_matrix`. Both reach
  the same result; this split lets it store raw vectors (e.g. the JD embedding) and normalize at
  comparison time.
- **`np.vstack(embeddings)`** — vertically stacks the per-batch `(bs, dim)` arrays into one
  `(N, dim)` matrix. The empty-input guard returns a `(0, 0)` array instead of crashing on
  `vstack([])`.
- `tqdm` is lazy-imported so `--skip-embeddings` runs need no tqdm.
- **Interview:** why batch (memory); `np.vstack` for concatenating batches; float32 for size.

---

## Part 5 — `cosine_similarity_matrix` (the manual, numerically-safe version)

```python
def cosine_similarity_matrix(candidate_embeddings, jd_embedding) -> np.ndarray:
    cand = np.asarray(candidate_embeddings, dtype=np.float32)
    jd = np.asarray(jd_embedding, dtype=np.float32).ravel()
    cand_norms = np.linalg.norm(cand, axis=1)
    cand_norms[cand_norms == 0] = 1.0            # avoid divide-by-zero
    cand_unit = cand / cand_norms[:, None]
    jd_norm = np.linalg.norm(jd)
    if jd_norm == 0:
        jd_norm = 1.0
    jd_unit = jd / jd_norm
    return cand_unit @ jd_unit
```

- **This is cosine similarity written out explicitly** (vs `demo_pipeline`'s shortcut of
  pre-normalizing then dotting). Worth understanding fully — it's a classic interview question:
  - `jd.ravel()` — flatten to 1-D `(dim,)`.
  - `np.linalg.norm(cand, axis=1)` — the **L2 norm (length)** of each candidate vector; `axis=1`
    computes it per row → shape `(N,)`.
  - **`cand_norms[cand_norms == 0] = 1.0`** — the **divide-by-zero guard**: a zero-length vector
    would produce `NaN` when divided; replacing its norm with 1.0 leaves it as the zero vector
    (similarity 0) instead. Same guard for the JD.
  - `cand / cand_norms[:, None]` — **broadcasting**: `cand_norms[:, None]` reshapes `(N,)` →
    `(N, 1)` so each row is divided by its own norm, yielding **unit vectors**.
  - `cand_unit @ jd_unit` — `(N, dim) @ (dim,)` → `(N,)`: the dot product of unit vectors **is
    cosine similarity**, one score per candidate, range [-1, 1].
- **Interview gold:** define cosine similarity (`a·b / (|a||b|)`); what `axis=1` and `[:, None]`
  (broadcasting) do; why guard zero norms.

---

## Part 6 — `_merge_track1_details`

```python
def _merge_track1_details(hard, avail, cred):
    merged = []
    for h, a, cr in zip(hard, avail, cred):
        row = {'candidate_id': h['candidate_id']}
        row.update(h); row.update(a); row.update(cr)
        merged.append(row)
    return merged
```

- Combines the three per-candidate Track 1 dicts into one (order preserved via `zip`) — this
  merged structure is what Phase 2's reasoning reads. (`demo_pipeline` does the same merge inline.)

---

## Part 7 — `run` (the six-step pipeline + skip-embeddings mode)

```python
def run(input_path, artifacts_dir='artifacts', batch_size=256, device=None, skip_embeddings=False):
    os.makedirs(artifacts_dir, exist_ok=True)
    ...
    # [1/6] load  → [2/6] Track 1 scores + merge → build parallel np arrays
    candidate_ids = np.array([get_candidate_id(c) for c in candidates], dtype=object)
    hard_scores  = np.array([r['hard_filter_score']  for r in hard],  dtype=np.float32)
    avail_scores = np.array([r['availability_score'] for r in avail], dtype=np.float32)
    cred_scores  = np.array([r['credibility_score']  for r in cred],  dtype=np.float32)

    if skip_embeddings:
        # save Track 1 artifacts only; leave semantic_scores/jd_embedding untouched
        np.save(.../'candidate_ids.npy', candidate_ids); ...
        return

    # [3/6] texts → [4/6] load model (auto device) → [5/6] embed + cosine → [6/6] save all
```

- **Six timed, numbered stages** with `print(..., end='', flush=True)` progress — the standard
  batch-job UX. Each stage prints its elapsed time.
- `os.makedirs(artifacts_dir, exist_ok=True)` — create the output dir if absent, no error if
  present.
- **Parallel arrays built together** in one place, guaranteeing the shared order the artifact
  contract promises. `candidate_ids` is `dtype=object` (strings); scores are `float32`.
- **`skip_embeddings` mode (nice operational design):** recompute *only* the cheap Track 1
  artifacts and re-save them, **deliberately leaving `semantic_scores.npy` / `jd_embedding.npy`
  untouched**. This lets you repair a stale `track1_details.pkl` **without paying for a full
  GPU embedding pass** — a real-world "don't recompute the expensive thing" optimization. Early
  `return` skips steps 3–5.

### Steps 4–5 (model + embedding)

```python
    import torch
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME, device=device)

    candidate_embeddings = embed_texts(texts, model, batch_size=batch_size)
    from jd_parser import JD
    jd_text = build_jd_text(JD)
    jd_embedding = embed_texts([jd_text], model, batch_size=batch_size)[0]
    semantic_scores = cosine_similarity_matrix(candidate_embeddings, jd_embedding)
```

- **Auto device detection** via `torch.cuda.is_available()` — GPU if present, else CPU. `torch`
  and `sentence_transformers` are **lazy-imported** so the skip-embeddings path never pays for
  them.
- Candidates embedded in batches; the **JD embedded separately** as a single vector (`[jd_text]`
  then `[0]` to unwrap the one-row result). Cosine similarity computed via Part 5.

### Step 6 (save)

```python
    np.save(.../'candidate_ids.npy', candidate_ids)
    np.save(.../'semantic_scores.npy', semantic_scores.astype(np.float32))
    np.save(.../'hard_filter_scores.npy', hard_scores)
    ... availability, credibility ...
    np.save(.../'jd_embedding.npy', jd_embedding.astype(np.float32))
    with open(.../'track1_details.pkl', 'wb') as f:
        pickle.dump(track1_details, f)
```

- All seven artifacts written, casting to `float32` to keep files small. Numeric → `.npy`,
  dict-list → pickle.

---

## Part 8 — `verify_artifacts` (post-run sanity)

```python
def verify_artifacts(artifacts_dir='artifacts'):
    candidate_ids = np.load(.../'candidate_ids.npy', allow_pickle=True)
    semantic = np.load(.../'semantic_scores.npy', allow_pickle=False)
    ...
    lengths = {len(candidate_ids), len(semantic), len(hard), len(avail), len(cred), len(track1_details)}
    print(f'  parallel lengths consistent: {len(lengths) == 1}')
    def dist(name, arr):
        q = np.percentile(arr, [0, 25, 50, 75, 100]); print(...)
    ...
```

- Reloads everything, prints shapes, score distributions (`np.percentile`), and 3 samples.
- **The clever consistency check:** put all six lengths into a **set**; if they're all equal the
  set has **exactly one element**, so `len(lengths) == 1` is `True`. A one-line way to assert
  "all parallel arrays are the same length" — directly validating the artifact contract.
- **`allow_pickle=True` for `candidate_ids`** (it's an object array of Python strings, which needs
  pickling to load) but **`allow_pickle=False` for the numeric arrays** — a **security best
  practice**: only allow pickle where genuinely required, since `allow_pickle=True` can execute
  arbitrary code from a malicious `.npy`.
- **Interview gold:** the set-length-1 trick; why `allow_pickle=False` is the safe default and when
  you must set it `True`.

---

## Part 9 — `argparse` CLI

```python
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='/kaggle/input/.../candidates.jsonl', ...)
    parser.add_argument('--artifacts', default='artifacts', ...)
    parser.add_argument('--batch-size', type=int, default=256, help='64 for CPU, 256+ for GPU')
    parser.add_argument('--device', default=None, help='cpu/cuda, auto-detect if None')
    parser.add_argument('--verify', action='store_true', ...)
    parser.add_argument('--skip-embeddings', action='store_true', ...)
    args = parser.parse_args()
    if args.verify:
        verify_artifacts(args.artifacts)
    else:
        run(args.input, args.artifacts, args.batch_size, args.device, skip_embeddings=args.skip_embeddings)
```

- Full CLI exposing every operational lever: input path, output dir, batch size (`type=int`;
  default 256 for GPU, comment notes 64 for CPU), device, and the two mode flags. `--batch-size`
  becomes `args.batch_size` (argparse converts dashes to underscores).
- **Interview:** `argparse` type coercion and dash→underscore attribute naming; `store_true` flags
  for modes.

---

## Big-picture takeaways

1. **Composition point** — this module wires together Track 1 (three scorers) + Track 2 (text
   builder + BGE) and persists seven **parallel, index-aligned artifacts**.
2. **Cosine similarity, spelled out** — norms, zero-guard, broadcasting, unit-vector dot product;
   the canonical implementation.
3. **Operational maturity** — auto GPU/CPU detection, batched embedding with tqdm, a
   `--skip-embeddings` repair mode that avoids the expensive pass, and a `verify` step with a
   one-line parallel-length assertion.
4. **Safety** — `allow_pickle=False` for numeric loads by default; only the string id array opts in.
5. **Same math as the demo** — consistency with `demo_pipeline.py` is deliberate; the difference is
   file-based batch vs in-memory interactive.
