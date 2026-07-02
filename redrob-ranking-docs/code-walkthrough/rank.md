# `phase2/rank.py` — Deep Line-by-Line Teaching Notes

> **The Phase 2 entry point: turn Phase 1's artifacts into the final ranking + a
> human-readable reasoning sentence per candidate, then write `submission.csv`.** This is
> the biggest, most nuanced file. Two ideas dominate: (1) the **multiplicative final
> score + stable multi-key sort**, and (2) the **honest, non-hallucinating reasoning
> generator** that only cites facts actually present in a profile.
>
> `demo_pipeline.py` imports `normalize_semantic`, `rank_top_n`, and `build_reasoning`
> from here directly — so understanding this file explains the demo's output too.

**Self-contained by design:** stdlib + numpy only. No model, no network, no GPU, no LLM.
Runs in well under 5 minutes on CPU. The heavy embedding work already happened in Phase 1.

---

## Part 1 — Docstring + module constants

```python
TOP_N = 100
_NPY_ARTIFACTS = {'candidate_ids':'candidate_ids.npy', 'semantic':'semantic_scores.npy',
                  'hard_filter':'hard_filter_scores.npy', 'availability':'availability_scores.npy',
                  'credibility':'credibility_scores.npy'}
_PKL_ARTIFACT = 'track1_details.pkl'
```

- `TOP_N = 100` — the submission size (capped at population for small fixtures).
- The artifact filename maps name → the seven parallel files written by Phase 1.

### The JD vocabularies — and the anti-hallucination principle

```python
_JD_SKILL_HINTS = {'embedding':'production embeddings', 'retrieval':'retrieval systems',
                   'faiss':'vector search', 'bm25':'ranking/IR', 'python':'Python', ...}
_JD_TEXT_TERMS = ('embedding','retrieval','vector','search','rank','rerank', ...)
_PROFICIENCY_RANK = {'expert':3, 'advanced':2, 'intermediate':1, 'beginner':0}
```

- `_JD_SKILL_HINTS` maps a raw skill token → a **human-readable JD requirement label**. Comment
  (paraphrased): *"Used only to decide which of a candidate's **actual** skills/text to surface —
  never to invent capability they do not have."* **This is the ethical core of the reasoning
  generator:** it never claims a skill the candidate lacks; it only *labels* skills they really
  list. Say this in an interview — it's the anti-hallucination guarantee.
- `_JD_TEXT_TERMS` — terms to scan for in free text (descriptions/summaries).
- `_PROFICIENCY_RANK` here includes `beginner: 0` (unlike the text builder, which excluded it) —
  because reasoning ranks/filters differently (`>= 1` threshold below).

---

## Part 2 — STEP 1: `load_artifacts`

```python
def load_artifacts(artifacts_dir):
    missing = [fn for fn in list(_NPY_ARTIFACTS.values()) + [_PKL_ARTIFACT]
               if not os.path.exists(os.path.join(artifacts_dir, fn))]
    if missing:
        raise SystemExit(f"Missing artifact(s) in {artifacts_dir}: {', '.join(missing)}")
    arrays = {}
    arrays['candidate_ids'] = np.load(..., allow_pickle=True)
    for key in ('semantic','hard_filter','availability','credibility'):
        arrays[key] = np.load(..., allow_pickle=False).astype(np.float64)
    with open(..., 'rb') as f:
        track1_details = pickle.load(f)
    ...
    lengths = {len(arrays['candidate_ids']), len(arrays['semantic']), ..., len(track1_details)}
    if len(lengths) != 1:
        raise SystemExit(f"Parallel artifacts have inconsistent lengths: {sorted(lengths)}. ...")
    return arrays, track1_details
```

- **Fail-fast presence check** — lists every missing file, `raise SystemExit` with a clear message
  (clean exit, non-zero status).
- **`allow_pickle` discipline:** `True` only for the string `candidate_ids` (object array needs it);
  `False` for numeric arrays (security default — a malicious pickled `.npy` could execute code).
- Scores cast to `float64` for precise downstream math.
- **The set-length-1 alignment check** (same trick as `verify_artifacts`): all six lengths in a
  set; `len(lengths) != 1` means they disagree → refuse to proceed. This enforces the
  **parallel-array contract** before anything can go silently wrong.
- **Interview:** why validate alignment up front; `allow_pickle` security.

---

## Part 3 — STEP 2: `normalize_semantic`

```python
def normalize_semantic(semantic):
    lo = float(semantic.min()); hi = float(semantic.max())
    span = hi - lo
    if span == 0:
        norm = np.zeros_like(semantic)
    else:
        norm = (semantic - lo) / span
    ...
    return norm
```

- **Min-max normalization across the whole population:** `(x - min) / (max - min)` maps raw cosine
  similarities to [0, 1]. This makes the semantic score **relative to the pool** (important: it's
  why the demo's semantic column is described as "relative to the uploaded set").
- **Flat-input guard:** if every score is identical (`span == 0`), return all zeros instead of
  dividing by zero (`np.zeros_like` matches shape/dtype).
- **Interview:** min-max vs z-score normalization; the divide-by-zero edge case.

---

## Part 4 — STEP 4: `rank_top_n` (stable multi-key sort)

```python
def rank_top_n(final_score, semantic_norm, hard_filter, n):
    order = np.lexsort((-hard_filter, -semantic_norm, -final_score))
    return order[:n]
```

- **The interview-critical line.** `np.lexsort` performs a **stable multi-key sort**, but with two
  quirks you must explain:
  1. It sorts **ascending**, so each key is **negated** (`-final_score`) to get **descending**.
  2. It treats the **last** tuple element as the **primary (most significant)** key. So the tuple
     `(-hard_filter, -semantic_norm, -final_score)` means: **primary = final_score, secondary =
     semantic_norm, tertiary = hard_filter** — keys listed *least-significant first*.
- Returns the indices of the top `n` — positions into the original parallel arrays.
- **Why tie-breakers matter:** many candidates can share a final score; `semantic_norm` then
  `hard_filter` break ties deterministically so the ranking is stable/reproducible.
- **Interview gold:** explain `np.lexsort`'s ascending-and-last-key-primary behavior and why the
  keys are negated and reverse-ordered. This is a very common "gotcha" question.

---

## Part 5 — STEP 5: `load_top_profiles` (memory-efficient streaming)

```python
def load_top_profiles(candidates_path, wanted_ids):
    wanted = set(wanted_ids)
    found = {}
    with open(candidates_path) as f:
        head = f.read(1); f.seek(0)
        if head == '[':
            for c in json.load(f):
                cid = c.get('candidate_id')
                if cid in wanted and cid not in found:
                    found[cid] = c
                    if len(found) == len(wanted): break
        else:
            for line in f:
                line = line.strip()
                if not line: continue
                c = json.loads(line)
                cid = c.get('candidate_id')
                if cid in wanted and cid not in found:
                    found[cid] = c
                    if len(found) == len(wanted): break
    return found
```

- We only need the **full profiles of the top ~100** (for reasoning), not all 100k. This **streams
  the file line-by-line** (JSONL path) so **the full population is never held in memory** — it
  parses one record at a time, keeps only wanted ids.
- `wanted = set(...)` → O(1) membership; **early exit** (`if len(found) == len(wanted): break`)
  stops reading as soon as all top profiles are found — a big win when the top-100 appear early.
- Same `[`-sniff to support the JSON-array fixture.
- **Interview:** streaming/lazy iteration vs loading everything; why a set for lookup; the early-exit
  optimization.

---

## Part 6 — STEP 6: the reasoning generator (the heart of the file)

### `_current_role`

```python
def _current_role(profile, career_history):
    title = (profile.get('current_title') or '').strip()
    company = (profile.get('current_company') or '').strip()
    desc = ''
    if career_history:
        current = next((j for j in career_history if j.get('is_current')), career_history[0])
        desc = (current.get('description') or '').strip()
    return title, company, desc
```

- Pulls the current role. **`next((j for j ... if j.get('is_current')), career_history[0])`** —
  find the first job flagged `is_current`, **defaulting to the most recent** (`career_history[0]`)
  if none is flagged. The 2-arg form of `next(generator, default)` is the idiomatic "find first
  match or fallback."
- **Interview:** `next(gen, default)` as a safe "first-or-default" find.

### `_top_skills`

```python
def _top_skills(skills, k=3):
    ranked = [s for s in skills
              if isinstance(s, dict) and s.get('name')
              and _PROFICIENCY_RANK.get((s.get('proficiency') or '').lower(), -1) >= 1]
    ranked.sort(key=lambda s: (_PROFICIENCY_RANK.get((s.get('proficiency') or '').lower(), 0),
                               s.get('endorsements') or 0), reverse=True)
    return ranked[:k]
```

- Filters to named skills with proficiency **≥ intermediate** (`>= 1`; excludes beginner/unknown),
  then sorts by **(proficiency, endorsements)** tuple key, descending; returns top `k`.
- Only ever returns skills the candidate **actually has** (anti-hallucination again).

### `_jd_hits` — evidence-based JD matching

```python
def _jd_hits(skills, desc, summary):
    hits, seen = [], set()
    for s in skills:
        name = (s.get('name') or '').lower()
        for token, label in _JD_SKILL_HINTS.items():
            if token in name and label not in seen:
                seen.add(label); hits.append(label)
    blob = f'{desc} {summary}'.lower()
    for term in _JD_TEXT_TERMS:
        label = _JD_SKILL_HINTS.get(term)
        if term in blob and label and label not in seen:
            seen.add(label); hits.append(label)
    return hits
```

- Returns which JD requirement **areas the candidate actually shows evidence for**, drawn from
  (a) their real skill names and (b) their free text. Deduped by label, order preserved.
- **This is the anti-hallucination engine:** a JD requirement label appears *only if* a matching
  token is present in the candidate's real data. It never asserts unearned capability.
- **Interview:** how to ground generated text in source facts (retrieval-of-evidence, not
  generation-of-claims).

### `_clean`

```python
def _clean(text):
    return ' '.join(text.replace(',', ';').split())
```

- **CSV-safety:** replaces commas with semicolons (the CSV uses comma delimiters) and
  `.split()`+`' '.join()` collapses all runs of whitespace/newlines into single spaces — so the
  reasoning is one tidy CSV-safe line. (Belt-and-suspenders alongside the CSV writer's own quoting.)

### `_career_clause` — the de-duplication logic

```python
_GENERIC_DESC_OPENERS = ('built systems that', 'worked at the intersection', 'responsible for', ...)

def _career_clause(title, company, industry, desc, prev_desc_key):
    desc = (desc or '').strip()
    def _fallback(): ...  # constructs "Current role: Currently {title} at {company} in {industry} industry."
    if len(desc) < 30:
        return _fallback(), None
    lowered = desc.lower()
    if any(lowered.startswith(op) for op in _GENERIC_DESC_OPENERS):
        snippet = desc.split('.', 1)[0][:100].rstrip()   # generic → first sentence, 100 chars
    else:
        snippet = desc[:200].rstrip()                    # else up to 200 chars
    fingerprint = snippet[:100]
    if prev_desc_key is not None and fingerprint == prev_desc_key:
        return _fallback(), fingerprint                  # same as previous row → construct instead
    return 'Current role: ' + snippet + '.', fingerprint
```

- Builds the "what they do now" clause with three safeguards against **templated, repetitive
  output**:
  1. **Too short (<30 chars)** → construct a sentence from title/company/industry instead of
     quoting noise.
  2. **Generic opener** (dataset descriptions often start with boilerplate) → keep only the first
     sentence, capped at 100 chars.
  3. **Duplicate of the previous candidate** → detect via a 100-char **fingerprint**; if it matches
     the prior row's, fall back to a constructed sentence so consecutive rows don't read
     identically.
- **`prev_desc_key` is threaded between calls** (returned and passed back in) — carrying state
  across candidates so adjacent rows never share description text. This is exactly the
  `prev_desc_key` variable seen in `demo_pipeline` and `main` below.
- **Interview:** why de-dup generated text; how a fingerprint + threaded state achieves it without
  global state.

### `build_reasoning` — composing the sentence

```python
def build_reasoning(rank, profile, candidate, det, semantic_norm, prev_desc_key=None):
    ...
    # opening "who they are": role + (yoe, location, willing-to-relocate)
    # semantic phrasing tied to normalized score:
    if sn >= 0.7: match = 'Strong semantic match...'
    elif sn >= 0.45: match = 'Moderate...'
    else: match = 'Weaker...'
    jd_hits = _jd_hits(...)
    match += ' — relevant to ' + '; '.join(jd_hits[:4]) if jd_hits else ' — limited ... exposure'
    # top skills (real only)
    # current-role clause (non-repeating)
    # concerns: signal-driven flags (location/notice/github/consulting/tenure/linkedin/semantic)
    # closing verdict toned to rank
    return _clean(' '.join(sentences)), desc_key
```

- Assembles the reasoning as a list of `sentences`, then cleans and joins. Structure: **strengths
  first → JD connections → skills → current-role evidence → honest concerns → rank-toned verdict.**
- `f'{yoe:g}y exp'` — the `:g` format drops trailing zeros (`6.5` stays `6.5`, `6.0` → `6`).
- **Semantic phrasing is bucketed by the normalized score** (0.7 / 0.45 thresholds) so the prose
  matches the number.
- **The "concerns" section is signal-driven and always emits at least one flag** (falling back to
  *"No major disqualifying signals detected."*). Each concern is gated on a Track 1 sub-signal from
  `det` (e.g. `notice_sig < 0.70` → "90d+ notice"). This makes the reasoning **honest about
  weaknesses**, not just a sales pitch — and it reads the *detail dict*, so the prose is tied to
  the actual scores.
  - Note `elif` chains pick the **most severe** flag (e.g. notice period, location bands).
  - `avg_ten` formatting guards type: `f'{avg_ten:.0f}'` only if it's a real number, else `'short'`.
- **Closing verdict toned to rank:** `rank <= 10` → "Strong overall fit...", `<= 40` → "Solid
  fit...", `<= 75` → "Reasonable...", else "Borderline...". Calibrates confidence to position.
- Returns `(reasoning, desc_key)` — the `desc_key` threads forward for de-duplication.
- **Interview gold:** how to generate *honest*, grounded, non-repetitive natural-language
  explanations from structured signals **without an LLM** — pure rules over real facts, with
  concerns surfaced and confidence calibrated to rank.

---

## Part 7 — `main` (orchestration)

```python
def main():
    parser = argparse.ArgumentParser(...)
    parser.add_argument('--candidates', required=True, ...)
    parser.add_argument('--out', required=True, ...)
    default_artifacts = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'artifacts')
    parser.add_argument('--artifacts', default=default_artifacts, ...)
    args = parser.parse_args()
    ...
    arrays, track1_details = load_artifacts(args.artifacts)                      # STEP 1
    semantic_norm = normalize_semantic(semantic)                                  # STEP 2
    final_score = semantic_norm * hard_filter * availability * credibility        # STEP 3
    n = min(TOP_N, len(candidate_ids))
    top_idx = rank_top_n(final_score, semantic_norm, hard_filter, n)              # STEP 4
    ... print top-10 preview ...
    top_ids = [str(candidate_ids[i]) for i in top_idx]
    profiles = load_top_profiles(args.candidates, top_ids)                        # STEP 5
    ... warn about any missing profiles ...
    # STEP 6 — reasoning loop (threads prev_desc_key)
    for r, i in enumerate(top_idx, start=1):
        ... build_reasoning(...) ; rows.append({...rounded scores..., 'reasoning':...}) ...
    # STEP 8 — sanity checks
    # STEP 7 — write CSV
```

- **Argparse:** `--candidates` and `--out` required; `--artifacts` defaults to `../artifacts`
  relative to this file (CWD-robust).
- **STEP 3 — the final score:** `semantic_norm * hard_filter * availability * credibility` — the
  headline **equal-weight multiplicative** combination (element-wise over aligned NumPy arrays).
  Any near-zero factor sinks the candidate; a disqualifying penalty propagates naturally. Same
  formula the demo uses.
- **Timing dict** (`timings[...] = time.time() - t`) instruments each stage; printed at the end.
- **Missing-profile warning** — if a top id has no matching profile, it's flagged and reasoning
  degrades gracefully rather than crashing.
- **Reasoning loop threads `prev_desc_key`** across iterations (the de-dup state) and rounds every
  score to 6 decimals.

### STEP 8 — sanity checks before writing

```python
print(f'  Total rows written           : {len(rows)} (expected {n})')
dup = len(ids) - len(set(ids));  print(f'  Duplicate candidate_ids : {dup} (expected 0)')
empty = sum(1 for row in rows if not row['reasoning'].strip())
hf_low = sum(1 for row in rows if row['hard_filter_score'] < 0.5)
sn_low = sum(1 for row in rows if row['semantic_score_norm'] < 0.3)
```

- **Self-auditing before output** — checks row count, duplicate ids (`len - len(set)`), empty
  reasoning, and "should never be in the top-100" conditions (hard_filter < 0.5, semantic_norm <
  0.3). These are *assertions about the ranking's quality*, printed for the operator. Great
  defensive-engineering habit to call out.

### STEP 7 — write CSV

```python
columns = ['candidate_id', 'rank', 'score', 'reasoning']
os.makedirs(out_dir, exist_ok=True)
with open(args.out, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
    writer.writeheader()
    for row in rows:
        writer.writerow({'candidate_id':..., 'rank':..., 'score': row['final_score'], 'reasoning':...})
```

- **`csv.DictWriter`** — writes dict rows to CSV by field name. `newline=''` is the **required
  idiom** when opening files for the csv module (prevents doubled line endings on Windows).
- `extrasaction='ignore'` — if a row dict has extra keys not in `fieldnames`, ignore them (the row
  dicts carry more columns than the CSV needs).
- Note the submission CSV emits only **candidate_id, rank, score, reasoning** (with `final_score`
  written as `score`) — the lean contest format.
- **Interview:** why `newline=''` for csv; `DictWriter` + `extrasaction`.

```python
if __name__ == '__main__':
    main()
```

---

## Big-picture takeaways

1. **Final score = `semantic_norm × hard_filter × availability × credibility`** — equal-weight
   multiplicative, so any disqualifying signal propagates; the project's central design decision.
2. **`np.lexsort` stable multi-key ranking** — primary final_score, tie-broken by semantic_norm then
   hard_filter; negated keys for descending; last-key-primary — the classic gotcha.
3. **Memory-efficient top-N profile loading** — stream the file, keep only wanted ids, early-exit.
4. **Honest, non-hallucinating reasoning without an LLM** — labels only real skills/evidence,
   surfaces concerns from Track 1 signals, de-duplicates via threaded fingerprints, tones the
   verdict to rank.
5. **Operational rigor** — up-front artifact alignment check, per-stage timings, pre-write sanity
   audit, CSV-safe cleaning, `allow_pickle=False` by default.
6. **Shared functions** — `normalize_semantic`, `rank_top_n`, `build_reasoning` are imported by the
   demo, guaranteeing identical output.
