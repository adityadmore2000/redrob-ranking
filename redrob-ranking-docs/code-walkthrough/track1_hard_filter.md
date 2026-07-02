# `phase1/track1_hard_filter.py` — Deep Line-by-Line Teaching Notes

> **Track 1, signal #1: the "hard filter" score.** Answers "does this person even fit
> the non-negotiables?" — location, years of experience, work mode, consulting-firm
> disqualification, and job tenure. It is a **multiplicative** score, so any one bad
> factor caps the whole thing.

> This file also establishes the **standard shape every Track 1 module follows**:
> pure scoring functions → `compute_*_score(c)` → `compute_all(candidates)`, plus a
> file-based `run()`/`verify()` and an `argparse` CLI for the production batch job.
> (Availability and credibility mirror this structure.)

**The headline formula:**
```
hard_filter_score = location_score × yoe_score × work_mode_score × consulting_penalty × tenure_score
```
Each sub-score is in (0, 1], so a low sub-score (e.g. `consulting_penalty = 0.3`) drags the
product down — a disqualifying trait can't be "averaged away." Same multiplicative philosophy
as the final score.

---

## Part 1 — Docstring (the score spec)

The docstring is essentially the **spec**: it tables out every sub-score and its cutoffs
(e.g. Pune/Noida = 1.0, other Tier-1 India = 0.9, non-Tier-1 = 0.3/0.6, etc.). Worth reading as
the source of truth for the tuning. Key line: *"All field access goes through field_map
accessors. All thresholds come from field_map constants or jd_parser.JD. Nothing JD-specific is
hardcoded except `_JD_OFFICE_CITIES`."* — the module is deliberately data-driven with one
documented exception.

---

## Part 2 — `sys.path` bootstrap (dual launch support)

```python
import os as _os
import sys as _sys
_THIS_DIR = _os.path.dirname(_os.path.abspath(__file__))
_PROJECT_ROOT = _os.path.dirname(_THIS_DIR)
for _p in (_PROJECT_ROOT, _THIS_DIR):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
```

- Adds **both** the project root (so `field_map`, `jd_parser` import) **and** `phase1/` itself to
  `sys.path`. The comment explains *why both*: `python phase1/track1_hard_filter.py` puts only
  `phase1/` on the path, while `python -m phase1.track1_hard_filter` puts only the root — adding
  both makes the module work **regardless of how it's launched**.
- Underscore-aliased `_os`/`_sys` keep them out of the public namespace.
- **Interview:** the difference between running a file directly vs `-m` (module) and how each
  affects `sys.path[0]`.

---

## Part 3 — Imports + the one hardcoded set

```python
from field_map import (get_candidate_id, get_location, get_country, get_willing_to_relocate,
                       get_yoe, get_preferred_work_mode, get_current_company, get_career_sorted,
                       TIER_1_CITIES, CONSULTING_FIRMS, TENURE_IDEAL_MONTHS, ...)
from jd_parser import JD

_JD_OFFICE_CITIES = {'pune', 'noida'}
```

- Pulls **accessors + tuning constants** from `field_map`, and the parsed **`JD`** dict.
- **`_JD_OFFICE_CITIES = {'pune', 'noida'}`** — the *one* documented hardcode. Why it can't be
  parsed: the JD names Pune/Noida as the actual office (score 1.0) but *also* says
  Hyderabad/Mumbai/Delhi are "welcome to apply" (score 0.9). Both appear as plain city mentions
  in `JD['preferred_locations']`, so text parsing alone can't tell "office" from "welcome." The
  distinction is encoded here by hand, with the source JD line cited in the comment.
- **Interview:** a good example of *when* to hardcode — when the signal genuinely isn't in the
  text and you document why.

---

## Part 4 — `location_score`

```python
def _city_matches(location, city):
    return city in location.lower()   # 'Bangalore, Karnataka' matches 'bangalore'

def location_score(c):
    country = get_country(c).strip().lower()
    willing = bool(get_willing_to_relocate(c))
    if country != 'india':
        return 0.6 if willing else 0.3
    location = get_location(c)
    if any(_city_matches(location, city) for city in _JD_OFFICE_CITIES):
        return 1.0
    if any(_city_matches(location, city) for city in TIER_1_CITIES):
        return 0.9
    return 0.6 if willing else 0.3
```

- `_city_matches` is a **substring** test (not exact equality) so `"Bangalore, Karnataka"`
  matches `"bangalore"` — real location strings carry extra state/country text.
- Logic tiers: outside India → 0.3/0.6 by relocation willingness; office city → 1.0; other
  Tier-1 → 0.9; else 0.3/0.6. Willingness to relocate rescues an otherwise-poor location.
- `bool(...)` coerces the relocate flag to a real boolean.
- **Interview:** why substring matching for locations; how relocation modifies the score.

---

## Part 5 — `yoe_score`

```python
def yoe_score(c):
    yoe = get_yoe(c)
    lo, hi = JD['yoe_min'], JD['yoe_max']
    if lo is None or hi is None:
        return 1.0                      # no usable JD range → neutral
    if lo <= yoe <= hi:
        return 1.0                      # ideal band
    if (lo - 1) <= yoe < lo or hi < yoe <= (hi + 1):
        return 0.85                     # outer band (±1 year)
    return 0.6
```

- Uses the **parsed JD range** (`yoe_min/yoe_max` from `jd_parser`). Three bands: in-range = 1.0,
  within ±1 year = 0.85, else 0.6.
- **Graceful degradation:** if the JD had no parseable range, everyone gets a neutral 1.0 rather
  than being penalized for a parsing gap.
- Note the **chained comparison** `lo <= yoe <= hi` — Pythonic range check (not `lo <= yoe and
  yoe <= hi`).
- **Interview:** Python chained comparisons; why "no range → neutral 1.0" (don't punish for a
  parser limitation).

---

## Part 6 — `work_mode_score`

```python
def work_mode_score(c):
    mode = get_preferred_work_mode(c).strip().lower()
    preferred = {m.lower() for m in JD.get('preferred_work_modes', [])}
    return 1.0 if mode in preferred else 0.8
```

- Builds a **set** of JD-preferred modes (set → O(1) `in`), compares the candidate's mode. Match
  = 1.0, else a soft 0.8 (work mode is a preference, not a hard disqualifier — hence the gentle
  penalty).

---

## Part 7 — `consulting_penalty` (the disqualifier)

```python
def _is_consulting_firm(company):
    company = (company or '').lower()
    if not company:
        return False
    return any(firm in company for firm in CONSULTING_FIRMS)

def consulting_penalty(c):
    career = get_career_sorted(c)
    companies = [job['company'] for job in career if job.get('company')]
    if not companies:
        return 0.3 if _is_consulting_firm(get_current_company(c)) else 1.0
    if all(_is_consulting_firm(co) for co in companies):
        return 0.3
    return 1.0
```

- Encodes the JD's rule: penalize only candidates whose **entire** career is consulting firms
  (0.3). **`all(...)`** = every job is consulting. If they have *any* non-consulting experience,
  no penalty (1.0) — so someone currently at Infosys but who previously worked at a product
  company is fine.
- Empty career history → fall back to judging the current company.
- `_is_consulting_firm` uses substring matching over `CONSULTING_FIRMS`.
- **Interview:** why `all(...)` (only fully-consulting careers are disqualified); the fallback
  path.

---

## Part 8 — Tenure (job-hopping / title-chasing signal)

```python
def average_tenure_months(c):
    career = get_career_sorted(c)
    durations = [job.get('duration_months') or 0 for job in career]
    durations = [d for d in durations if d]     # drop zeros/missing
    if not durations:
        return None
    return sum(durations) / len(durations)

def tenure_score(c):
    avg = average_tenure_months(c)
    if avg is None:
        return TENURE_SCORE_UNKNOWN            # 0.75 neutral
    if avg >= TENURE_IDEAL_MONTHS:   return TENURE_SCORE_IDEAL      # >=24 → 1.0
    if avg >= TENURE_STABLE_MONTHS:  return TENURE_SCORE_STABLE     # >=18 → 0.85
    if avg >= TENURE_MODERATE_MONTHS: return TENURE_SCORE_MODERATE  # >=12 → 0.70
    return TENURE_SCORE_SHORT                                        # <12 → 0.50
```

- Average months per role = a proxy for **stability vs job-hopping**. Two-step filtering: default
  missing durations to 0, then drop the zeros so they don't drag the mean down.
- `None` (no usable data) → neutral `0.75` (don't punish absence of data).
- Cascading `if` thresholds map the average to a score, all from `field_map` constants (no magic
  numbers inline).
- **Interview:** why filter zeros separately; why "unknown" is neutral, not zero.

---

## Part 9 — Assembling the score

```python
def compute_hard_filter_score(c: dict) -> dict:
    loc = location_score(c); yoe = yoe_score(c); wm = work_mode_score(c)
    pen = consulting_penalty(c); ten = tenure_score(c)
    avg_tenure = average_tenure_months(c)
    return {
        'hard_filter_score': loc * yoe * wm * pen * ten,
        'location_score': loc, 'yoe_score': yoe, 'work_mode_score': wm,
        'consulting_penalty': pen, 'tenure_score': ten,
        'average_tenure_months': avg_tenure,
    }

def compute_all(candidates: list) -> list:
    results = []
    for c in candidates:
        scores = compute_hard_filter_score(c)
        scores['candidate_id'] = get_candidate_id(c)
        results.append(scores)
    return results
```

- `compute_hard_filter_score` returns **both the final product and every sub-score** — the
  breakdown is what the reasoning generator later cites ("strong location, penalized for tenure").
  Transparency by design.
- **`compute_all`** is the **public entry point** the demo pipeline calls: map the scorer over all
  candidates, tagging each with its `candidate_id`. Same signature across all three Track 1
  modules (a consistent interface).

---

## Part 10 — File-based production path (`run` / `verify` / CLI)

```python
def _load_candidates(input_path):
    import json
    with open(input_path) as f:
        head = f.read(1); f.seek(0)
        if head == '[':
            return json.load(f)          # JSON array fixture
        return [json.loads(line) for line in f if line.strip()]   # JSONL

def run(input_path, artifacts_dir='artifacts'):
    ...
    hard_filter_scores = np.array(hard_filter_scores, dtype=np.float32)
    np.save(os.path.join(artifacts_dir, 'hard_filter_scores.npy'), hard_filter_scores)
    with open(os.path.join(artifacts_dir, 'track1_details.pkl'), 'wb') as f:
        pickle.dump(track1_details, f)
    ids_path = os.path.join(artifacts_dir, 'candidate_ids.npy')
    if os.path.exists(ids_path):
        print('... not overwriting')
    else:
        np.save(ids_path, np.array(candidate_ids, dtype=object))
```

- `_load_candidates` sniffs the **first character** (`[` → JSON array, else JSONL) — same
  forgiving idea as `demo_pipeline`. `f.read(1); f.seek(0)` peeks then rewinds.
- **`run`** is the production batch path (used to precompute artifacts for the fixed 100k pool, not
  used by the interactive demo). It:
  - saves `hard_filter_scores.npy` as **float32** (`.npy` = NumPy's binary array format; float32
    halves memory vs float64 for a large array).
  - **pickles** `track1_details.pkl` (Python's object serialization) since the details are dicts,
    not a numeric array.
  - **`candidate_ids.npy` — the "don't clobber" guard:** this file defines the canonical candidate
    ordering shared by *all* Phase-1 scripts and is normally written by `track2_embedding.py`. So
    `run` only writes it **if it doesn't already exist** — protecting the shared ordering from being
    overwritten with a different one. This is a subtle but important **cross-artifact consistency**
    contract.
  - Progress prints every 10,000 candidates with `flush=True` (force output immediately, useful in
    long batch logs).
- **`verify`** reloads the artifacts and prints a summary — a lightweight sanity check.
- **Interview:** `.npy` vs pickle (numeric array vs arbitrary objects); float32 vs float64
  tradeoff; why not clobber `candidate_ids.npy`; `flush=True`.

```python
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(...)
    parser.add_argument('--input', default='/kaggle/input/.../candidates.jsonl', ...)
    parser.add_argument('--artifacts', default='artifacts', ...)
    parser.add_argument('--verify', action='store_true', ...)
    args = parser.parse_args()
    if args.verify: verify(args.artifacts)
    else:
        run(args.input, args.artifacts); verify(args.artifacts)
```

- Standard **`argparse` CLI**. `--verify` uses `action='store_true'` (a **flag**: present = True,
  absent = False — no value needed). The default input path points at the Kaggle dataset location
  (this was developed/run on Kaggle). Running the script computes then immediately verifies.
- **Interview:** `argparse` basics; `store_true` flags; why a `run`-then-`verify` pattern.

---

## Big-picture takeaways

1. **Multiplicative sub-scores** — location × yoe × work_mode × consulting × tenure; any weak
   factor caps the result, mirroring the final-score philosophy.
2. **Data-driven with one documented hardcode** (`_JD_OFFICE_CITIES`) — and the comment explains
   exactly why the parser can't derive it.
3. **Transparency** — every sub-score is returned, feeding human-readable reasoning later.
4. **Graceful defaults** — no JD range → neutral yoe; unknown tenure → 0.75; missing data never
   crashes or unfairly penalizes.
5. **Dual identity** — pure functions + `compute_all` for the in-memory demo, *and* a
   `run/verify/CLI` for the file-based production batch, sharing a canonical `candidate_ids.npy`.
