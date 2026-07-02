# `eda/eda_full.py` + `eda/eda_career_history.py` — Deep Line-by-Line Teaching Notes

> **Exploratory Data Analysis (EDA) scripts.** These are the *origin story* of the whole
> project: they profile the raw candidate dataset and **produce the numbers that became the
> constants** in `field_map.py` (normalization ranges, tier scores, token budgets) and the
> design decisions in the scorers (which signals to keep/drop). They aren't run in
> production — they're one-off, flat top-level scripts whose *output* informed the code.

**Why they matter in an interview:** they show the work is **data-driven, not guessed**. Every
threshold in the pipeline traces back to a percentile printed here. When asked "why is
`RESPONSE_TIME_MAX = 280.0`?", the honest answer is "EDA measured it" — and this is that EDA.

---

## Shared idioms (both files)

```python
import json
import numpy as np
from collections import defaultdict, Counter

candidates = []
with open('/kaggle/input/.../candidates.jsonl', 'r') as f:
    for line in f:
        candidates.append(json.loads(line.strip()))
```

- Load the full JSONL into memory (fine for a Kaggle analysis run; the hardcoded `/kaggle/input/...`
  path shows where this executed).
- **`collections.Counter`** — a dict subclass for **counting**: `Counter(list)` tallies occurrences,
  `.most_common(k)` returns the top-k by count. The workhorse for categorical distributions.
- **`collections.defaultdict(int)` / `defaultdict(list)`** — a dict that **auto-creates a default
  value** for missing keys (`0` for `int`, `[]` for `list`), so you can write
  `field_counts[k] += 1` without first checking if `k` exists. Removes boilerplate.
- **`np.percentile(arr, q)`** — the core statistic here; `p25`/`median`/`p75`/`p90` describe a
  distribution far better than mean alone (robust to outliers).
- **The ASCII bar chart trick:** `"█" * min(40, int(40 * count / total))` — draw a proportional bar
  in the terminal by repeating a block character. `min(40, ...)` caps the width. Poor-man's
  histogram, no plotting library needed.
- **Interview:** `Counter`/`defaultdict` use cases; why report percentiles; terminal
  visualization with string repetition.

---

## `eda/eda_full.py` — profiling every field

Organized into printed **sections** (profile, skills, education, redrob signals, token budget).

### Helpers

```python
def p(arr, pct): return np.percentile(arr, pct)
def safe_get(c, *keys):
    val = c
    for k in keys:
        if not isinstance(val, dict): return None
        val = val.get(k)
    return val
def days_since(date_str):
    if not date_str: return None
    try:
        d = datetime.strptime(date_str[:10], "%Y-%m-%d").date()
        return (date.today() - d).days
    except: return None
```

- **`safe_get(c, *keys)`** — a mini nested-accessor using **`*keys` (varargs)** to walk a path:
  `safe_get(c, 'profile', 'years_of_experience')` drills `c['profile']['years_of_experience']`,
  returning `None` at any missing/non-dict level. This is the **prototype of `field_map`'s
  accessors** — the same defensive idea, written inline before it was formalized.
- **`datetime.strptime(s, "%Y-%m-%d")`** — parse a date string by an explicit format; `[:10]` takes
  the `YYYY-MM-DD` prefix. (Bare `except:` again — throwaway-script style.)
- **Interview:** `*args` varargs; why a safe nested getter; `strptime` format codes.

### Section 1–4 — field distributions

Each field follows the same recipe: **extract → filter `None` → `np.array` → print percentiles /
Counter distribution**. A few analyses that directly shaped the code:

- **[1.5] consulting firms** — counts how many candidates are at TCS/Infosys/etc. → justified the
  `CONSULTING_FIRMS` set and the consulting penalty.
- **[4.1] github_activity_score** — measures the **`-1` sentinel** proportion vs real scores, and
  the real-score range → became `GITHUB_SCORE_MIN/MAX` and the `get_has_github` sentinel logic.
- **[4.2] notice_period_days** — the value distribution → shaped `NOTICE_PERIOD_SCORES`.
- **[4.4] last_active** recency buckets → `LAST_ACTIVE_MIN/MAX_DAYS`.
- **[4.5] avg_response_time_hours** → `RESPONSE_TIME_MIN/MAX`.
- **[4.8] interview_completion_rate** — measured its **narrow range**, which is *why it was
  excluded* from the availability score.
- **[4.15] skill_assessment_scores** — measured ~24% presence → *why it's a nudge, not a weighted
  term* in credibility.
- **Interview:** point to any pipeline constant and trace it to the EDA cell that measured it.

### Section 5 — token budget derivation

```python
def approx_tokens(text):
    if not text: return 0
    return max(1, int(len(str(text)) / 4))
...
fixed = med(title_tok) + med(headline_tok) + med(summary_tok) + med(skills_tok) + med(edu_tok)
career_budget = 512 - fixed
jobs_fit = career_budget // 99
weights = [0.40, 0.30, 0.20, 0.10]
for i, (w, label) in enumerate(zip(weights, labels)):
    print(f"Job {i} ({label}): ~{int(career_budget * w)} tokens")
```

- **The origin of `CAREER_TOKEN_BUDGET`.** Uses the same **4-chars/token** approximation as the text
  builder. Computes median tokens for the fixed fields, subtracts from **BGE's 512-token limit** to
  get the **career-history budget**, then splits it by recency weights (0.40/0.30/0.20/0.10) per job
  position. Those percentages → the recency-weighted per-job budgets baked into `field_map`.
- **Interview:** how a model's context limit (512) drives a token-allocation strategy; recency
  weighting justified by data.

---

## `eda/eda_career_history.py` — drilling into career history

A focused companion that answers "how should we serialize career history?" — the analysis behind
the text builder's recency weighting.

### Key analyses

```python
# [3] which fields exist inside career_history entries
field_counts = defaultdict(int); total_jobs = 0
for c in candidates:
    for job in c.get("career_history", []):
        total_jobs += 1
        for k in job.keys():
            field_counts[k] += 1
```
- Discovers the **schema** of job entries empirically (field presence %), rather than assuming it.

```python
# [4]/[5] description length BY JOB POSITION (0 = most recent)
history_sorted = sorted(history, key=lambda x: x.get("start_date","") or "", reverse=True)
for pos, job in enumerate(history_sorted):
    desc_words_by_pos[pos].append(word_count(desc))
```
- **The crucial finding:** measures description length **as a function of recency position**. Uses
  `defaultdict(list)` keyed by position, sorts by `start_date` descending (same key-defaulting sort
  the real accessors use). This per-position distribution is exactly what justified giving recent
  jobs a larger token budget — the data showed where the information density is.
- The fallback loop `for field in ["description", "responsibilities", "summary"]` picks the first
  populated description-like field — discovering the data isn't consistent about field names.

```python
# [6] token budget simulation vs 512 limit — same 0.40/0.30/0.20/0.10 split
```
- Re-derives the career token budget independently (cross-checking `eda_full.py`).

```python
# [7] print 3 sample career histories with 3+ jobs
```
- Qualitative spot-check — actually *look* at real records to confirm the quantitative story.

- **Interview:** empirical schema discovery; analyzing a feature *conditioned on position/recency*;
  combining quantitative percentiles with qualitative sample inspection.

---

## Big-picture takeaways

1. **These scripts justify the constants.** Normalization ranges, tier scores, the `-1` sentinel
   handling, excluded features, and the token budgets all come from percentiles/counts printed here.
2. **`Counter` + `defaultdict` + `np.percentile`** are the EDA toolkit; ASCII bars give quick
   terminal histograms.
3. **`safe_get` is the ancestor of `field_map`'s accessors** — the same defensive nested-read idea.
4. **Recency-conditioned analysis** (description length by job position) is what motivated the text
   builder's recency-weighted token budget.
5. **Throwaway-script style** (flat code, hardcoded Kaggle paths, bare `except`) is appropriate for
   analysis — but note the contrast with the disciplined, accessor-driven production modules.
