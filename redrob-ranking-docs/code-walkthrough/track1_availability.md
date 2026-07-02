# `phase1/track1_availability.py` — Deep Line-by-Line Teaching Notes

> **Track 1, signal #2: availability.** How ready/reachable is this candidate *right
> now* — open to work, recently active, fast to respond, short notice period, engaged
> with recruiters, actively applying. Follows the same module shape as the hard filter
> (scorers → `compute_*_score` → `compute_all`).

**The key contrast to call out in an interview:** the hard filter **multiplies** its
sub-scores (any one can veto). Availability instead uses a **weighted average** — because
these are *soft, additive* signals where a weakness in one (e.g. slow response) should be
*balanced* against strengths in others, not treated as a disqualifier.

```
availability_score = Σ (signal_i × weight_i),   Σ weight_i = 1.0
```

---

## Part 1 — Docstring: weights + a deliberate exclusion

The docstring lists the six sub-signals and their weights (open_to_work 0.30, last_active
0.25, response_time 0.20, notice_period 0.15, recruiter_response 0.05, applications_30d 0.05
— summing to 1.0), and one notable design note:

> *"`interview_completion_rate` is intentionally excluded: EDA showed a very narrow range
> (0.30–1.00, p25=0.48, p75=0.76) making it a weak discriminator."*

- **Interview point:** a signal with low *variance* across the population can't separate
  candidates, so it's dropped. Feature selection driven by EDA, not intuition.

---

## Part 2 — Imports + weights + `_clamp`

```python
from datetime import date
from field_map import (get_candidate_id, get_open_to_work, get_last_active_date,
                       get_avg_response_time, get_notice_period, get_recruiter_response_rate,
                       get_applications_30d, LAST_ACTIVE_MIN_DAYS, LAST_ACTIVE_MAX_DAYS,
                       RESPONSE_TIME_MIN, RESPONSE_TIME_MAX, NOTICE_PERIOD_SCORES)

WEIGHTS = {'open_to_work':0.30, 'last_active':0.25, 'response_time':0.20,
           'notice_period':0.15, 'recruiter_response':0.05, 'applications_30d':0.05}

def _clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))
```

- `from datetime import date` — Python's date type, for recency math.
- Note **no `sys.path` bootstrap here** — this module lives in `phase1/` but assumes it's run
  with the project root already importable (the demo adds it via `demo_pipeline`'s path hack;
  the `__main__` block below runs from the project root). A small inconsistency vs the hard
  filter, worth noticing.
- `WEIGHTS` centralizes the weighting; the comment asserts they sum to 1.0 (so the result stays
  in [0, 1] when each signal is in [0, 1]).
- **`_clamp(x, lo, hi)` = `max(lo, min(hi, x))`** — the classic one-liner to bound a value into a
  range. `min(hi, x)` caps the top, `max(lo, ...)` floors the bottom. Used everywhere below to
  keep normalized scores in [0, 1].
- **Interview:** derive why `max(lo, min(hi, x))` clamps correctly.

---

## Part 3 — The six sub-signals

### open_to_work

```python
def open_to_work_signal(c):
    return 1.0 if get_open_to_work(c) else 0.5
```
- Binary flag → 1.0 / 0.5 (note the floor is 0.5, not 0 — being not-flagged isn't a hard veto).

### last_active (recency, min-max inverted)

```python
def last_active_signal(c):
    raw = get_last_active_date(c)
    if not raw:
        return 0.0
    try:
        last = date.fromisoformat(raw[:10])
    except (ValueError, TypeError):
        return 0.0
    days_since = (date.today() - last).days
    score = 1.0 - (days_since - LAST_ACTIVE_MIN_DAYS) / (LAST_ACTIVE_MAX_DAYS - LAST_ACTIVE_MIN_DAYS)
    return _clamp(score)
```
- `date.fromisoformat(raw[:10])` — parse the first 10 chars (`YYYY-MM-DD`) of the date string,
  ignoring any time portion. Wrapped in `try/except (ValueError, TypeError)` so a malformed/absent
  date → treated as stale (0.0).
- `(date.today() - last).days` — subtracting two dates gives a **`timedelta`**; `.days` extracts
  whole days.
- **The normalization formula** `1.0 - (days_since - MIN)/(MAX - MIN)` is **inverted min-max**:
  more days since active → *lower* score. At MIN days → 1.0, at MAX days → 0.0, clamped outside.
- **Interview:** min-max normalization and why it's inverted here (recent = good); `timedelta.days`.

### response_time (inverted min-max, "faster = better")

```python
def response_time_signal(c):
    hours = get_avg_response_time(c)
    score = 1.0 - (hours - RESPONSE_TIME_MIN) / (RESPONSE_TIME_MAX - RESPONSE_TIME_MIN)
    return _clamp(score)
```
- Same inverted-normalization pattern: fewer response hours → higher score. (Recall the
  accessor defaults missing response time to the slow end, 280.0 → score ~0.)

### notice_period (nearest-key lookup)

```python
def notice_period_signal(c):
    days = get_notice_period(c)
    if days in NOTICE_PERIOD_SCORES:
        return NOTICE_PERIOD_SCORES[days]
    nearest = min(NOTICE_PERIOD_SCORES, key=lambda k: abs(k - days))
    return NOTICE_PERIOD_SCORES[nearest]
```
- Table lookup, but robust to values not in the table: **`min(..., key=lambda k: abs(k - days))`**
  finds the **nearest key** by absolute distance. Elegant "snap to closest bucket."
- **Interview:** using `min` with a `key` to find the closest value; why not require exact keys.

### recruiter_response

```python
def recruiter_response_signal(c):
    return _clamp(get_recruiter_response_rate(c))
```
- Already a rate in [0, 1]; just clamped defensively.

### applications_30d (piecewise linear)

```python
def applications_30d_signal(c):
    apps = get_applications_30d(c)
    if apps <= 0:
        return 0.3
    score = 0.5 + (apps - 1) / (24 - 1) * (1.0 - 0.5)
    return _clamp(score, 0.5, 1.0)
```
- **Piecewise:** 0 applications → 0.3 (not actively looking). 1–24 → linearly mapped onto
  [0.5, 1.0]; 25+ clamps at 1.0. The formula `0.5 + (apps-1)/23 * 0.5` is the general
  **linear interpolation** `start + t·(end-start)` where `t = (apps-1)/23`.
- Note the discontinuity: 0 apps = 0.3 but 1 app = 0.5 (a small jump encoding "at least trying").
- **Interview:** linear interpolation formula; why a piecewise/discontinuous mapping is used.

---

## Part 4 — Weighted combination

```python
def compute_availability_score(c: dict) -> dict:
    signals = {
        'open_to_work': open_to_work_signal(c),
        'last_active': last_active_signal(c),
        'response_time': response_time_signal(c),
        'notice_period': notice_period_signal(c),
        'recruiter_response': recruiter_response_signal(c),
        'applications_30d': applications_30d_signal(c),
    }
    availability = sum(signals[k] * WEIGHTS[k] for k in WEIGHTS)
    return {'availability_score': availability, 'open_to_work_signal': signals['open_to_work'], ...}
```

- **`sum(signals[k] * WEIGHTS[k] for k in WEIGHTS)`** — the weighted average as a generator
  expression: multiply each signal by its weight and sum. Iterating over `WEIGHTS` keys guarantees
  every weight is applied (and would `KeyError` if a signal were missing — a useful fail-fast).
- Returns the final score **plus every sub-signal** (transparency for reasoning, like the hard
  filter).
- **Interview:** weighted average vs the hard filter's product — *when* to use each (soft,
  compensable signals → average; hard, vetoing constraints → product).

```python
def compute_all(candidates: list) -> list:
    results = []
    for c in candidates:
        scores = compute_availability_score(c)
        scores['candidate_id'] = get_candidate_id(c)
        results.append(scores)
    return results
```
- Identical public interface to the other Track 1 modules — the demo calls this uniformly.

---

## Part 5 — `__main__` block (EDA-style self-report)

```python
if __name__ == '__main__':
    import json, os
    jsonl_path = os.path.join('data', 'candidates.jsonl')
    sample_path = os.path.join('data', 'sample_candidates.json')
    LIMIT = 1000
    ... load up to LIMIT candidates from jsonl, else sample array, else SystemExit ...
    results = compute_all(candidates)

    def pctile(values, q):
        s = sorted(values)
        if not s: return float('nan')
        idx = min(len(s)-1, max(0, int(round(q * (len(s)-1)))))
        return s[idx]

    # print a min / p25 / median / p75 / max table per component, then 3 samples
```

- Not used by the app — this is a **developer diagnostic**. Running the file loads up to
  `LIMIT=1000` candidates (preferring the full JSONL, falling back to the sample array,
  `raise SystemExit(...)` with a helpful message if neither exists) and prints a **distribution
  table** (min/p25/median/p75/max) for each sub-signal plus a few worked examples.
- **`pctile`** is a tiny hand-rolled percentile (sort, index at `round(q·(n-1))`, clamped) — no
  NumPy dependency needed for a quick check. `float('nan')` guards the empty case.
- The `!r` in `{get_last_active_date(c)!r}` uses **`repr()`** formatting (shows quotes/escapes) —
  handy for spotting whitespace/None in raw values.
- `raise SystemExit(msg)` — clean exit with an error message and non-zero status (better than a
  bare `print` + `return` at module top level).
- **Interview:** `if __name__ == '__main__'` for a diagnostic harness; `!r` vs `!s` in f-strings;
  a from-scratch percentile.

---

## Big-picture takeaways

1. **Weighted average, not product** — availability signals are compensable, so a weakness in one
   is balanced by strength in others (contrast the hard filter).
2. **EDA-driven design** — weights and the *exclusion* of `interview_completion_rate` (too little
   variance) come from data analysis.
3. **Normalization toolkit** — inverted min-max for recency/response-time, nearest-key lookup for
   notice period, piecewise linear interpolation for applications; all clamped to [0, 1].
4. **Robust parsing** — malformed/missing dates degrade to a stale score rather than crashing.
5. **Same public interface** (`compute_all`) as the other Track 1 modules + a self-contained
   diagnostic `__main__`.
