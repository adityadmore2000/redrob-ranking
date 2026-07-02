# `field_map.py` — Deep Line-by-Line Teaching Notes

> **The single source of truth for reading candidate data.** Every scoring module
> (Track 1 and Track 2) imports accessors from here instead of digging into the raw
> candidate dict directly. It also holds the project's tuning **constants** (city lists,
> normalization ranges, tier scores).

**The core discipline (say this in an interview):** never write `c['profile']['location']`
in a scoring script. Always call `get_location(c)`. This centralizes (a) the field *path*,
(b) the *default* when missing, and (c) the *type* — so a schema change touches one file,
not twenty. This is the classic **anti-corruption layer / accessor pattern**.

---

## Part 1 — Docstring

```python
"""
field_map.py — single source of truth for all field paths
Both Track 1 and Track 2 import from here.
Never access candidate fields directly in scoring scripts — always use these accessors.
"""
```

- States the contract explicitly: all field access goes through this module.

---

## Part 2 — Top-level container accessors

```python
def get_profile(c):       return c.get('profile', {}) or {}
def get_signals(c):       return c.get('redrob_signals', {}) or {}
def get_skills(c):        return c.get('skills', []) or []
def get_education(c):     return c.get('education', []) or []
def get_career(c):        return c.get('career_history', []) or []
def get_certifications(c):return c.get('certifications', []) or []
```

- Each returns a **sub-container** (dict or list) of the candidate.
- **The `... or {}` / `... or []` idiom is the crucial defensive trick.** `c.get('profile', {})`
  already defaults to `{}` if the key is *missing* — but if the key is *present and explicitly
  `None`* (common in real data), `.get` returns `None`. The `or {}` converts that `None` into an
  empty container, so downstream `.get(...)` never crashes with `AttributeError: 'NoneType'`.
- **Interview:** "What's the difference between `c.get('profile', {})` and
  `c.get('profile', {}) or {}`?" → the second also handles an explicit `null`/`None` value, not
  just a missing key. This is the single most repeated pattern in the file.

---

## Part 3 — Profile field accessors

```python
def get_candidate_id(c):     return c.get('candidate_id')
def get_current_title(c):    return get_profile(c).get('current_title', '') or ''
def get_current_company(c):  return get_profile(c).get('current_company', '') or ''
...
def get_yoe(c):              return get_profile(c).get('years_of_experience') or 0.0
def get_willing_to_relocate(c):
    return get_signals(c).get('willing_to_relocate', False)   # lives in redrob_signals per EDA
```

- Each reads through `get_profile(c)` (never the raw dict), applies the `or ''` / `or 0.0`
  default, so callers always get a **usable, correctly-typed value**.
- `get_yoe` defaults to `0.0` (numeric) — years of experience should be a float.
- `get_willing_to_relocate` — note the comment: this field actually lives in `redrob_signals`,
  **not** `profile`, "per EDA" (discovered during exploratory data analysis). This is *exactly*
  why the accessor layer exists: it hides that non-obvious location so no scorer has to know it.
- **Interview:** why defaults are typed (`''` for strings, `0.0` for numbers, `False` for
  booleans) — so arithmetic/string ops downstream never hit a `None`.

---

## Part 4 — Skills accessors

```python
def get_skill_names(c):
    return [s.get('name', '') for s in get_skills(c) if isinstance(s, dict)]

def get_skill_name_proficiency(c):
    return [
        (s.get('name', ''), s.get('proficiency', ''), s.get('duration_months', 0))
        for s in get_skills(c) if isinstance(s, dict)
    ]
```

- List comprehensions that also **filter with `if isinstance(s, dict)`** — defends against a
  malformed skills list containing non-dict junk (a string, `None`). Real-world data is messy.
- `get_skill_name_proficiency` returns **tuples** `(name, proficiency, duration_months)` — a
  compact structured form the credibility/text builders consume.

---

## Part 5 — Education accessors + tier ranking

```python
def get_education_entries(c):
    return [ {'institution':..., 'degree':..., 'field_of_study':..., 'tier':..., 'end_year':...}
             for e in get_education(c) if isinstance(e, dict) ]

def get_education_tier(c):
    """Return highest education tier (tier_1 > tier_2 > tier_3 > tier_4)."""
    tier_rank = {'tier_1': 1, 'tier_2': 2, 'tier_3': 3, 'tier_4': 4}
    tiers = [e.get('tier', '') for e in get_education(c) if isinstance(e, dict)]
    ranked = [tier_rank[t] for t in tiers if t in tier_rank]
    return min(ranked) if ranked else 4  # lower = better
```

- `get_education_entries` normalizes each education record into a **fixed-shape dict** (always
  the same keys, always defaulted).
- `get_education_tier` — nice bit of logic worth explaining:
  - Maps tier strings to ranks where **lower number = better** (`tier_1` → 1).
  - `ranked = [tier_rank[t] for t in tiers if t in tier_rank]` — the `if t in tier_rank` guard
    skips unrecognized/empty tiers (avoids `KeyError`).
  - `min(ranked)` = the candidate's **best** (lowest-number) tier across all degrees.
  - Fallback `4` (worst) if no valid tier — a neutral-but-conservative default.
- **Interview:** why "lower = better" and why `min` gives the *best* tier; why the `if t in
  tier_rank` guard matters.

---

## Part 6 — Career history accessors

```python
def get_career_sorted(c):
    history = get_career(c)
    try:
        history_sorted = sorted(history, key=lambda x: x.get('start_date', '') or '', reverse=True)
    except Exception:
        history_sorted = history
    return [ {'title':..., 'company':..., 'duration_months':..., 'is_current':..., ...}
             for job in history_sorted ]

def get_current_company_from_career(c):
    career = get_career_sorted(c)
    return career[0]['company'] if career else ''
```

- Sorts jobs by `start_date` **descending** (most recent first) so index 0 is the current/most
  recent role.
- `key=lambda x: x.get('start_date', '') or ''` — sort key defaults to `''` so entries with a
  missing date don't crash the sort (comparing `None` to a string would raise).
- **`try/except Exception` around `sorted`** — belt-and-suspenders: if the data is malformed
  enough that sorting still fails, fall back to the unsorted list rather than crashing the whole
  pipeline. Pragmatic robustness over strictness.
- Each job normalized to a fixed-shape dict with typed defaults.
- `get_current_company_from_career` — a **fallback** for `profile.current_company` (uses the most
  recent career entry when the profile field is blank).
- **Interview:** why default the sort key; why wrap `sorted` in try/except (defensive against
  dirty data).

---

## Part 7 — Availability signal accessors

```python
def get_open_to_work(c):          return get_signals(c).get('open_to_work_flag', False)
def get_avg_response_time(c):     return get_signals(c).get('avg_response_time_hours') or 280.0
def get_notice_period(c):         return get_signals(c).get('notice_period_days') or 90
...
```

- All read through `get_signals(c)` (the `redrob_signals` block).
- **Defaults encode a philosophy:** missing data defaults to the *pessimistic/neutral* end.
  `avg_response_time` → `280.0` (the slowest, from EDA); `notice_period` → `90` days (long).
  So a candidate who simply hasn't provided the data isn't rewarded for the gap.
- **Interview:** why choose pessimistic defaults — to avoid gaming/ over-crediting incomplete
  profiles; missing ≠ good.

---

## Part 8 — Credibility signal accessors + the `-1` sentinel

```python
def get_github_raw(c):     return get_signals(c).get('github_activity_score', -1)
def get_has_github(c):     return get_github_raw(c) != -1
def get_github_score(c):
    raw = get_github_raw(c)
    return raw if raw != -1 else None
```

- **The `-1` sentinel pattern (interview-worthy):** GitHub score uses `-1` to mean "no GitHub
  account at all," distinct from a *real* score of `0` (has an account, no activity). These are
  semantically different and must not be conflated.
  - `get_has_github` → boolean "do they even have an account."
  - `get_github_score` → the real score, or `None` if no account (so the scorer can treat
    "no account" ≠ "score of 0").
- Other accessors (`get_endorsements`, `get_profile_completeness`, `get_verified_email`, …)
  follow the same defaulted pattern.
- `get_skill_assessment_scores` → dict, defaults to `{}`.
- **Interview:** why a sentinel instead of `None` in the data; why "no GitHub" and "GitHub
  score 0" must be distinguished.

---

## Part 9 — Constants (tuning knobs derived from EDA)

```python
TIER_1_CITIES = {'bangalore', 'bengaluru', 'mumbai', 'delhi', ...}
CONSULTING_FIRMS = {'tcs', 'infosys', 'wipro', 'accenture', 'cognizant', 'capgemini'}
LAST_ACTIVE_MIN_DAYS = 23;  LAST_ACTIVE_MAX_DAYS = 263
RESPONSE_TIME_MIN = 2.1;    RESPONSE_TIME_MAX = 280.0
GITHUB_SCORE_MIN = 0.0;     GITHUB_SCORE_MAX = 96.9
NOTICE_PERIOD_SCORES = {0:1.0, 15:1.0, 30:1.0, 45:0.90, 60:0.85, 90:0.70, ...}
TENURE_IDEAL_MONTHS = 24; ... TENURE_SCORE_UNKNOWN = 0.75
EDUCATION_TIER_SCORES = {1:1.0, 2:0.85, 3:0.70, 4:0.55}
CAREER_TOKEN_BUDGET = {0:125, 1:94, 2:62, 3:31}
```

- **`TIER_1_CITIES` / `CONSULTING_FIRMS`** — Python **sets** (O(1) membership tests). Sets, not
  lists, because they're used for `in` checks. Note both **Bangalore and Bengaluru** are
  included (same city, two spellings) — real-world normalization. Also imported by `jd_parser`
  for shared vocabulary.
- **Normalization ranges** (`*_MIN`/`*_MAX`) — empirically observed min/max from EDA, used to
  min-max scale raw values into [0, 1] in the Track 1 scorers. Centralizing them here means the
  scoring math and the observed data ranges stay in sync.
- **`NOTICE_PERIOD_SCORES`** — a lookup table mapping notice-period days → a score; shorter
  notice = higher score (more available sooner).
- **`TENURE_*`** — thresholds + scores for the "title-chasing / job-hopping" check (ideal ~24
  months per role). `TENURE_SCORE_UNKNOWN = 0.75` is the neutral default when tenure can't be
  computed.
- **`EDUCATION_TIER_SCORES`** — maps the numeric tier (from `get_education_tier`) to a
  credibility contribution.
- **`CAREER_TOKEN_BUDGET`** — a Track-2 constant: how many tokens to spend describing each job by
  recency (most recent job gets 125 tokens, oldest gets 31). This shapes the text that gets
  embedded — recent experience matters more, so it gets more of the token budget.
- **Interview:** why sets for city/firm lookups; what min-max normalization ranges are for; why
  keep all tuning constants in one file (single source of truth, easy to tune).

---

## Big-picture takeaways

1. **Accessor / anti-corruption layer** — no scorer touches the raw dict; all reads go through
   typed, defaulted accessors, so schema quirks (like `willing_to_relocate` living in
   `redrob_signals`) are hidden in one place.
2. **`... or default` everywhere** — handles both missing keys *and* explicit `null`s, and
   guarantees a usable type so downstream math/strings never see `None`.
3. **Pessimistic defaults** — missing availability/credibility data defaults toward the
   unfavorable end so incomplete profiles aren't over-credited.
4. **The `-1` GitHub sentinel** distinguishes "no account" from "score 0."
5. **All tuning constants in one place**, many derived from EDA, shared with `jd_parser`.
