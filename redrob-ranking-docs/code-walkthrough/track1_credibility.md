# `phase1/track1_credibility.py` — Deep Line-by-Line Teaching Notes

> **Track 1, signal #3: credibility / trust.** How much can we trust this profile —
> completeness, endorsements, education tier, GitHub activity, LinkedIn, verified
> contact. Structurally a twin of `track1_availability.py`: six sub-signals combined by
> **weighted average**, same `compute_*_score`/`compute_all` interface, same diagnostic
> `__main__`. This doc focuses on what's *distinctive* here.

```
credibility_score = Σ (signal_i × weight_i)   with weights summing to 1.0
```

---

## Part 1 — Docstring: another EDA-driven exclusion + a "nudge"

Weights: profile_completeness 0.25, endorsements 0.20, education_tier 0.20, github 0.20,
linkedin 0.10, verified_contact 0.05.

- **`skill_assessment_scores` is excluded as a standalone signal** — only ~24.2% of candidates
  have it, too **sparse** to weight directly. Instead it's **folded into profile_completeness as a
  small upward nudge** for candidates who *do* have it. This is a clean way to use a sparse
  feature without letting its absence unfairly penalize the 76% who lack it.
- **Interview:** how to incorporate a sparse feature — not as its own weighted term (its absence
  would look like a low score) but as a bonus/nudge on a related dense feature.

---

## Part 2 — Imports + constants

```python
import math
from field_map import (get_candidate_id, get_profile_completeness, get_skill_assessment_scores,
                       get_endorsements, get_education_tier, get_has_github, get_github_score,
                       get_linkedin_connected, get_verified_email, get_verified_phone,
                       EDUCATION_TIER_SCORES, GITHUB_SCORE_MIN, GITHUB_SCORE_MAX)

WEIGHTS = {'profile_completeness':0.25, 'endorsements':0.20, 'education_tier':0.20,
           'github':0.20, 'linkedin':0.10, 'verified_contact':0.05}
PROFILE_COMPLETENESS_MIN = 25; PROFILE_COMPLETENESS_MAX = 100
ENDORSEMENTS_MAX = 242

def _clamp(x, lo=0.0, hi=1.0): return max(lo, min(hi, x))
```

- `import math` — needed for `math.log` (log-scaling below).
- `ENDORSEMENTS_MAX = 242` — the observed long-tail max, used as the log-scale denominator.
- `_clamp` — identical helper (note: `_clamp` is re-defined in each Track 1 module rather than
  shared — mild duplication that keeps each module self-contained).

---

## Part 3 — `profile_completeness_signal` (with the sparse-feature nudge)

```python
def profile_completeness_signal(c):
    raw = get_profile_completeness(c)
    base = (raw - PROFILE_COMPLETENESS_MIN) / (PROFILE_COMPLETENESS_MAX - PROFILE_COMPLETENESS_MIN)
    base = _clamp(base)
    assessments = get_skill_assessment_scores(c)
    if assessments:
        avg_assessment = sum(assessments.values()) / len(assessments)
        nudge = (avg_assessment / 100) * 0.05
        return min(1.0, base + nudge)
    return base
```

- **`base`** = min-max normalization of completeness within its observed [25, 100] range → [0, 1].
- **The nudge:** if the candidate has skill-assessment scores, average them, scale to a **max
  +0.05 bonus** (`(avg/100) * 0.05`), add to base, capped at 1.0. So assessment data can lift a
  profile slightly but its absence costs nothing.
- **Interview:** why cap the nudge and why `min(1.0, ...)` (never exceed the normalized ceiling).

---

## Part 4 — `endorsements_signal` (log-scaling the long tail)

```python
def endorsements_signal(c):
    n = get_endorsements(c)
    return _clamp(math.log(1 + n) / math.log(1 + ENDORSEMENTS_MAX))
```

- **The interview-worthy technique: log-scaling.** Endorsement counts are **heavy-tailed** (most
  people have few, a handful have hundreds). Linear normalization would squash almost everyone
  into a tiny range near 0. **`log(1 + n) / log(1 + MAX)`** compresses the tail so differences at
  the low end (5 vs 20 endorsements) matter more than at the high end (200 vs 240).
- **`log(1 + n)` not `log(n)`** — the `1 +` handles `n = 0` gracefully (`log(1) = 0`) and avoids
  `log(0) = -inf`. This "log1p" trick is standard.
- Dividing by `log(1 + MAX)` normalizes the result to ~[0, 1].
- **Interview gold:** when/why to log-scale a feature (skewed/long-tailed distributions); the
  `log(1 + x)` trick for zero-safety.

---

## Part 5 — `education_tier_signal` and `github_signal`

```python
def education_tier_signal(c):
    tier = get_education_tier(c)
    return EDUCATION_TIER_SCORES.get(tier, EDUCATION_TIER_SCORES[4])

def github_signal(c):
    if not get_has_github(c):
        return 0.0
    score = get_github_score(c)
    norm = (score - GITHUB_SCORE_MIN) / (GITHUB_SCORE_MAX - GITHUB_SCORE_MIN)
    return _clamp(norm)
```

- `education_tier_signal` — table lookup; `.get(tier, EDUCATION_TIER_SCORES[4])` **defaults to the
  worst tier's score** if the tier is unrecognized (conservative).
- `github_signal` — **uses the `-1` sentinel logic from `field_map`**: `get_has_github(c)`
  distinguishes "no account" (→ 0.0) from a real score of 0. Only real scores get min-max
  normalized. This is exactly why the accessor layer split `get_has_github` / `get_github_score`.
- **Interview:** why "no GitHub → 0.0" is *correct* here but relies on the sentinel to not confuse
  "no account" with "account, no activity."

---

## Part 6 — `linkedin_signal` and `verified_contact_signal`

```python
def linkedin_signal(c):
    return 1.0 if get_linkedin_connected(c) else 0.0

def verified_contact_signal(c):
    verified = int(bool(get_verified_email(c))) + int(bool(get_verified_phone(c)))
    if verified == 2: return 1.0
    if verified == 1: return 0.6
    return 0.2
```

- `verified_contact_signal` — **`int(bool(x))` coerces each flag to 0/1** and sums them, giving a
  clean count of 0/1/2 verified channels → mapped to 0.2/0.6/1.0. Neat idiom for "count how many
  of these booleans are true."
- **Interview:** the `int(bool(...))` counting idiom; a tiered mapping from a small integer count.

---

## Part 7 — Combination + `compute_all` + `__main__`

```python
credibility = sum(signals[k] * WEIGHTS[k] for k in WEIGHTS)
```
- Same weighted-average combination and same transparent return-all-sub-signals pattern as
  availability. `compute_all` has the identical interface.
- The `__main__` block is the same distribution-table diagnostic (min/p25/median/p75/max +
  3 sample candidates), with credibility-specific fields printed (including the computed
  skill-assessment average per sample).

---

## Big-picture takeaways

1. **Weighted average of six trust signals** — mirrors availability's structure and philosophy.
2. **Log-scaling for heavy-tailed endorsements** — the standout technique; `log(1 + n)` for
   zero-safety and tail compression.
3. **Sparse feature handled as a nudge, not a term** — skill assessments fold into
   profile_completeness so their absence doesn't penalize the majority.
4. **Sentinel-aware GitHub scoring** — "no account" (0.0) is distinct from "account, zero
   activity," relying on `field_map`'s `-1` sentinel accessors.
5. **Small idioms worth knowing** — `int(bool(...))` boolean counting, conservative
   `.get(tier, worst)` defaults, per-module `_clamp`.
