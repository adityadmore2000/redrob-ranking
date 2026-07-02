# `honeypot_check.py` — Deep Line-by-Line Teaching Notes

> **A data-integrity auditor.** "Honeypots" here means **planted inconsistencies** — fake
> or exaggerated profile data the challenge may have seeded to catch naive rankers. This
> script re-runs the ranking, takes the top 100, and flags any of them whose profile shows
> **internal contradictions** (impossible skill claims, inflated experience). It's a
> *quality/red-team check on the output*, not part of the pipeline.
>
> Note: this is a **flat top-level script** (no functions, runs on import) — simpler and
> more ad-hoc than the pipeline modules.

---

## Part 1 — Load artifacts + recompute the ranking

```python
import json
import numpy as np
from datetime import datetime

candidate_ids       = np.load("artifacts/candidate_ids.npy", allow_pickle=True)
semantic_scores     = np.load("artifacts/semantic_scores.npy")
hard_filter_scores  = np.load("artifacts/hard_filter_scores.npy")
availability_scores = np.load("artifacts/availability_scores.npy")
credibility_scores  = np.load("artifacts/credibility_scores.npy")

semantic_norm = (semantic_scores - semantic_scores.min()) / (semantic_scores.max() - semantic_scores.min())
final_scores = semantic_norm * hard_filter_scores * availability_scores * credibility_scores
top100_idx = np.argsort(-final_scores)[:100]
top100_ids = set(candidate_ids[top100_idx])
rank_lookup = {candidate_ids[idx]: int(r) + 1 for r, idx in enumerate(top100_idx)}
```

- Loads the same Phase 1 artifacts and **re-derives** the ranking with the same min-max +
  multiplicative formula as `rank.py`.
- **`np.argsort(-final_scores)[:100]`** — `argsort` returns indices that would sort ascending;
  negating the scores flips it to **descending**, then slice the top 100. (Simpler than `rank.py`'s
  `lexsort`, but note: **no tie-breakers** here — fine for an audit, not for the official
  submission.)
- **`candidate_ids[top100_idx]`** — NumPy **fancy indexing**: index an array with an array of
  indices to pull those elements. Wrapped in `set()` for O(1) membership.
- `rank_lookup` — a dict comprehension mapping id → 1-based rank via `enumerate`.
- **Interview:** `argsort` + negation for descending; fancy indexing; why this differs from
  `lexsort` (no tie-break).

---

## Part 2 — Stream raw profiles for the top 100

```python
profiles = {}
with open("data/candidates.jsonl", "r") as f:
    for line in f:
        c = json.loads(line)
        cid = c.get("candidate_id") or c.get("id")
        if cid in top100_ids:
            profiles[cid] = c
```

- Streams the JSONL, keeping only the top-100 profiles (same memory-frugal idea as `rank.py`).
- `c.get("candidate_id") or c.get("id")` — tolerates either key name.

---

## Part 3 — The two honeypot checks

```python
current_year = datetime.now().year
flagged = []
for cid, c in profiles.items():
    issues = []

    # check 1 — expert/advanced skill with 0 months duration
    for skill in c.get("skills", []):
        proficiency = (skill.get("proficiency") or "").lower()
        duration_months = skill.get("duration_months")
        name = skill.get("name", "unknown")
        if proficiency in ("expert", "advanced") and duration_months == 0:
            issues.append(f"skill '{name}': {proficiency} proficiency but 0 months used")

    # check 2 — stated YOE vs actual career span
    profile = c.get("profile", {})
    yoe = profile.get("years_of_experience") or profile.get("yoe")
    career = c.get("career_history", [])
    if yoe and career:
        start_dates = []
        for job in career:
            sd = (job.get("start_date") or "")
            if sd:
                try:
                    start_dates.append(int(str(sd)[:4]))
                except:
                    pass
        if start_dates:
            earliest = min(start_dates)
            career_span = current_year - earliest
            if yoe > career_span + 2:
                issues.append(f"stated YOE {yoe} but career history only spans {career_span} years ...")
```

- **Check 1 — impossible expertise:** claiming **expert/advanced** proficiency but **0 months** of
  usage is a logical contradiction → a honeypot signal.
- **Check 2 — inflated experience:** parse the **year** from each job's `start_date`
  (`int(str(sd)[:4])` = first 4 chars), find the earliest, compute career span vs *now*. If
  **stated YOE exceeds the actual span by more than 2 years** (`yoe > career_span + 2`, a tolerance
  for gaps/rounding), flag it.
- The bare `except:` (catch-everything) around the year parse is sloppy style (a linter would warn
  `E722 bare except`), but pragmatic for a throwaway audit — it just skips unparseable dates.
- **Interview:** what internal-consistency checks catch (fabricated data); why a ±2 year tolerance;
  why a bare `except` is discouraged.

---

## Part 4 — Attach context + report + export

```python
    if issues:
        score_idx = np.where(candidate_ids == cid)[0][0]
        flagged.append({"candidate_id": cid, "rank": rank_lookup[cid],
                        "final_score": round(float(final_scores[score_idx]), 6),
                        ..., "honeypot_signals": issues, "raw_profile": c})
...
for f in sorted(flagged, key=lambda x: x["rank"]):
    print(f"Rank #{f['rank']:>3}  {f['candidate_id']} ...")
    for issue in f["honeypot_signals"]:
        print(f"           ⚠  {issue}")
...
with open("artifacts/flagged_honeypots.jsonl", "w") as out:
    for f in sorted(flagged, key=lambda x: x["rank"]):
        out.write(json.dumps(f) + "\n")
```

- **`np.where(candidate_ids == cid)[0][0]`** — find the array index for a given id: `candidate_ids
  == cid` is a boolean mask, `np.where(mask)` returns the matching indices, `[0][0]` takes the
  first. Used to fetch that candidate's scores for context.
- Each flagged candidate is enriched with rank + all sub-scores + the issues + the raw profile.
- Prints a sorted report (with ⚠ per issue), then **exports to `artifacts/flagged_honeypots.jsonl`**
  (one JSON object per line) for offline review.
- **Interview:** `np.where` for index lookup; boolean-mask indexing.

---

## Big-picture takeaways

1. **Red-team the output** — after ranking, audit the top 100 for **fabricated/contradictory data**
   that a keyword-matcher might reward.
2. **Two consistency rules** — expert-skill-with-zero-duration, and stated-YOE-exceeds-career-span.
3. **Recomputes the ranking independently** with the same formula (a cross-check on `rank.py`), but
   uses the simpler `argsort` without tie-breakers.
4. **Ad-hoc script style** — flat top-level code, bare `except`, hardcoded paths — appropriate for a
   one-off audit, and a useful contrast to the disciplined pipeline modules.
