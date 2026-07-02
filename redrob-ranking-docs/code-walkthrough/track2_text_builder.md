# `phase1/track2_text_builder.py` — Deep Line-by-Line Teaching Notes

> **Track 2, step 1: turn a candidate (and the JD) into embedding-ready text.** The
> *quality of this text directly determines the quality of the semantic match* — garbage
> in, garbage out. Every choice here (field order, recency-weighted truncation, dropping
> beginner skills, instruction prefixes) is about **shaping what the embedding pays
> attention to**.

**Public API:** `build_candidate_text(c) -> str`, `build_jd_text(jd) -> str`.

---

## Part 1 — Docstring: the design philosophy

Read this docstring as the *rationale* for the whole file:

- **Field order = importance order:** current role → career history (recency-weighted) → summary
  → skills → education. Embeddings are influenced more by earlier/prominent content, so the
  strongest signals go first.
- **Recency weighting:** the most recent role is the best predictor of current fit, so recent
  jobs get a **generous token budget** and older ones are compressed — preventing a long history
  from *diluting* the present-day signal.
- **Skills are subordinate & beginner skills dropped:** skill lists are noisy/keyword-like (the JD
  explicitly warns against keyword matching), so they come after the narrative and only
  intermediate+ proficiency is included.
- **Certifications excluded:** sparse, low-signal, not worth the token budget.
- **Interview point:** this is *feature engineering for embeddings* — you're deciding what the
  vector represents by curating and ordering the text.

---

## Part 2 — Imports + constants

```python
from field_map import (get_current_title, get_current_company, get_current_industry, get_summary,
                       get_career_sorted, get_skill_name_proficiency, get_education_entries,
                       CAREER_TOKEN_BUDGET)
CHARS_PER_TOKEN = 4
_PROFICIENCY_RANK = {'expert': 3, 'advanced': 2, 'intermediate': 1}
```

- **`CHARS_PER_TOKEN = 4`** — a cheap approximation (English text averages ~4 chars/token) used to
  budget text length *without* running the real tokenizer. Matches the EDA methodology. This lets
  truncation be a fast string slice instead of a tokenizer call.
- **`_PROFICIENCY_RANK`** — the allowlist *and* sort order for skills. **Beginner is absent** →
  beginner skills are excluded (filtered out by `prof in _PROFICIENCY_RANK`). `expert` ranks above
  `advanced` so top skills sort first.
- **Interview:** why approximate tokens by char count (speed, no tokenizer dependency); how a dict
  doubles as both a filter and a ranking.

---

## Part 3 — `_truncate_to_tokens` (word-safe truncation)

```python
def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    if not text:
        return ''
    max_chars = max_tokens * CHARS_PER_TOKEN
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    if ' ' in truncated:
        truncated = truncated.rsplit(' ', 1)[0]
    return truncated.rstrip()
```

- Convert a token budget to a char budget, slice, then **back off to the last whole word** so we
  never cut mid-word (`rsplit(' ', 1)[0]` drops the trailing partial word). `.rstrip()` trims
  trailing whitespace.
- Cutting mid-word would create junk sub-tokens that pollute the embedding — hence the word-safe
  backoff.
- **Interview:** `rsplit(sep, maxsplit)` semantics; why word-boundary truncation matters for
  embeddings.

---

## Part 4 — `_build_career_block` (recency weighting in action)

```python
def _build_career_block(career_entries: list) -> str:
    lines = []
    max_budget_idx = max(CAREER_TOKEN_BUDGET)   # 3 -> title/company only
    for i, job in enumerate(career_entries):
        title = job.get('title', ''); company = job.get('company', '')
        if i >= max_budget_idx:
            lines.append(f"{title} at {company}")
            continue
        duration = job.get('duration_months', 0)
        description = job.get('description', '') or ''
        budget = CAREER_TOKEN_BUDGET[i]
        desc = _truncate_to_tokens(description, budget)
        if desc:
            lines.append(f"{title} at {company} ({duration} months): {desc}")
        else:
            lines.append(f"{title} at {company} ({duration} months)")
    return '\n'.join(lines)
```

- `career_entries` come from `get_career_sorted` → **index 0 is the most recent job**.
- `CAREER_TOKEN_BUDGET` (from `field_map`) = `{0:125, 1:94, 2:62, 3:31}`. **`max(CAREER_TOKEN_BUDGET)`
  returns the largest *key* (3)** — jobs at index ≥ 3 get only `"{title} at {company}"` (no
  description). This is the recency weighting: recent jobs verbose, older jobs one-liners.
- For jobs 0–2, the description is truncated to that position's budget; the line includes duration
  and (if present) the truncated description.
- **Interview subtlety:** `max()` on a dict iterates its **keys**, so `max(CAREER_TOKEN_BUDGET)` = 3,
  the "everything beyond here is minimal" threshold. Explaining that shows you understand dict
  iteration.

---

## Part 5 — `_build_skills_block` (filter + multi-key sort)

```python
def _build_skills_block(c: dict) -> str:
    kept = [(name, prof, duration)
            for (name, prof, duration) in get_skill_name_proficiency(c)
            if prof in _PROFICIENCY_RANK and name]
    kept.sort(key=lambda t: (_PROFICIENCY_RANK[t[1]], t[2] or 0), reverse=True)
    return ', '.join(f"{name} ({prof})" for (name, prof, _) in kept)
```

- Comprehension filters to allowed proficiencies (drops beginner and unnamed skills).
- **`sort(key=lambda t: (rank, duration), reverse=True)`** — sorts by a **tuple key**: primary =
  proficiency rank, secondary (tie-break) = duration. Tuple keys are the idiomatic Python way to
  do multi-level sorting; `reverse=True` puts expert/longest first. `t[2] or 0` guards a missing
  duration.
- Output: `"Python (expert), SQL (advanced), ..."`.
- **Interview gold:** multi-key sorting with tuple keys; why the secondary key breaks ties
  deterministically.

---

## Part 6 — `_build_education_block` and `build_candidate_text`

```python
def _build_education_block(entries: list) -> str:
    parts = [f"{e.get('degree','')} in {e.get('field_of_study','')} from {e.get('institution','')} ({e.get('tier','')})"
             for e in entries]
    return '; '.join(parts)

def build_candidate_text(c: dict) -> str:
    ...
    sections = [
        f"Current role: {title} at {company} ({industry})",
        f"Career history:\n{career_block}",
        f"Summary: {summary}",
        f"Skills: {skills_block}",
        f"Education: {education_block}",
    ]
    text = '\n\n'.join(sections)
    instruction = "Represent this candidate profile for matching to a job description: "
    return instruction + text
```

- Assembles the five labeled sections (in importance order) joined by blank lines.
- **The instruction prefix** `"Represent this candidate profile for matching to a job
  description: "` — see Part 7; this is a BGE-specific technique, not decoration.

---

## Part 7 — `build_jd_text` + the instruction-prefix technique

```python
def build_jd_text(jd: dict) -> str:
    instruction = "Represent this job description for retrieving matching candidate profiles: "
    return instruction + jd['jd_text']
```

- **The single most important embedding-quality detail (interview gold):** BGE models are trained
  with **instruction prefixes** for **asymmetric retrieval** (query vs document are phrased
  differently). Prefixing the query (JD) and document (candidate) with role-appropriate
  instructions makes the model produce **more discriminative similarity scores** for
  query→document matching. The candidate gets a "represent this profile *for matching*" prefix;
  the JD gets a "represent this JD *for retrieving candidates*" prefix — a deliberate
  **asymmetric** pairing.
- **Interview:** what asymmetric retrieval is; why instruction-tuned embedding models want a task
  prefix; why the two sides use *different* prefixes.

---

## Part 8 — `__main__` diagnostic

```python
if __name__ == '__main__':
    ... load up to LIMIT=5 candidates (jsonl → sample → SystemExit) ...
    def approx_tokens(text): return len(text) // CHARS_PER_TOKEN
    for c in candidates:
        text = build_candidate_text(c)
        ... print the text + approx token counts per block ...
    try:
        from jd_parser import JD
        jd_text = build_jd_text(JD)
        print(jd_text[:300])
    except Exception as exc:  # pragma: no cover
        print(f"Could not load JD text (...)")
```

- Prints the built text for 5 candidates plus **approximate token counts per block** — the dev's
  way to *see* the recency budgeting working and confirm nothing blows the budget.
- **`from jd_parser import JD` is done lazily inside the try** — so building *candidate* text
  doesn't require `python-docx` (only the JD path does). Sensible dependency isolation.
- `# pragma: no cover` — tells coverage tools to ignore this diagnostic-only branch.
- **Interview:** why isolate the docx dependency to the JD path; what `# pragma: no cover` does.

---

## Big-picture takeaways

1. **Text construction is feature engineering for embeddings** — order, curation, and truncation
   decide what the vector "means."
2. **Recency weighting via token budget** — recent jobs verbose, old jobs one-liners, so present
   trajectory dominates.
3. **Noise control** — skills are subordinate, beginner skills and certifications dropped, per the
   JD's warning against keyword matching.
4. **Word-safe, tokenizer-free truncation** using a 4-chars/token approximation.
5. **BGE instruction prefixes for asymmetric retrieval** — the key to discriminative JD↔candidate
   scores.
