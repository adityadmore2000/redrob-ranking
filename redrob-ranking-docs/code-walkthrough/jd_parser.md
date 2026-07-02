# `jd_parser.py` — Deep Line-by-Line Teaching Notes

> **Rule-based (no-NLP) parser that turns a `.docx` job description into structured
> requirements.** It reads `job_description.docx`, extracts plain text, then uses only
> `re` (regex) + string matching to derive: min/max years of experience, preferred
> locations, work modes, disqualified consulting firms, required skills, and seniority.
> Parsed **once on import** and exposed as the `JD` dict.

**The design principle (interview-worthy):** *nothing is hardcoded to a specific JD.* The
constants are **recognition vocabularies** ("what to look for"), and a term is only emitted
if it *actually appears* in the JD text. Swap the docx, get a different `JD` — no code change.

---

## Part 1 — Docstring + imports

```python
import re
from field_map import TIER_1_CITIES, CONSULTING_FIRMS
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
```

- `re` — Python's **regular-expression** engine, the only "parsing" tool used (deliberately no
  spaCy/NLTK — keeps dependencies light and behavior predictable).
- Imports `TIER_1_CITIES` and `CONSULTING_FIRMS` **from `field_map`** — the *same* vocabularies
  the scorers use, so the JD parser and the candidate scorers share one source of truth (a city
  the JD recognizes is the same set a candidate is matched against).
- `import os as _os` with a leading underscore alias — cosmetic; marks `os` as module-private so
  it isn't re-exported when someone does `from jd_parser import *`.
- `_HERE` — CWD-robust base dir (same `__file__` pattern seen elsewhere).

---

## Part 2 — Recognition vocabularies

```python
WORK_MODE_KEYWORDS = {'hybrid':'hybrid', 'remote':'remote', 'onsite':'onsite',
                      'on-site':'onsite', 'in-office':'onsite', 'flexible':'flexible'}
SKILL_TERMS = ['embeddings', 'retrieval', 'hybrid retrieval', ..., 'bge', 'faiss', 'bm25',
               'llms', 'lora', 'python', 'ndcg', 'mrr', ...]
SENIORITY_KEYWORDS = [('senior','senior'), ('staff','senior'), ('lead','senior'),
                      ('mid-level','mid'), ('junior','junior'), ('associate','junior')]
```

- **`WORK_MODE_KEYWORDS`** — maps surface spellings → a **canonical label** (`on-site`,
  `in-office` both normalize to `onsite`). Emitting the canonical label means downstream code
  deals with one spelling.
- **`SKILL_TERMS`** — a flat list of IR/ML/embedding technologies to detect. Only those present
  in the JD are returned.
- **`SENIORITY_KEYWORDS`** — a **list of tuples** (not a dict) because **order matters**: it's
  scanned top-to-bottom and `senior` variants come first so "senior" wins over "associate" if
  both somehow appear. A list preserves that priority order; a dict wouldn't guarantee it as
  clearly.
- **Interview:** why a dict for work-modes (many spellings → one label) but an *ordered list* for
  seniority (priority matters).

---

## Part 3 — Text extraction + normalization

```python
def extract_text_from_docx(path):
    from docx import Document
    doc = Document(path)
    return '\n'.join([para.text for para in doc.paragraphs if para.text.strip()])
```

- Uses **`python-docx`** (`from docx import Document`) — lazy-imported inside the function so the
  heavy dependency loads only when actually parsing a docx.
- Reads all paragraphs, drops blank ones (`if para.text.strip()`), joins with newlines → one
  plain-text blob.

```python
def _normalize(text):
    return text.replace('–', '-').replace('—', '-').lower()
```

- **Normalizes dash variants** (en-dash `–`, em-dash `—` → plain hyphen) and lowercases. This is
  essential: Word documents love "smart" typographic dashes, so `"5–9 years"` must be turned into
  `"5-9 years"` for the range regex to match. A classic real-world-text gotcha.
- **Interview:** why normalize dashes — Word auto-converts hyphens to en/em-dashes, breaking naive
  regex.

---

## Part 4 — `extract_yoe` (years of experience)

```python
def extract_yoe(text):
    norm = _normalize(text)
    range_mins, range_maxs = [], []
    for m in re.finditer(r'(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*\+?\s*years?', norm):
        range_mins.append(float(m.group(1)))
        range_maxs.append(float(m.group(2)))
    if range_mins:
        return min(range_mins), max(range_maxs)

    mins, maxs = [], []
    for m in re.finditer(r'(\d+(?:\.\d+)?)\s*\+\s*years?', norm):
        mins.append(float(m.group(1)))
    for m in re.finditer(r'(?:minimum|at least)\s+(?:of\s+)?(\d+(?:\.\d+)?)\s*years?', norm):
        mins.append(float(m.group(1)))
    for m in re.finditer(r'(?:maximum|up to)\s+(?:of\s+)?(\d+(?:\.\d+)?)\s*years?', norm):
        maxs.append(float(m.group(1)))
    yoe_min = min(mins) if mins else None
    yoe_max = max(maxs) if maxs else None
    return yoe_min, yoe_max
```

- **A precedence rule with a real reason:** if any explicit **range** (`"5-9 years"`) exists, it
  is *authoritative* and open-ended cues are ignored. Why? To avoid false positives from phrases
  like *"plans to be here for 3+ years"* (tenure, not required experience). Clever domain-aware
  parsing.
- **Regex breakdown** `(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*\+?\s*years?`:
  - `(\d+(?:\.\d+)?)` — an integer or decimal, **captured** (group 1). `(?:...)` is a
    *non-capturing* group (grouping without a back-reference).
  - `\s*-\s*` — a hyphen with optional surrounding whitespace.
  - second `(\d+(?:\.\d+)?)` — the upper bound (group 2).
  - `\s*\+?` — an optional `+`; `years?` — "year" or "years" (`?` = optional `s`).
- `re.finditer` yields **all** matches (there can be several); `m.group(1)`/`group(2)` pull the
  captured numbers; the **widest** span is returned (`min` of mins, `max` of maxs) so outer bounds
  are respected.
- Fallback cues use non-capturing alternations `(?:minimum|at least)` and optional `(?:of\s+)?`.
- Missing bound → `None`.
- **Interview gold:** capturing vs non-capturing groups; `finditer` vs `search`/`findall`; why
  ranges beat "X+ years"; `?` and `\s*` semantics.

---

## Part 5 — `extract_locations`

```python
def extract_locations(text):
    norm = _normalize(text)
    found = []
    for city in sorted(TIER_1_CITIES):
        if re.search(r'\b' + re.escape(city) + r'\b', norm):
            found.append(city)
    if re.search(r'\btier[\s-]?1\b', norm):
        found.append('tier-1')
    seen, ordered = set(), []
    for item in found:
        if item not in seen:
            seen.add(item); ordered.append(item)
    return ordered
```

- Iterates the shared `TIER_1_CITIES` set and keeps those that appear.
- **`r'\b' + re.escape(city) + r'\b'`** — the interview-critical idiom:
  - `\b` = **word boundary** — ensures whole-word matches (`"pune"` won't match inside
    `"puneet"`).
  - `re.escape(city)` — escapes regex-special characters in the term so it's matched literally
    (defensive even if a city name had a `.`), and safe if the vocab ever changes.
- Also detects a literal "tier-1"/"tier 1" mention via `tier[\s-]?1` (`[\s-]?` = optional space
  or hyphen).
- The `seen`/`ordered` loop **de-duplicates while preserving first-appearance order** (a set
  alone would lose order).
- **Interview:** what `\b` does; why `re.escape`; how to dedupe while preserving order.

---

## Part 6 — `extract_work_modes`

```python
def extract_work_modes(text):
    norm = _normalize(text)
    found, seen = [], set()
    for keyword, label in WORK_MODE_KEYWORDS.items():
        if re.search(r'\b' + re.escape(keyword) + r'\b', norm) and label not in seen:
            seen.add(label); found.append(label)
    return found
```

- Emits **canonical labels**, deduped by label (so `on-site` and `in-office` both → one
  `onsite`).

---

## Part 7 — `extract_disqualified_companies` (context-aware)

```python
def extract_disqualified_companies(text):
    norm = _normalize(text)
    found = [firm for firm in sorted(CONSULTING_FIRMS) if re.search(r'\b'+re.escape(firm)+r'\b', norm)]
    if not found:
        return []
    negative_cues = ('do not want', "don't want", 'not want', 'consulting firm',
                     'bad fit', 'not a fit', 'disqualif', 'only worked')
    if any(cue in norm for cue in negative_cues):
        return found
    return []
```

- **Two-stage detection** — presence *and context*. A consulting firm name alone isn't a
  disqualifier; it only counts if a **negative cue** ("do not want", "consulting firm", "bad
  fit", …) also appears in the JD. This avoids wrongly disqualifying candidates when a firm is
  mentioned neutrally.
- `disqualif` (truncated) matches "disqualify/disqualified/disqualification" in one cue.
- **Interview:** why context matters — a keyword's presence ≠ its intent; two-stage guard reduces
  false positives.

---

## Part 8 — `extract_required_skills` (longest-match-first)

```python
def extract_required_skills(text):
    norm = _normalize(text)
    found, seen = [], set()
    for term in sorted(SKILL_TERMS, key=len, reverse=True):
        pattern = r'(?<!\w)' + re.escape(term.lower()) + r'(?!\w)'
        if re.search(pattern, norm) and term not in seen:
            seen.add(term); found.append(term)
    return found
```

- **`sorted(SKILL_TERMS, key=len, reverse=True)`** — sort **longest term first** so
  `"hybrid retrieval"` is recognized before the substring `"retrieval"`. Prevents a longer,
  more-specific phrase from being shadowed by its shorter component.
- **`(?<!\w)...(?!\w)`** — lookarounds used *instead of* `\b`:
  - `(?<!\w)` = negative **lookbehind** ("not preceded by a word char").
  - `(?!\w)` = negative **lookahead** ("not followed by a word char").
  - Why not `\b`? Because many skill terms contain non-word characters (`a/b test`, `on-site`,
    `c++`-style), where `\b` behaves unintuitively at the boundary between word and non-word
    chars. Lookarounds give a cleaner "whole token" match for terms with slashes/dashes.
- **Interview gold:** longest-match-first ordering; lookbehind/lookahead vs `\b`, and *why*
  they're used for terms containing punctuation.

---

## Part 9 — `extract_seniority`

```python
def extract_seniority(text):
    norm = _normalize(text)
    title = norm.split('\n', 1)[0]
    for scope in (title, norm):
        for keyword, level in SENIORITY_KEYWORDS:
            if re.search(r'\b' + re.escape(keyword) + r'\b', scope):
                return level
    return 'mid'
```

- **Scope preference:** check the **first line (the title)** first, then the whole body. A
  seniority word in the title is a stronger signal than one buried in the description.
- `norm.split('\n', 1)[0]` — split on the first newline only (`maxsplit=1`) and take the first
  piece = the title line.
- Returns the **first** matching level (relies on the ordered `SENIORITY_KEYWORDS`), defaulting
  to `'mid'` if nothing matches.
- **Interview:** why check the title first (signal strength); `split(sep, maxsplit)` semantics.

---

## Part 10 — `parse_jd` + module-level `JD`

```python
def parse_jd(jd_path=None):
    if jd_path is None:
        jd_path = _os.path.join(_HERE, 'job_description.docx')
    jd_text = extract_text_from_docx(jd_path)
    yoe_min, yoe_max = extract_yoe(jd_text)
    return {
        'yoe_min': yoe_min, 'yoe_max': yoe_max,
        'preferred_locations': extract_locations(jd_text),
        'preferred_work_modes': extract_work_modes(jd_text),
        'disqualified_companies': extract_disqualified_companies(jd_text),
        'jd_text': jd_text,
        'required_skills': extract_required_skills(jd_text),
        'seniority_level': extract_seniority(jd_text),
    }

JD = parse_jd()

if __name__ == '__main__':
    from pprint import pprint
    pprint(JD)
```

- `parse_jd` orchestrates all extractors into **one structured dict**. Default path anchored to
  `_HERE`. Note it keeps the raw `jd_text` too (needed by `build_jd_text` for embedding).
- **`JD = parse_jd()` at module level** — the docx is parsed **exactly once, at import time**.
  So `from jd_parser import JD` gives every consumer the same pre-computed result with **no
  repeated file I/O**. This is why `demo_pipeline` and the scorers can just import `JD`.
  - *Hidden cost/caveat:* import now has a side effect (reads a file). If the docx is missing,
    **importing the module raises** — a tradeoff for the convenience of a ready-made `JD`.
- **`if __name__ == '__main__':`** — the standard "run as a script" guard. Running
  `python jd_parser.py` pretty-prints the parsed `JD` for inspection/debugging; importing it does
  not. `pprint` = pretty-print (nicely formats the nested dict).
- **Interview:** what `if __name__ == '__main__'` means; the pro/con of parsing at import time
  (convenient shared singleton vs import-time side effect / failure).

---

## Big-picture takeaways

1. **Data-driven, not hardcoded** — vocabularies say *what to look for*; every returned value is
   extracted from the actual JD text, so swapping the docx changes the requirements with no code
   edits.
2. **Regex craftsmanship** — word boundaries / lookarounds for clean matching, longest-match-first
   for phrase precedence, dash normalization for Word docs, and context-aware disqualification.
3. **Shared vocabulary** — cities and consulting firms come from `field_map`, so the JD and the
   candidate scorers agree.
4. **Parsed once as `JD`** — an import-time singleton; convenient but couples import to file I/O.
5. **Domain awareness** — e.g. ranges override "X+ years", title beats body for seniority — rules
   that reflect how JDs are actually written.
