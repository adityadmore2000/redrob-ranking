"""
track2_text_builder.py
======================
Converts a candidate dict into a single text string optimized for embedding
with BGE-base-en-v1.5. The text is structured so the embedding captures career
trajectory and semantic fit — not raw keyword overlap.

Field order (most important signal first)
-----------------------------------------
1. Current role block
2. Career history (recency-weighted: recent jobs verbose, older compressed)
3. Summary
4. Skills (name + proficiency — a subordinate signal)
5. Education

Design decisions
----------------
- Recency weighting: a candidate's most recent role is the strongest predictor
  of current capability and fit, so recent jobs get a generous token budget and
  older ones are compressed. This biases the embedding toward present-day
  trajectory instead of letting a long career history dilute the signal.
- Skills are subordinate: skill lists are noisy and keyword-like (the JD
  explicitly warns against keyword matching). They follow the narrative blocks
  so the embedding anchors on what the candidate *did*, not a tag cloud.
- Beginner skills excluded: beginner proficiency is too weak a signal and adds
  noise; only intermediate and above are serialized.
- Certifications excluded: certifications are sparse, low-signal for semantic
  fit, and would consume token budget better spent on career narrative.

Token budget (4 chars/token approximation, matching EDA methodology)
--------------------------------------------------------------------
    job position  ->  max tokens
    0 (most recent)   125
    1                  94
    2                  62
    3+                 31 (title + company only, no description)

Public API
----------
build_candidate_text(c)  -> str  candidate text representation for embedding
build_jd_text(jd)        -> str  JD anchor text for embedding (no truncation)
"""

from field_map import (
    get_current_title,
    get_current_company,
    get_current_industry,
    get_summary,
    get_career_sorted,
    get_skill_name_proficiency,
    get_education_entries,
    CAREER_TOKEN_BUDGET,
)

# ~4 characters per token (EDA approximation).
CHARS_PER_TOKEN = 4

# Proficiency levels we serialize, ranked high -> low. Beginner is excluded.
# 'expert' is present in the data above 'advanced'; it is kept as the top tier
# so the strongest skills are not dropped or mis-ordered.
_PROFICIENCY_RANK = {
    'expert': 3,
    'advanced': 2,
    'intermediate': 1,
}


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    """
    Truncate text to approximately max_tokens.
    Uses 4 chars per token approximation (matches EDA methodology).
    Truncates at word boundary — never mid-word.
    """
    if not text:
        return ''
    max_chars = max_tokens * CHARS_PER_TOKEN
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    # Back off to the last whole word so we never cut mid-word.
    if ' ' in truncated:
        truncated = truncated.rsplit(' ', 1)[0]
    return truncated.rstrip()


def _build_career_block(career_entries: list) -> str:
    """
    Serialize career history with recency weighting.
    Uses CAREER_TOKEN_BUDGET from field_map for per-position token limits.

    Job 0 (most recent) — full description, truncated to 125 tokens
    Job 1               — medium detail, truncated to 94 tokens
    Job 2               — brief, truncated to 62 tokens
    Job 3+              — title and company only, no description

    Format per job:
        {title} at {company} ({duration_months} months): {description}

    For job 3+:
        {title} at {company}

    Joins all job blocks with newline.
    """
    lines = []
    max_budget_idx = max(CAREER_TOKEN_BUDGET)  # 3 -> title/company only
    for i, job in enumerate(career_entries):
        title = job.get('title', '')
        company = job.get('company', '')
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


def _build_skills_block(c: dict) -> str:
    """
    Serialize skills as: name (proficiency), name (proficiency), ...
    Only include advanced and intermediate proficiency skills (and expert,
    which ranks above advanced). Beginner skills excluded — too weak a signal.
    Sort by proficiency descending (advanced first) then duration_months
    descending.
    """
    kept = [
        (name, prof, duration)
        for (name, prof, duration) in get_skill_name_proficiency(c)
        if prof in _PROFICIENCY_RANK and name
    ]
    kept.sort(
        key=lambda t: (_PROFICIENCY_RANK[t[1]], t[2] or 0),
        reverse=True,
    )
    return ', '.join(f"{name} ({prof})" for (name, prof, _) in kept)


def _build_education_block(entries: list) -> str:
    """
    Serialize education entries.
    Format: {degree} in {field_of_study} from {institution} ({tier})
    Join multiple entries with '; '
    """
    parts = []
    for e in entries:
        degree = e.get('degree', '')
        field = e.get('field_of_study', '')
        institution = e.get('institution', '')
        tier = e.get('tier', '')
        parts.append(f"{degree} in {field} from {institution} ({tier})")
    return '; '.join(parts)


def build_candidate_text(c: dict) -> str:
    """
    Builds the full candidate text representation for embedding.

    Structure:
        Current role: {title} at {company} ({industry})

        Career history:
        {career_block}

        Summary: {summary}

        Skills: {skills_block}

        Education: {education_block}

    Returns a single string ready to be passed to the embedding model.
    """
    title = get_current_title(c)
    company = get_current_company(c)
    industry = get_current_industry(c)

    career_block = _build_career_block(get_career_sorted(c))
    summary = get_summary(c)
    skills_block = _build_skills_block(c)
    education_block = _build_education_block(get_education_entries(c))

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


def build_jd_text(jd: dict) -> str:
    """
    Builds the JD text for embedding from the parsed JD dict.
    Prefixes with BGE instruction for asymmetric retrieval — the model
    was trained with instruction prefixes and produces more discriminative
    scores when used correctly for query-document matching tasks.
    """
    instruction = "Represent this job description for retrieving matching candidate profiles: "
    return instruction + jd['jd_text']


if __name__ == '__main__':
    import json
    import os

    # Prefer the full dataset (JSONL, one record per line). Fall back to the
    # provided 50-record fixture (a single JSON array) when it isn't present.
    jsonl_path = os.path.join('data', 'candidates.jsonl')
    sample_path = os.path.join('data', 'sample_candidates.json')

    LIMIT = 5
    candidates = []
    if os.path.exists(jsonl_path):
        candidates_path = jsonl_path
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                candidates.append(json.loads(line))
                if len(candidates) >= LIMIT:
                    break
    elif os.path.exists(sample_path):
        candidates_path = sample_path
        with open(sample_path) as f:
            candidates = json.load(f)[:LIMIT]
    else:
        raise SystemExit(
            "No candidate data found. Expected data/candidates.jsonl "
            "(full JSONL dataset) or data/sample_candidates.json (sample array). "
            "Copy a dataset into the data/ directory before running."
        )

    def approx_tokens(text):
        return len(text) // CHARS_PER_TOKEN

    print(f"Loaded {len(candidates)} candidates from {candidates_path}")

    for c in candidates:
        text = build_candidate_text(c)
        career_block = _build_career_block(get_career_sorted(c))
        skills_block = _build_skills_block(c)
        education_block = _build_education_block(get_education_entries(c))
        summary = get_summary(c)

        print('=' * 72)
        print(text)
        print('-' * 72)
        print("Approx token counts per block:")
        print(f"  career    : {approx_tokens(career_block):4d} tokens")
        print(f"  summary   : {approx_tokens(summary):4d} tokens")
        print(f"  skills    : {approx_tokens(skills_block):4d} tokens")
        print(f"  education : {approx_tokens(education_block):4d} tokens")
        print(f"  TOTAL     : {approx_tokens(text):4d} tokens")

    # Confirm JD text loads. Imported lazily so candidate text building above
    # does not require python-docx.
    print('=' * 72)
    try:
        from jd_parser import JD
        jd_text = build_jd_text(JD)
        print("build_jd_text (first 300 chars):")
        print(jd_text[:300])
    except Exception as exc:  # pragma: no cover - diagnostic aid
        print(f"Could not load JD text ({type(exc).__name__}: {exc})")
