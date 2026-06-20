"""
jd_parser.py — dynamic, rule-based parser for the job description.

Reads job_description.docx, extracts plain text, then derives structured
requirements from that text using only `re` and string matching (no NLP
libraries). Nothing here is hardcoded to a specific JD: every value returned
is extracted from the JD text. The only constants are recognition
vocabularies (Indian cities, consulting firms, work modes, skill terms) used
to decide *what to look for* — a term is emitted only if it actually appears
in the JD text.

The city and consulting-firm vocabularies are imported from field_map so the
whole project shares one source of truth.

Usage:
    from jd_parser import JD
    yoe_min = JD['yoe_min']
"""

import re

from field_map import TIER_1_CITIES, CONSULTING_FIRMS

import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))

# ── recognition vocabularies (what to look for, not what to return) ──

# Work-mode keywords -> canonical label.
WORK_MODE_KEYWORDS = {
    'hybrid':   'hybrid',
    'remote':   'remote',
    'onsite':   'onsite',
    'on-site':  'onsite',
    'in-office': 'onsite',
    'flexible': 'flexible',
}

# Skill / technology terms to detect in the JD text. Detection is
# case-insensitive and word-boundary aware; only terms present are emitted.
SKILL_TERMS = [
    'embeddings', 'retrieval', 'hybrid retrieval', 'hybrid search',
    'ranking', 'learning-to-rank', 're-ranking', 'reranking',
    'vector databases', 'vector database',
    'sentence-transformers', 'openai embeddings', 'bge', 'e5',
    'pinecone', 'weaviate', 'qdrant', 'milvus', 'opensearch',
    'elasticsearch', 'faiss', 'bm25',
    'llms', 'llm', 'fine-tuning', 'lora', 'qlora', 'peft',
    'python', 'xgboost',
    'ndcg', 'mrr', 'map',
    'evaluation', 'a/b testing', 'a/b test',
    'recommendation', 'nlp', 'ir', 'information retrieval',
]

# Seniority keywords -> canonical level. Order matters: senior > mid > junior.
SENIORITY_KEYWORDS = [
    ('senior',   'senior'),
    ('staff',    'senior'),
    ('principal', 'senior'),
    ('lead',     'senior'),
    ('mid-level', 'mid'),
    ('mid level', 'mid'),
    ('intermediate', 'mid'),
    ('junior',   'junior'),
    ('entry-level', 'junior'),
    ('entry level', 'junior'),
    ('associate', 'junior'),
]


def extract_text_from_docx(path):
    from docx import Document
    doc = Document(path)
    return '\n'.join([para.text for para in doc.paragraphs if para.text.strip()])


def _normalize(text):
    """Lowercase and normalize dash variants for matching."""
    # Normalize en-dash / em-dash to plain hyphen so "5–9" and "5-9" match.
    return text.replace('–', '-').replace('—', '-').lower()


def extract_yoe(text):
    """
    Extract (yoe_min, yoe_max) from the JD text.

    Handles patterns like "X-Y years", "X+ years", "minimum X years".

    Explicit ranges ("X-Y years") are the authoritative experience
    requirement, so when any range is present we derive the bounds from
    ranges alone. This avoids false positives from open-ended "X+ years"
    phrases that refer to tenure/duration rather than required experience
    (e.g. "plans to be here for 3+ years"). When no range exists we fall
    back to "X+ years" / "minimum X years" / "maximum X years" cues.

    The widest min/max across the chosen matches is returned so explicitly
    mentioned outer bounds are respected. Missing bound -> None.
    """
    norm = _normalize(text)

    # "X-Y years" (range) — the authoritative signal when present.
    range_mins, range_maxs = [], []
    for m in re.finditer(r'(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*\+?\s*years?', norm):
        range_mins.append(float(m.group(1)))
        range_maxs.append(float(m.group(2)))

    if range_mins:
        return min(range_mins), max(range_maxs)

    # No explicit range — fall back to open-ended cues.
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


def extract_locations(text):
    """City names from TIER_1_CITIES that appear in the JD, plus 'tier-1' if mentioned."""
    norm = _normalize(text)
    found = []
    for city in sorted(TIER_1_CITIES):
        if re.search(r'\b' + re.escape(city) + r'\b', norm):
            found.append(city)
    if re.search(r'\btier[\s-]?1\b', norm):
        found.append('tier-1')
    # Preserve order of first appearance, de-duplicated.
    seen, ordered = set(), []
    for item in found:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def extract_work_modes(text):
    """Canonical work modes whose keywords appear in the JD."""
    norm = _normalize(text)
    found, seen = [], set()
    for keyword, label in WORK_MODE_KEYWORDS.items():
        if re.search(r'\b' + re.escape(keyword) + r'\b', norm) and label not in seen:
            seen.add(label)
            found.append(label)
    return found


def extract_disqualified_companies(text):
    """
    Consulting firms from CONSULTING_FIRMS that appear in the JD in a
    disqualifying context. We look for firms mentioned near negative
    language ("do not want", "only worked at", "consulting firms").
    """
    norm = _normalize(text)
    found = []
    for firm in sorted(CONSULTING_FIRMS):
        if re.search(r'\b' + re.escape(firm) + r'\b', norm):
            found.append(firm)

    if not found:
        return []

    # Only treat them as disqualifiers if a negative cue co-occurs in the text.
    negative_cues = (
        'do not want', "don't want", 'not want', 'consulting firm',
        'bad fit', 'not a fit', 'disqualif', 'only worked',
    )
    if any(cue in norm for cue in negative_cues):
        return found
    return []


def extract_required_skills(text):
    """Skill/technology terms from SKILL_TERMS that appear in the JD text."""
    norm = _normalize(text)
    found, seen = [], set()
    # Match longer phrases first so "hybrid retrieval" wins over "retrieval".
    for term in sorted(SKILL_TERMS, key=len, reverse=True):
        pattern = r'(?<!\w)' + re.escape(term.lower()) + r'(?!\w)'
        if re.search(pattern, norm) and term not in seen:
            seen.add(term)
            found.append(term)
    return found


def extract_seniority(text):
    """Seniority level inferred from title/text. Defaults to 'mid' if unclear."""
    norm = _normalize(text)
    # Prefer a match in the first line (the title), then fall back to body.
    title = norm.split('\n', 1)[0]
    for scope in (title, norm):
        for keyword, level in SENIORITY_KEYWORDS:
            if re.search(r'\b' + re.escape(keyword) + r'\b', scope):
                return level
    return 'mid'


def parse_jd(jd_path=None):
    if jd_path is None:
        jd_path = _os.path.join(_HERE, 'job_description.docx')
    jd_text = extract_text_from_docx(jd_path)
    yoe_min, yoe_max = extract_yoe(jd_text)
    return {
        'yoe_min': yoe_min,
        'yoe_max': yoe_max,
        'preferred_locations': extract_locations(jd_text),
        'preferred_work_modes': extract_work_modes(jd_text),
        'disqualified_companies': extract_disqualified_companies(jd_text),
        'jd_text': jd_text,
        'required_skills': extract_required_skills(jd_text),
        'seniority_level': extract_seniority(jd_text),
    }


# Parsed once on import so other modules can `from jd_parser import JD`.
JD = parse_jd()


if __name__ == '__main__':
    from pprint import pprint
    pprint(JD)
