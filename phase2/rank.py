"""
rank.py
=======
Phase 2 entry point. Combines Track 1 structured signals and Track 2 semantic
embeddings (pre-computed in Phase 1) into a single final ranking, then emits a
submission CSV with one human-readable reasoning sentence per candidate.

Pipeline
--------
1. Load the six Phase 1 artifacts (parallel arrays + merged Track 1 details).
2. Min-max normalize semantic scores across the full population.
3. final_score = semantic_norm * hard_filter * availability * credibility.
   (Equal-weight multiplicative — disqualifying penalties propagate naturally.)
4. Sort (final_score, then semantic_norm, then hard_filter, all descending)
   via a stable np.lexsort, slice top min(100, N).
5. Stream the raw candidates file once, keeping only the top-N profiles.
6. Generate honest, profile-specific reasoning per candidate.
7. Write submission.csv.

Self-contained: standard library + numpy + pickle + json + csv only. No model
load, no network, no GPU, no LLM calls. Runs in well under 5 minutes on CPU.

Usage
-----
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv
    python rank.py --candidates data/sample_candidates.json \
        --artifacts artifacts --out submission_sample.csv
"""

import argparse
import csv
import json
import os
import pickle
import time

import numpy as np

# Top-N to emit. Capped at the population size for small fixtures.
TOP_N = 100

# Artifact filenames (parallel arrays share candidate order).
_NPY_ARTIFACTS = {
    'candidate_ids': 'candidate_ids.npy',
    'semantic': 'semantic_scores.npy',
    'hard_filter': 'hard_filter_scores.npy',
    'availability': 'availability_scores.npy',
    'credibility': 'credibility_scores.npy',
}
_PKL_ARTIFACT = 'track1_details.pkl'

# JD requirement vocabulary (from the Redrob JD): production retrieval /
# embeddings, vector databases, Python, ranking/IR evaluation. Used only to
# decide which of a candidate's *actual* skills/text to surface — never to
# invent capability they do not have.
_JD_SKILL_HINTS = {
    'embedding': 'production embeddings',
    'embeddings': 'production embeddings',
    'retrieval': 'retrieval systems',
    'rag': 'retrieval systems',
    'vector': 'vector search',
    'faiss': 'vector search',
    'pinecone': 'vector search',
    'weaviate': 'vector search',
    'qdrant': 'vector search',
    'milvus': 'vector search',
    'elasticsearch': 'search infrastructure',
    'opensearch': 'search infrastructure',
    'bm25': 'ranking/IR',
    'ranking': 'ranking/IR',
    'rerank': 'ranking/IR',
    'reranking': 'ranking/IR',
    'learning-to-rank': 'ranking/IR',
    'ndcg': 'ranking evaluation',
    'mrr': 'ranking evaluation',
    'information retrieval': 'retrieval systems',
    'nlp': 'NLP',
    'python': 'Python',
    'pytorch': 'deep learning',
    'tensorflow': 'deep learning',
    'transformers': 'transformer models',
    'sentence-transformers': 'production embeddings',
    'llm': 'LLMs',
    'llms': 'LLMs',
    'fine-tuning': 'model fine-tuning',
    'xgboost': 'ML modeling',
}

# JD-relevant terms to look for in free text (career description / summary).
_JD_TEXT_TERMS = (
    'embedding', 'retrieval', 'vector', 'search', 'rank', 'rerank',
    'recommendation', 'semantic', 'rag', 'nlp', 'llm', 'information retrieval',
)

_PROFICIENCY_RANK = {'expert': 3, 'advanced': 2, 'intermediate': 1, 'beginner': 0}


# ──────────────────────────────────────────────────────────────────────────
# STEP 1 — artifact loading
# ──────────────────────────────────────────────────────────────────────────
def load_artifacts(artifacts_dir):
    """Load all six artifacts, verify presence and equal length."""
    missing = []
    for fn in list(_NPY_ARTIFACTS.values()) + [_PKL_ARTIFACT]:
        if not os.path.exists(os.path.join(artifacts_dir, fn)):
            missing.append(fn)
    if missing:
        raise SystemExit(
            f"Missing artifact(s) in {artifacts_dir}: {', '.join(missing)}"
        )

    arrays = {}
    arrays['candidate_ids'] = np.load(
        os.path.join(artifacts_dir, _NPY_ARTIFACTS['candidate_ids']),
        allow_pickle=True,
    )
    for key in ('semantic', 'hard_filter', 'availability', 'credibility'):
        arrays[key] = np.load(
            os.path.join(artifacts_dir, _NPY_ARTIFACTS[key]),
            allow_pickle=False,
        ).astype(np.float64)

    with open(os.path.join(artifacts_dir, _PKL_ARTIFACT), 'rb') as f:
        track1_details = pickle.load(f)

    print('Artifact shapes:')
    for key in ('candidate_ids', 'semantic', 'hard_filter',
                'availability', 'credibility'):
        print(f'  {_NPY_ARTIFACTS[key]:24} {arrays[key].shape}')
    print(f'  {_PKL_ARTIFACT:24} ({len(track1_details)},)')

    lengths = {
        len(arrays['candidate_ids']), len(arrays['semantic']),
        len(arrays['hard_filter']), len(arrays['availability']),
        len(arrays['credibility']), len(track1_details),
    }
    if len(lengths) != 1:
        raise SystemExit(
            f"Parallel artifacts have inconsistent lengths: {sorted(lengths)}. "
            "Re-run phase1/track2_embedding.py (optionally --skip-embeddings) "
            "to regenerate aligned artifacts."
        )
    print(f'All parallel arrays aligned at length {lengths.pop()}.')
    return arrays, track1_details


# ──────────────────────────────────────────────────────────────────────────
# STEP 2 — semantic normalization
# ──────────────────────────────────────────────────────────────────────────
def normalize_semantic(semantic):
    """Min-max normalize across the full population. Flat input -> all zeros."""
    lo = float(semantic.min())
    hi = float(semantic.max())
    span = hi - lo
    if span == 0:
        norm = np.zeros_like(semantic)
    else:
        norm = (semantic - lo) / span
    print(f'Semantic raw range : [{lo:.6f}, {hi:.6f}]')
    print(f'Semantic norm range: [{float(norm.min()):.6f}, {float(norm.max()):.6f}]')
    return norm


# ──────────────────────────────────────────────────────────────────────────
# STEP 4 — sort + slice
# ──────────────────────────────────────────────────────────────────────────
def rank_top_n(final_score, semantic_norm, hard_filter, n):
    """
    Stable multi-key descending sort:
      primary  final_score, secondary semantic_norm, tertiary hard_filter.
    np.lexsort sorts ascending by the LAST key first, so we negate to get
    descending and list keys least-significant first.
    """
    order = np.lexsort((-hard_filter, -semantic_norm, -final_score))
    return order[:n]


# ──────────────────────────────────────────────────────────────────────────
# STEP 5 — load raw profiles for the top-N only
# ──────────────────────────────────────────────────────────────────────────
def load_top_profiles(candidates_path, wanted_ids):
    """
    Stream the candidates file once, keeping only profiles whose candidate_id
    is in wanted_ids. Supports JSONL (one object per line) and a single JSON
    array. Never holds the full population in memory for the JSONL path.
    """
    wanted = set(wanted_ids)
    found = {}
    with open(candidates_path) as f:
        head = f.read(1)
        f.seek(0)
        if head == '[':
            # JSON array fixture — load once, filter, discard the rest.
            for c in json.load(f):
                cid = c.get('candidate_id')
                if cid in wanted and cid not in found:
                    found[cid] = c
                    if len(found) == len(wanted):
                        break
        else:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                c = json.loads(line)
                cid = c.get('candidate_id')
                if cid in wanted and cid not in found:
                    found[cid] = c
                    if len(found) == len(wanted):
                        break
    return found


# ──────────────────────────────────────────────────────────────────────────
# STEP 6 — reasoning generation
# ──────────────────────────────────────────────────────────────────────────
def _current_role(profile, career_history):
    """Return (title, company) and the current-role description."""
    title = (profile.get('current_title') or '').strip()
    company = (profile.get('current_company') or '').strip()
    desc = ''
    if career_history:
        current = next(
            (j for j in career_history if j.get('is_current')),
            career_history[0],
        )
        desc = (current.get('description') or '').strip()
    return title, company, desc


def _top_skills(skills, k=3):
    """Top-k skills by proficiency (expert>advanced>intermediate), then endorsements."""
    ranked = [
        s for s in skills
        if isinstance(s, dict) and s.get('name')
        and _PROFICIENCY_RANK.get((s.get('proficiency') or '').lower(), -1) >= 1
    ]
    ranked.sort(
        key=lambda s: (
            _PROFICIENCY_RANK.get((s.get('proficiency') or '').lower(), 0),
            s.get('endorsements') or 0,
        ),
        reverse=True,
    )
    return ranked[:k]


def _jd_hits(skills, desc, summary):
    """
    Which JD requirement areas the candidate *actually* shows evidence for,
    drawn only from their real skills and free text. Returns an ordered,
    de-duplicated list of requirement labels.
    """
    hits, seen = [], set()
    for s in skills:
        name = (s.get('name') or '').lower()
        for token, label in _JD_SKILL_HINTS.items():
            if token in name and label not in seen:
                seen.add(label)
                hits.append(label)
    blob = f'{desc} {summary}'.lower()
    for term in _JD_TEXT_TERMS:
        label = _JD_SKILL_HINTS.get(term)
        if term in blob and label and label not in seen:
            seen.add(label)
            hits.append(label)
    return hits


def _clean(text):
    """Strip commas (CSV uses comma delimiter) and collapse whitespace."""
    return ' '.join(text.replace(',', ';').split())


# Generic openers that signal a low-information, templated description. When a
# description starts with one of these we keep only its first sentence.
_GENERIC_DESC_OPENERS = (
    'built systems that',
    'worked at the intersection',
    'responsible for',
    'worked on',
    'helped build',
    'contributed to',
)


def _career_clause(title, company, industry, desc, prev_desc_key):
    """
    Build the current-role evidence clause, returning (clause_or_empty, key).

    - Skip the description entirely if it is empty or under 30 chars.
    - For descriptions opening with a generic phrase, keep only the first
      sentence, capped at 100 chars (not the full 200).
    - Otherwise use up to 200 chars.
    - If the resulting first-100-char fingerprint matches the previous
      candidate's, fall back to a constructed sentence so consecutive rows do
      not share the same description text.

    prev_desc_key is the fingerprint from the previous candidate (or None).
    The returned key is this candidate's fingerprint, to be threaded forward.
    """
    desc = (desc or '').strip()

    def _fallback():
        parts = []
        if title:
            parts.append(f'Currently {title}')
            if company:
                parts[-1] += f' at {company}'
        elif company:
            parts.append(f'Currently at {company}')
        if industry:
            parts.append(f'in {industry} industry')
        if not parts:
            return ''
        return 'Current role: ' + ' '.join(parts) + '.'

    if len(desc) < 30:
        # Nothing meaningful to quote — use the constructed sentence.
        return _fallback(), None

    lowered = desc.lower()
    if any(lowered.startswith(op) for op in _GENERIC_DESC_OPENERS):
        snippet = desc.split('.', 1)[0][:100].rstrip()
    else:
        snippet = desc[:200].rstrip()

    fingerprint = snippet[:100]
    if prev_desc_key is not None and fingerprint == prev_desc_key:
        # Same description as the immediately preceding candidate — avoid the
        # templated look by constructing a sentence instead.
        return _fallback(), fingerprint

    return 'Current role: ' + snippet + '.', fingerprint


def build_reasoning(rank, profile, candidate, det, semantic_norm,
                    prev_desc_key=None):
    """
    Compose an honest, profile-specific reasoning sentence. Strengths first,
    JD connections next, concerns flagged plainly, closing assessment toned to
    the candidate's rank position. Only cites facts present in the profile.

    prev_desc_key threads the previous candidate's description fingerprint so
    consecutive rows never reuse the same career text. Returns
    (reasoning, desc_key) where desc_key feeds the next call.
    """
    skills_list = candidate.get('skills', []) or []
    career = candidate.get('career_history', []) or []
    signals = candidate.get('redrob_signals', {}) or {}

    title, company, desc = _current_role(profile, career)
    yoe = profile.get('years_of_experience')
    location = (profile.get('location') or '').strip()
    country = (profile.get('country') or '').strip()
    summary = (profile.get('summary') or '').strip()

    # ── opening: who they are ──
    role = title or 'Candidate'
    if company:
        role += f' at {company}'
    bits = []
    if yoe is not None:
        bits.append(f'{yoe:g}y exp')
    loc_str = location or country
    if loc_str:
        bits.append(loc_str)
    willing = bool(signals.get('willing_to_relocate'))
    outside_india = country and country.strip().lower() != 'india'
    if outside_india and willing:
        bits.append('willing to relocate')
    header = role + (f' ({"; ".join(bits)})' if bits else '')

    # ── semantic match phrasing tied to normalized score ──
    sn = float(semantic_norm)
    if sn >= 0.7:
        match = 'Strong semantic match to the JD'
    elif sn >= 0.45:
        match = 'Moderate semantic match to the JD'
    else:
        match = 'Weaker semantic match to the JD'
    jd_hits = _jd_hits(skills_list, desc, summary)
    if jd_hits:
        match += ' — relevant to ' + '; '.join(jd_hits[:4])
    else:
        match += ' — limited retrieval/embeddings exposure visible in profile'

    # ── top skills (only real ones) ──
    tops = _top_skills(skills_list)
    if tops:
        skill_str = 'Top skills: ' + '; '.join(
            f'{s["name"]} ({s.get("proficiency", "")})' for s in tops
        )
    else:
        skill_str = 'No intermediate-or-above skills listed'

    sentences = [header + '.', match + '.', skill_str + '.']

    # ── current-role evidence (smart, non-repeating) ──
    industry = (profile.get('current_industry') or '').strip()
    career_clause, desc_key = _career_clause(
        title, company, industry, desc, prev_desc_key
    )
    if career_clause:
        sentences.append(career_clause)

    # ── honest concerns (signal-driven; always at least one flag) ──
    location_score = float(det.get('location_score', 1.0))
    notice_sig = float(det.get('notice_period_signal', 1.0))
    github_sig = float(det.get('github_signal', 0.0))
    consulting_pen = float(det.get('consulting_penalty', 1.0))
    tenure_sig = float(det.get('tenure_score', 1.0))
    avg_ten = det.get('average_tenure_months')
    linkedin_sig = float(det.get('linkedin_signal', 0.0))
    loc_label = (location or country or 'location unknown')

    concerns = []

    # Location
    if 0.9 <= location_score < 1.0:
        concerns.append(
            f'Not in office city ({loc_label}; {country}) — quarterly travel expected.')
    elif 0.6 <= location_score < 0.9:
        concerns.append(
            f'Outside Pune/Noida offices ({loc_label}) — relocation or travel needed.')
    elif location_score < 0.6:
        concerns.append(
            f'Outside India ({loc_label}; {country}) — case-by-case per JD.')

    # Notice period (most severe flag wins)
    if notice_sig < 0.70:
        concerns.append('Notice period likely 90d+ — long; reduces hiring speed.')
    elif notice_sig < 0.85:
        concerns.append('Notice period likely 60d+ — above JD preferred sub-30d.')

    # GitHub
    if github_sig == 0.0:
        concerns.append(
            'No GitHub linked — external validation of technical work unavailable.')
    elif 0.0 < github_sig < 0.3:
        concerns.append('GitHub present but low activity — limited open-source signal.')

    # Consulting
    if consulting_pen < 1.0:
        concerns.append(
            'Current or recent consulting background — culture fit risk per JD.')

    # Tenure
    if tenure_sig < 0.85:
        ten_txt = f'{avg_ten:.0f}' if isinstance(avg_ten, (int, float)) and avg_ten else 'short'
        concerns.append(
            f'Average tenure {ten_txt} months — job-hopping signal; '
            'JD requires 3+ year commitment.')

    # LinkedIn
    if linkedin_sig == 0.0:
        concerns.append('LinkedIn not connected — profile verification limited.')

    # Semantic
    if sn < 0.5:
        concerns.append(
            'Moderate semantic match to JD — profile relevance is borderline.')

    if not concerns:
        concerns.append('No major disqualifying signals detected.')

    sentences.append('Concerns: ' + ' '.join(concerns))

    # ── closing assessment toned to rank ──
    if rank <= 10:
        verdict = 'Strong overall fit for the founding-team role.'
    elif rank <= 40:
        verdict = 'Solid fit; worth advancing.'
    elif rank <= 75:
        verdict = 'Reasonable fit with some gaps; worth a look.'
    else:
        verdict = 'Borderline fit; included for broad coverage of the top-100.'
    sentences.append(verdict)

    return _clean(' '.join(sentences)), desc_key


# ──────────────────────────────────────────────────────────────────────────
# orchestration
# ──────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Phase 2 ranking: combine Track 1 + semantic artifacts '
                    'into a ranked submission CSV.'
    )
    parser.add_argument('--candidates', required=True,
                        help='Path to candidates.jsonl (or a JSON array fixture)')
    parser.add_argument('--out', required=True,
                        help='Path to write the submission CSV')
    default_artifacts = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'artifacts'
    )
    parser.add_argument('--artifacts', default=default_artifacts,
                        help='Artifacts directory (default: ../artifacts)')
    args = parser.parse_args()

    timings = {}
    total_start = time.time()

    # STEP 1 — load artifacts
    t = time.time()
    arrays, track1_details = load_artifacts(args.artifacts)
    timings['Artifact loading'] = time.time() - t

    candidate_ids = arrays['candidate_ids']
    semantic = arrays['semantic']
    hard_filter = arrays['hard_filter']
    availability = arrays['availability']
    credibility = arrays['credibility']

    # STEP 2 + 3 — normalize + score
    t = time.time()
    semantic_norm = normalize_semantic(semantic)
    final_score = semantic_norm * hard_filter * availability * credibility

    # STEP 4 — sort + slice
    n = min(TOP_N, len(candidate_ids))
    top_idx = rank_top_n(final_score, semantic_norm, hard_filter, n)
    timings['Normalization + scoring'] = time.time() - t

    print(f'\nTop 10 preview (of {n} ranked):')
    print(f'{"rk":>2} {"candidate_id":14} {"final":>8} {"sem_norm":>8} '
          f'{"hard":>6} {"avail":>6} {"cred":>6}')
    for r, i in enumerate(top_idx[:10], start=1):
        print(f'{r:>2} {str(candidate_ids[i]):14} {final_score[i]:8.6f} '
              f'{semantic_norm[i]:8.6f} {hard_filter[i]:6.3f} '
              f'{availability[i]:6.3f} {credibility[i]:6.3f}')

    # STEP 5 — load raw profiles for the top-N only
    t = time.time()
    top_ids = [str(candidate_ids[i]) for i in top_idx]
    profiles = load_top_profiles(args.candidates, top_ids)
    timings['Profile loading (top N)'] = time.time() - t
    missing_profiles = [cid for cid in top_ids if cid not in profiles]
    if missing_profiles:
        print(f'WARNING: {len(missing_profiles)} top candidate_ids had no '
              f'matching profile in {args.candidates} '
              f'(e.g. {missing_profiles[:3]}). Reasoning will be minimal for these.')

    # STEP 6 — reasoning
    t = time.time()
    rows = []
    prev_desc_key = None
    for r, i in enumerate(top_idx, start=1):
        cid = str(candidate_ids[i])
        det = track1_details[i]
        candidate = profiles.get(cid, {})
        profile = candidate.get('profile', {}) if candidate else {}
        reasoning, prev_desc_key = build_reasoning(
            r, profile, candidate, det, semantic_norm[i], prev_desc_key
        )
        if not reasoning:
            reasoning = f'Ranked #{r}; profile unavailable for detailed reasoning.'
        rows.append({
            'rank': r,
            'candidate_id': cid,
            'final_score': round(float(final_score[i]), 6),
            'semantic_score_raw': round(float(semantic[i]), 6),
            'semantic_score_norm': round(float(semantic_norm[i]), 6),
            'hard_filter_score': round(float(hard_filter[i]), 6),
            'availability_score': round(float(availability[i]), 6),
            'credibility_score': round(float(credibility[i]), 6),
            'reasoning': reasoning,
        })
    timings['Reasoning generation'] = time.time() - t

    # STEP 8 — sanity checks (before writing)
    print('\nSanity checks:')
    print(f'  Total rows written           : {len(rows)} (expected {n})')
    ids = [row['candidate_id'] for row in rows]
    dup = len(ids) - len(set(ids))
    print(f'  Duplicate candidate_ids      : {dup} (expected 0)')
    empty = sum(1 for row in rows if not row['reasoning'].strip())
    print(f'  Empty reasoning strings      : {empty} (expected 0)')
    hf_low = sum(1 for row in rows if row['hard_filter_score'] < 0.5)
    print(f'  hard_filter < 0.5 in top {n:<3} : {hf_low} (should be 0)')
    sn_low = sum(1 for row in rows if row['semantic_score_norm'] < 0.3)
    print(f'  semantic_norm < 0.3 in top {n:<3}: {sn_low} (should be 0)')
    if rows:
        fmin = min(row['final_score'] for row in rows)
        fmax = max(row['final_score'] for row in rows)
        print(f'  final_score min/max          : {fmin:.6f} / {fmax:.6f}')

    # STEP 7 — write CSV
    t = time.time()
    columns = ['candidate_id', 'rank', 'score', 'reasoning']
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({
                'candidate_id': row['candidate_id'],
                'rank': row['rank'],
                'score': row['final_score'],
                'reasoning': row['reasoning'],
            })
    timings['CSV writing'] = time.time() - t

    # STEP 9 — timing
    timings['Total'] = time.time() - total_start
    print('\nTimings (s):')
    for k, v in timings.items():
        print(f'  {k:26} {v:7.3f}')
    print(f'\nWrote {len(rows)} rows to {args.out}')


if __name__ == '__main__':
    main()
