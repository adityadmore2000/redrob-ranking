"""
demo_pipeline.py
================
In-memory, end-to-end ranking pipeline for the Streamlit demo.

The production scripts in phase1/ and phase2/rank.py are file-based: Phase 1
writes six artifacts to artifacts/ for a fixed candidate population, and Phase 2
reads them back. That split exists so the heavy embedding pass is computed once
and reused. For an interactive demo the user uploads a *fresh, arbitrary* pool of
candidates, so there are no pre-computed artifacts to reuse — every signal must be
recomputed for that exact pool.

This module reuses the real scoring logic (the same compute_* functions and the
same text builder + BGE embedding the production pipeline uses) but runs it all in
memory on an uploaded list of candidate dicts and returns a ranked pandas
DataFrame. The semantic normalization, multiplicative final score, sort order and
reasoning text all mirror phase2/rank.py exactly.

Public API
----------
load_candidates_from_bytes(raw, filename) -> list[dict]
rank_candidates(candidates, model, progress=None) -> pandas.DataFrame
get_model() -> SentenceTransformer   (cache this in the app)
MAX_CANDIDATES
"""

import io
import json
import os
import sys

import numpy as np
import pandas as pd

# Make the project root importable so we reuse the real scoring modules.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PHASE1 = os.path.join(_HERE, 'phase1')
for _p in (_HERE, _PHASE1):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from field_map import get_candidate_id
import track1_hard_filter
import track1_availability
import track1_credibility
from track2_text_builder import build_candidate_text, build_jd_text

# Reuse the exact reasoning logic from Phase 2 so the demo output matches the
# real submission column verbatim.
from phase2.rank import build_reasoning, normalize_semantic, rank_top_n

MODEL_NAME = 'BAAI/bge-base-en-v1.5'
MAX_CANDIDATES = 100


# ──────────────────────────────────────────────────────────────────────────
# input loading
# ──────────────────────────────────────────────────────────────────────────
class InputError(ValueError):
    """Raised for user-facing input problems (bad format, too many rows, …)."""


def load_candidates_from_bytes(raw: bytes, filename: str) -> list:
    """
    Parse uploaded bytes into a list of candidate dicts.

    Accepts:
      * a single JSON array of candidate objects  (``.json``)
      * JSON Lines — one candidate object per line (``.jsonl``)

    Raises InputError with a clear message on any structural problem.
    """
    name = (filename or '').lower()
    try:
        text = raw.decode('utf-8')
    except UnicodeDecodeError as exc:
        raise InputError(
            'File is not valid UTF-8 text. Upload a .json or .jsonl file.'
        ) from exc

    text = text.strip()
    if not text:
        raise InputError('The uploaded file is empty.')

    candidates = None
    # Prefer the extension, but fall back to sniffing the first non-space char
    # so a mislabeled file still loads.
    if name.endswith('.jsonl') or (not name.endswith('.json') and text[0] != '['):
        candidates = _parse_jsonl(text)
    else:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            # Maybe it is really JSONL with a .json extension — try that.
            candidates = _parse_jsonl(text)
        else:
            if isinstance(parsed, list):
                candidates = parsed
            elif isinstance(parsed, dict):
                candidates = [parsed]
            else:
                raise InputError(
                    'JSON must be an array of candidate objects '
                    '(or one object per line for .jsonl).'
                )

    if not candidates:
        raise InputError('No candidate records found in the file.')

    if any(not isinstance(c, dict) for c in candidates):
        raise InputError(
            'Every record must be a JSON object with at least a '
            '"candidate_id" and a "profile" field.'
        )

    missing_profile = sum(1 for c in candidates if 'profile' not in c)
    if missing_profile == len(candidates):
        raise InputError(
            'No record has a "profile" field. This does not look like a '
            'candidate dataset — see the expected schema in the sidebar.'
        )

    if len(candidates) > MAX_CANDIDATES:
        raise InputError(
            f'Too many candidates: {len(candidates)}. This demo accepts at '
            f'most {MAX_CANDIDATES}. Trim the file and try again.'
        )

    # Give every record a stable id so the output and reasoning never break on
    # a record that forgot candidate_id.
    for i, c in enumerate(candidates):
        if not c.get('candidate_id'):
            c['candidate_id'] = f'CAND_{i + 1:07d}'

    return candidates


def _parse_jsonl(text: str) -> list:
    out = []
    for n, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise InputError(f'Line {n} is not valid JSON: {exc.msg}.') from exc
    return out


# ──────────────────────────────────────────────────────────────────────────
# model
# ──────────────────────────────────────────────────────────────────────────
def get_model(device: str = 'cpu'):
    """Load the BGE embedding model. Heavy — the app should cache the result."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_NAME, device=device)


# ──────────────────────────────────────────────────────────────────────────
# ranking
# ──────────────────────────────────────────────────────────────────────────
# Output columns, in the same order phase2/rank.py writes them.
OUTPUT_COLUMNS = [
    'rank', 'candidate_id', 'name', 'final_score', 'semantic_score_raw',
    'semantic_score_norm', 'hard_filter_score', 'availability_score',
    'credibility_score', 'reasoning',
]


def rank_candidates(candidates: list, model, progress=None) -> pd.DataFrame:
    """
    Run the full Track 1 + Track 2 pipeline on an in-memory candidate list and
    return a ranked DataFrame (rank #1 first).

    Mirrors phase2/rank.py:
      final_score = semantic_norm * hard_filter * availability * credibility
    with the same min-max semantic normalization, the same stable lexsort
    ordering, and the same per-candidate reasoning text.

    progress: optional callable(fraction: float, message: str) for UI updates.
    """
    def _tick(frac, msg):
        if progress:
            progress(frac, msg)

    n = len(candidates)

    # ── Track 1: structured signals (fast, pure-python) ──
    _tick(0.05, 'Scoring structured signals (Track 1)…')
    hard = track1_hard_filter.compute_all(candidates)
    avail = track1_availability.compute_all(candidates)
    cred = track1_credibility.compute_all(candidates)

    # Merge per-candidate Track 1 detail dicts — same shape rank.py expects.
    track1_details = []
    for h, a, cr in zip(hard, avail, cred):
        row = {'candidate_id': h['candidate_id']}
        row.update(h)
        row.update(a)
        row.update(cr)
        track1_details.append(row)

    candidate_ids = np.array(
        [get_candidate_id(c) for c in candidates], dtype=object
    )
    hard_filter = np.array(
        [r['hard_filter_score'] for r in hard], dtype=np.float64
    )
    availability = np.array(
        [r['availability_score'] for r in avail], dtype=np.float64
    )
    credibility = np.array(
        [r['credibility_score'] for r in cred], dtype=np.float64
    )

    # ── Track 2: semantic representation + embedding ──
    _tick(0.20, 'Building candidate text representations…')
    texts = [build_candidate_text(c) for c in candidates]

    _tick(0.35, 'Embedding candidates against the job description…')
    from jd_parser import JD
    jd_text = build_jd_text(JD)

    all_vecs = model.encode(
        [jd_text] + texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    jd_vec = all_vecs[0]
    cand_vecs = all_vecs[1:]
    # Vectors are already L2-normalized, so the dot product is cosine similarity.
    semantic = (cand_vecs @ jd_vec).astype(np.float64)

    # ── Phase 2: normalize, score, rank ──
    _tick(0.75, 'Normalizing and computing final scores…')
    semantic_norm = normalize_semantic(semantic)
    final_score = semantic_norm * hard_filter * availability * credibility

    top_n = min(MAX_CANDIDATES, n)
    top_idx = rank_top_n(final_score, semantic_norm, hard_filter, top_n)

    # ── reasoning (same generator as the real submission) ──
    _tick(0.88, 'Generating per-candidate reasoning…')
    by_id = {str(get_candidate_id(c)): c for c in candidates}

    rows = []
    prev_desc_key = None
    for r, i in enumerate(top_idx, start=1):
        cid = str(candidate_ids[i])
        candidate = by_id.get(cid, {})
        profile = candidate.get('profile', {}) if candidate else {}
        reasoning, prev_desc_key = build_reasoning(
            r, profile, candidate, track1_details[i], semantic_norm[i],
            prev_desc_key,
        )
        if not reasoning:
            reasoning = f'Ranked #{r}; profile unavailable for detailed reasoning.'
        rows.append({
            'rank': r,
            'candidate_id': cid,
            'name': (profile.get('anonymized_name') or '').strip(),
            'final_score': round(float(final_score[i]), 6),
            'semantic_score_raw': round(float(semantic[i]), 6),
            'semantic_score_norm': round(float(semantic_norm[i]), 6),
            'hard_filter_score': round(float(hard_filter[i]), 6),
            'availability_score': round(float(availability[i]), 6),
            'credibility_score': round(float(credibility[i]), 6),
            'reasoning': reasoning,
        })

    _tick(1.0, 'Done.')
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
