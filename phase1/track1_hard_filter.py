"""
track1_hard_filter.py
=====================
Computes hard_filter_score for each candidate against the parsed JD.

Formula
-------
hard_filter_score = location_score
                  * yoe_score
                  * work_mode_score
                  * consulting_penalty
                  * tenure_score

Sub-scores
----------
location_score     : Based on city/country and willingness to relocate.
                     Pune/Noida (JD office cities) = 1.0
                     Other Tier-1 India            = 0.9
                     Non Tier-1 India + relocate   = 0.6
                     Non Tier-1 India              = 0.3
                     Outside India + relocate      = 0.6
                     Outside India                 = 0.3

yoe_score          : Based on years of experience vs JD range (5-9 yrs).
                     Ideal range [yoe_min, yoe_max]     = 1.0
                     Outer band [yoe_min-1, yoe_max+1]  = 0.85
                     Outside outer band                 = 0.6

work_mode_score    : Candidate preferred mode vs JD preferred modes.
                     Mode in JD preferred modes = 1.0
                     Otherwise                  = 0.8

consulting_penalty : Entire career history is disqualified consulting firms.
                     All jobs at consulting firms = 0.3
                     Any non-consulting experience = 1.0

tenure_score       : Average job tenure (title-chasing signal).
                     avg >= 24 months          = 1.0
                     18 <= avg < 24            = 0.85
                     12 <= avg < 18            = 0.70
                     avg < 12                  = 0.50
                     no usable tenure data     = 0.75

All field access goes through field_map accessors.
All thresholds come from field_map constants or jd_parser.JD.
Nothing JD-specific is hardcoded except _JD_OFFICE_CITIES (see comment).

Public API
----------
compute_hard_filter_score(c)  -> dict with score components + final score
compute_all(candidates)       -> list of dicts with candidate_id + components
"""

import os as _os
import sys as _sys

# Make both the project root (field_map, jd_parser) and this phase1/ directory
# importable regardless of how the script is launched: `python
# phase1/track1_hard_filter.py` puts only phase1/ on sys.path, while
# `python -m phase1.track1_hard_filter` puts only the project root.
_THIS_DIR = _os.path.dirname(_os.path.abspath(__file__))
_PROJECT_ROOT = _os.path.dirname(_THIS_DIR)
for _p in (_PROJECT_ROOT, _THIS_DIR):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

from field_map import (
    get_candidate_id,
    get_location,
    get_country,
    get_willing_to_relocate,
    get_yoe,
    get_preferred_work_mode,
    get_current_company,
    get_career_sorted,
    TIER_1_CITIES,
    CONSULTING_FIRMS,
    TENURE_IDEAL_MONTHS,
    TENURE_STABLE_MONTHS,
    TENURE_MODERATE_MONTHS,
    TENURE_SCORE_IDEAL,
    TENURE_SCORE_STABLE,
    TENURE_SCORE_MODERATE,
    TENURE_SCORE_SHORT,
    TENURE_SCORE_UNKNOWN,
)
from jd_parser import JD

# Pune and Noida are the JD's preferred office locations ("Pune/Noida-preferred").
# The JD also mentions Hyderabad, Mumbai, Delhi NCR in a "welcome to apply" line —
# those get 0.9 (Tier-1 non-office) not 1.0. We hardcode the office-city set here
# because the parser cannot distinguish "office location" from "welcome to apply"
# purely from text — both appear as city mentions in JD['preferred_locations'].
# Source: JD line "Location: Pune/Noida, India (Hybrid — flexible cadence)"
_JD_OFFICE_CITIES = {'pune', 'noida'}


def _city_matches(location, city):
    """Case-insensitive partial match: 'Bangalore, Karnataka' matches 'bangalore'."""
    return city in location.lower()


def location_score(c):
    """
    country != India -> 0.6 if willing to relocate else 0.3
    country == India:
        city in a JD-named city (Pune/Noida) -> 1.0
        city in another Tier-1 city          -> 0.9
        city not Tier-1 -> 0.6 if willing to relocate else 0.3
    """
    country = get_country(c).strip().lower()
    willing = bool(get_willing_to_relocate(c))

    if country != 'india':
        return 0.6 if willing else 0.3

    location = get_location(c)
    if any(_city_matches(location, city) for city in _JD_OFFICE_CITIES):
        return 1.0
    if any(_city_matches(location, city) for city in TIER_1_CITIES):
        return 0.9
    return 0.6 if willing else 0.3


def yoe_score(c):
    """
    [yoe_min, yoe_max]                              -> 1.0  (ideal band from JD)
    [yoe_min-1, yoe_min) or (yoe_max, yoe_max+1]    -> 0.85 (outer band)
    otherwise                                       -> 0.6
    """
    yoe = get_yoe(c)
    lo = JD['yoe_min']
    hi = JD['yoe_max']

    if lo is None or hi is None:
        # No usable range parsed from the JD — treat all candidates neutrally.
        return 1.0

    if lo <= yoe <= hi:
        return 1.0
    if (lo - 1) <= yoe < lo or hi < yoe <= (hi + 1):
        return 0.85
    return 0.6


def work_mode_score(c):
    """candidate's preferred mode in JD preferred modes -> 1.0 else 0.8."""
    mode = get_preferred_work_mode(c).strip().lower()
    preferred = {m.lower() for m in JD.get('preferred_work_modes', [])}
    return 1.0 if mode in preferred else 0.8


def _is_consulting_firm(company):
    """True if the company name matches any disqualified consulting firm."""
    company = (company or '').lower()
    if not company:
        return False
    return any(firm in company for firm in CONSULTING_FIRMS)


def consulting_penalty(c):
    """
    Penalize candidates whose ENTIRE career history is consulting firms.

    - All career-history jobs at consulting firms -> 0.3
    - At least one prior/other job at a non-consulting firm -> 1.0
      (e.g. currently at a consulting firm but has non-consulting experience)
    - No usable career history -> fall back to current company match.
    """
    career = get_career_sorted(c)
    companies = [job['company'] for job in career if job.get('company')]

    if not companies:
        # No career history to judge — fall back to current company.
        return 0.3 if _is_consulting_firm(get_current_company(c)) else 1.0

    if all(_is_consulting_firm(co) for co in companies):
        return 0.3
    return 1.0


def average_tenure_months(c):
    """
    Mean of duration_months across all career-history jobs.
    Returns None when there is no usable duration data (empty history or
    all durations missing/zero).
    """
    career = get_career_sorted(c)
    durations = [job.get('duration_months') or 0 for job in career]
    durations = [d for d in durations if d]
    if not durations:
        return None
    return sum(durations) / len(durations)


def tenure_score(c):
    """
    Average job tenure as a title-chasing signal.

    avg >= 24 months          -> 1.0   (stable)
    18 <= avg < 24            -> 0.85
    12 <= avg < 18            -> 0.70
    avg < 12                  -> 0.50  (switching every year)
    no usable tenure data     -> 0.75  (neutral)
    """
    avg = average_tenure_months(c)
    if avg is None:
        return TENURE_SCORE_UNKNOWN
    if avg >= TENURE_IDEAL_MONTHS:
        return TENURE_SCORE_IDEAL
    if avg >= TENURE_STABLE_MONTHS:
        return TENURE_SCORE_STABLE
    if avg >= TENURE_MODERATE_MONTHS:
        return TENURE_SCORE_MODERATE
    return TENURE_SCORE_SHORT


def compute_hard_filter_score(c: dict) -> dict:
    """
    Returns a dict with:
    {
        'hard_filter_score': float,  # final product
        'location_score': float,
        'yoe_score': float,
        'work_mode_score': float,
        'consulting_penalty': float,
        'tenure_score': float,
        'average_tenure_months': float | None,
    }
    """
    loc = location_score(c)
    yoe = yoe_score(c)
    wm = work_mode_score(c)
    pen = consulting_penalty(c)
    ten = tenure_score(c)
    avg_tenure = average_tenure_months(c)
    return {
        'hard_filter_score': loc * yoe * wm * pen * ten,
        'location_score': loc,
        'yoe_score': yoe,
        'work_mode_score': wm,
        'consulting_penalty': pen,
        'tenure_score': ten,
        'average_tenure_months': avg_tenure,
    }


def compute_all(candidates: list) -> list:
    """
    Takes full list of candidate dicts.
    Returns list of dicts with candidate_id + all score components.
    """
    results = []
    for c in candidates:
        scores = compute_hard_filter_score(c)
        scores['candidate_id'] = get_candidate_id(c)
        results.append(scores)
    return results


def _load_candidates(input_path):
    """
    Load candidates from a path. Supports JSONL (one JSON object per line) and
    a single JSON array (the data/sample_candidates.json fixture).
    """
    import json
    with open(input_path) as f:
        head = f.read(1)
        f.seek(0)
        if head == '[':
            # Single JSON array (sample fixture).
            return json.load(f)
        candidates = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            candidates.append(json.loads(line))
        return candidates


def run(input_path, artifacts_dir='artifacts'):
    """
    Compute hard-filter scores for every candidate in input_path and save:
        hard_filter_scores.npy   shape (N,) float32
        track1_details.pkl       list of N per-candidate detail dicts
    candidate_ids.npy is shared across Phase 1 scripts (saved by
    track2_embedding.py) — only written here if it does not already exist.
    Other artifacts (semantic/availability/credibility/jd_embedding) are
    untouched.
    """
    import os
    import pickle
    import time
    import numpy as np

    os.makedirs(artifacts_dir, exist_ok=True)
    total_start = time.time()

    print(f'Loading candidates from {input_path} ...', flush=True)
    candidates = _load_candidates(input_path)
    n = len(candidates)
    print(f'  {n} candidates loaded')

    candidate_ids = []
    hard_filter_scores = []
    track1_details = []
    for i, c in enumerate(candidates):
        scores = compute_hard_filter_score(c)
        scores['candidate_id'] = get_candidate_id(c)
        candidate_ids.append(scores['candidate_id'])
        hard_filter_scores.append(scores['hard_filter_score'])
        track1_details.append(scores)
        if (i + 1) % 10000 == 0:
            print(f'  scored {i + 1}/{n} candidates '
                  f'({time.time() - total_start:.1f}s elapsed)', flush=True)

    hard_filter_scores = np.array(hard_filter_scores, dtype=np.float32)

    np.save(os.path.join(artifacts_dir, 'hard_filter_scores.npy'),
            hard_filter_scores)
    with open(os.path.join(artifacts_dir, 'track1_details.pkl'), 'wb') as f:
        pickle.dump(track1_details, f)

    # candidate_ids.npy is shared across all Phase 1 scripts and is normally
    # saved by track2_embedding.py. Only write it if it is not already present,
    # so we never clobber the canonical ordering.
    ids_path = os.path.join(artifacts_dir, 'candidate_ids.npy')
    if os.path.exists(ids_path):
        print(f'  candidate_ids.npy already exists — not overwriting')
    else:
        np.save(ids_path, np.array(candidate_ids, dtype=object))
        print(f'  candidate_ids.npy not found — saved {n} ids')

    print(f'Saved hard_filter_scores.npy {hard_filter_scores.shape} and '
          f'track1_details.pkl ({len(track1_details)} dicts) to {artifacts_dir}')
    print(f'Total time: {time.time() - total_start:.1f}s')


def verify(artifacts_dir='artifacts'):
    """Load the saved hard-filter artifacts and print a short summary."""
    import os
    import pickle
    import numpy as np

    hard = np.load(os.path.join(artifacts_dir, 'hard_filter_scores.npy'))
    with open(os.path.join(artifacts_dir, 'track1_details.pkl'), 'rb') as f:
        track1_details = pickle.load(f)

    print(f'hard_filter_scores.npy shape: {hard.shape}')
    print(f'track1_details: {len(track1_details)} dicts')
    print('\nFirst 3 track1_details entries:')
    for row in track1_details[:3]:
        print('-' * 60)
        for k, v in row.items():
            print(f'  {k}: {v}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Compute and save hard-filter scores for all candidates.'
    )
    parser.add_argument(
        '--input',
        default='/kaggle/input/datasets/moreadityad/candidate-dataset/candidates.jsonl',
        help='Path to candidates.jsonl (or a JSON array fixture)',
    )
    parser.add_argument(
        '--artifacts',
        default='artifacts',
        help='Directory to save artifacts',
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify existing artifacts instead of running the pipeline',
    )
    args = parser.parse_args()

    if args.verify:
        verify(args.artifacts)
    else:
        run(args.input, args.artifacts)
        verify(args.artifacts)
