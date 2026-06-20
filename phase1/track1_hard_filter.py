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

Sub-scores
----------
location_score     : Based on city/country and willingness to relocate.
                     Pune/Noida (JD office cities) = 1.0
                     Other Tier-1 India            = 0.9
                     Non Tier-1 India + relocate   = 0.6
                     Non Tier-1 India              = 0.3
                     Outside India + relocate      = 0.4
                     Outside India                 = 0.1

yoe_score          : Based on years of experience vs JD range (5-9 yrs).
                     Ideal range [yoe_min, yoe_max]     = 1.0
                     Outer band [yoe_min-1, yoe_max+1]  = 0.85
                     Outside outer band                 = 0.6

work_mode_score    : Candidate preferred mode vs JD preferred modes.
                     Mode in JD preferred modes = 1.0
                     Otherwise                  = 0.8

consulting_penalty : Current company is a disqualified consulting firm.
                     Firm match  = 0.3
                     No match    = 1.0

All field access goes through field_map accessors.
All thresholds come from field_map constants or jd_parser.JD.
Nothing JD-specific is hardcoded except _JD_OFFICE_CITIES (see comment).

Public API
----------
compute_hard_filter_score(c)  -> dict with score components + final score
compute_all(candidates)       -> list of dicts with candidate_id + components
"""

from field_map import (
    get_candidate_id,
    get_location,
    get_country,
    get_willing_to_relocate,
    get_yoe,
    get_preferred_work_mode,
    get_current_company,
    TIER_1_CITIES,
    CONSULTING_FIRMS,
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
    country != India -> 0.4 if willing to relocate else 0.1
    country == India:
        city in a JD-named city (Pune/Noida) -> 1.0
        city in another Tier-1 city          -> 0.9
        city not Tier-1 -> 0.6 if willing to relocate else 0.3
    """
    country = get_country(c).strip().lower()
    willing = bool(get_willing_to_relocate(c))

    if country != 'india':
        return 0.4 if willing else 0.1

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


def consulting_penalty(c):
    """current company matches any disqualified consulting firm -> 0.3 else 1.0."""
    company = get_current_company(c).lower()
    if not company:
        return 1.0
    if any(firm in company for firm in CONSULTING_FIRMS):
        return 0.3
    return 1.0


def compute_hard_filter_score(c: dict) -> dict:
    """
    Returns a dict with:
    {
        'hard_filter_score': float,  # final product
        'location_score': float,
        'yoe_score': float,
        'work_mode_score': float,
        'consulting_penalty': float,
    }
    """
    loc = location_score(c)
    yoe = yoe_score(c)
    wm = work_mode_score(c)
    pen = consulting_penalty(c)
    return {
        'hard_filter_score': loc * yoe * wm * pen,
        'location_score': loc,
        'yoe_score': yoe,
        'work_mode_score': wm,
        'consulting_penalty': pen,
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


if __name__ == '__main__':
    import json
    import os

    # Prefer the full dataset (JSONL, one record per line). Fall back to the
    # provided 50-record fixture (a single JSON array) when it isn't present.
    jsonl_path = os.path.join('data', 'candidates.jsonl')
    sample_path = os.path.join('data', 'sample_candidates.json')

    LIMIT = 1000
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

    results = compute_all(candidates)

    def pctile(values, q):
        s = sorted(values)
        if not s:
            return float('nan')
        idx = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
        return s[idx]

    components = [
        'location_score', 'yoe_score', 'work_mode_score',
        'consulting_penalty', 'hard_filter_score',
    ]

    print(f"Loaded {len(candidates)} candidates from {candidates_path}\n")
    print(f"{'component':20} {'min':>6} {'p25':>6} {'median':>7} {'p75':>6} {'max':>6}")
    print('-' * 60)
    for comp in components:
        vals = [r[comp] for r in results]
        print(
            f"{comp:20} "
            f"{min(vals):6.3f} {pctile(vals, 0.25):6.3f} "
            f"{pctile(vals, 0.50):7.3f} {pctile(vals, 0.75):6.3f} "
            f"{max(vals):6.3f}"
        )

    print("\nSample candidates:")
    for c, r in list(zip(candidates, results))[:3]:
        print('-' * 60)
        print(f"  candidate_id : {r['candidate_id']}")
        print(f"  location     : {get_location(c)!r} ({get_country(c)!r}), "
              f"relocate={get_willing_to_relocate(c)}")
        print(f"  yoe          : {get_yoe(c)}")
        print(f"  work_mode    : {get_preferred_work_mode(c)!r}")
        print(f"  company      : {get_current_company(c)!r}")
        print(f"    location_score     = {r['location_score']}")
        print(f"    yoe_score          = {r['yoe_score']}")
        print(f"    work_mode_score    = {r['work_mode_score']}")
        print(f"    consulting_penalty = {r['consulting_penalty']}")
        print(f"    hard_filter_score  = {r['hard_filter_score']:.4f}")
