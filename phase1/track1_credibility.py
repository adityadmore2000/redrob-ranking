"""
track1_credibility.py
=====================
Computes credibility_score for each candidate from profile and trust signals.

Formula
-------
credibility_score = weighted_average(
    profile_completeness  weight=0.25
    endorsements          weight=0.20
    education_tier        weight=0.20
    github                weight=0.20
    linkedin              weight=0.10
    verified_contact      weight=0.05
)

skill_assessment_scores is intentionally excluded as a standalone signal:
only ~24.2% of candidates have it, making it too sparse to weight directly.
Instead it is folded into profile_completeness as a small upward nudge for the
candidates who do have assessment data.

Sub-signals
-----------
profile_completeness : profile_completeness_score normalized within [25, 100]
                       score = (raw - 25) / (100 - 25)
                       Plus a skill-assessment nudge when present:
                         avg = mean(skill_assessment_scores.values())
                         nudge = (avg / 100) * 0.05
                         final = min(1.0, base + nudge)

endorsements         : endorsements_received, log-scaled for the long tail.
                       score = log(1 + endorsements) / log(1 + 242)

education_tier       : get_education_tier() (1=best .. 4=worst) looked up in
                       EDUCATION_TIER_SCORES.

github               : no GitHub        -> 0.0
                       has GitHub        -> github_activity_score normalized
                       within [GITHUB_SCORE_MIN, GITHUB_SCORE_MAX], clamped.

linkedin             : linkedin_connected True -> 1.0 else 0.0

verified_contact     : both email+phone verified -> 1.0
                       one verified              -> 0.6
                       none verified             -> 0.2

All field access goes through field_map accessors.
All thresholds come from field_map constants.

Public API
----------
compute_credibility_score(c)  -> dict with sub-signal values + final score
compute_all(candidates)       -> list of dicts with candidate_id + components
"""

import math

from field_map import (
    get_candidate_id,
    get_profile_completeness,
    get_skill_assessment_scores,
    get_endorsements,
    get_education_tier,
    get_has_github,
    get_github_score,
    get_linkedin_connected,
    get_verified_email,
    get_verified_phone,
    EDUCATION_TIER_SCORES,
    GITHUB_SCORE_MIN,
    GITHUB_SCORE_MAX,
)

# Weights for the six sub-signals (sum to 1.0).
WEIGHTS = {
    'profile_completeness': 0.25,
    'endorsements': 0.20,
    'education_tier': 0.20,
    'github': 0.20,
    'linkedin': 0.10,
    'verified_contact': 0.05,
}

# Normalization range for profile_completeness_score (observed [25, 100]).
PROFILE_COMPLETENESS_MIN = 25
PROFILE_COMPLETENESS_MAX = 100

# Max endorsements used as the log-scale denominator (observed long-tail max).
ENDORSEMENTS_MAX = 242


def _clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def profile_completeness_signal(c):
    """
    Normalize profile_completeness_score within [25, 100], then add a small
    nudge from skill assessment scores when the candidate has them.
    """
    raw = get_profile_completeness(c)
    base = (raw - PROFILE_COMPLETENESS_MIN) / (
        PROFILE_COMPLETENESS_MAX - PROFILE_COMPLETENESS_MIN
    )
    base = _clamp(base)

    assessments = get_skill_assessment_scores(c)
    if assessments:
        avg_assessment = sum(assessments.values()) / len(assessments)
        nudge = (avg_assessment / 100) * 0.05
        return min(1.0, base + nudge)
    return base


def endorsements_signal(c):
    """Log-scale endorsements: log(1 + n) / log(1 + 242)."""
    n = get_endorsements(c)
    return _clamp(math.log(1 + n) / math.log(1 + ENDORSEMENTS_MAX))


def education_tier_signal(c):
    """Look up education tier (1=best .. 4=worst) in EDUCATION_TIER_SCORES."""
    tier = get_education_tier(c)
    return EDUCATION_TIER_SCORES.get(tier, EDUCATION_TIER_SCORES[4])


def github_signal(c):
    """
    No GitHub -> 0.0. Otherwise normalize github_activity_score within
    [GITHUB_SCORE_MIN, GITHUB_SCORE_MAX], clamped to [0, 1].
    """
    if not get_has_github(c):
        return 0.0
    score = get_github_score(c)
    norm = (score - GITHUB_SCORE_MIN) / (GITHUB_SCORE_MAX - GITHUB_SCORE_MIN)
    return _clamp(norm)


def linkedin_signal(c):
    """linkedin_connected True -> 1.0 else 0.0."""
    return 1.0 if get_linkedin_connected(c) else 0.0


def verified_contact_signal(c):
    """both verified -> 1.0, one verified -> 0.6, none -> 0.2."""
    verified = int(bool(get_verified_email(c))) + int(bool(get_verified_phone(c)))
    if verified == 2:
        return 1.0
    if verified == 1:
        return 0.6
    return 0.2


def compute_credibility_score(c: dict) -> dict:
    """
    Returns a dict with:
    {
        'credibility_score': float,              # final weighted average
        'profile_completeness_signal': float,
        'endorsements_signal': float,
        'education_tier_signal': float,
        'github_signal': float,
        'linkedin_signal': float,
        'verified_contact_signal': float,
    }
    """
    signals = {
        'profile_completeness': profile_completeness_signal(c),
        'endorsements': endorsements_signal(c),
        'education_tier': education_tier_signal(c),
        'github': github_signal(c),
        'linkedin': linkedin_signal(c),
        'verified_contact': verified_contact_signal(c),
    }
    credibility = sum(signals[k] * WEIGHTS[k] for k in WEIGHTS)
    return {
        'credibility_score': credibility,
        'profile_completeness_signal': signals['profile_completeness'],
        'endorsements_signal': signals['endorsements'],
        'education_tier_signal': signals['education_tier'],
        'github_signal': signals['github'],
        'linkedin_signal': signals['linkedin'],
        'verified_contact_signal': signals['verified_contact'],
    }


def compute_all(candidates: list) -> list:
    """
    Takes full list of candidate dicts.
    Returns list of dicts with candidate_id + all score components.
    """
    results = []
    for c in candidates:
        scores = compute_credibility_score(c)
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
        'profile_completeness_signal', 'endorsements_signal',
        'education_tier_signal', 'github_signal', 'linkedin_signal',
        'verified_contact_signal', 'credibility_score',
    ]

    print(f"Loaded {len(candidates)} candidates from {candidates_path}\n")
    print(f"{'component':28} {'min':>6} {'p25':>6} {'median':>7} {'p75':>6} {'max':>6}")
    print('-' * 68)
    for comp in components:
        vals = [r[comp] for r in results]
        print(
            f"{comp:28} "
            f"{min(vals):6.3f} {pctile(vals, 0.25):6.3f} "
            f"{pctile(vals, 0.50):7.3f} {pctile(vals, 0.75):6.3f} "
            f"{max(vals):6.3f}"
        )

    print("\nSample candidates:")
    for c, r in list(zip(candidates, results))[:3]:
        assessments = get_skill_assessment_scores(c)
        avg_assessment = (
            sum(assessments.values()) / len(assessments) if assessments else None
        )
        print('-' * 68)
        print(f"  candidate_id        : {r['candidate_id']}")
        print(f"  profile_completeness: {get_profile_completeness(c)}")
        print(f"  skill_assess avg    : {avg_assessment}")
        print(f"  endorsements        : {get_endorsements(c)}")
        print(f"  education_tier      : {get_education_tier(c)}")
        print(f"  has_github          : {get_has_github(c)}  github_score: {get_github_score(c)}")
        print(f"  linkedin_connected  : {get_linkedin_connected(c)}")
        print(f"  verified email/phone: {get_verified_email(c)} / {get_verified_phone(c)}")
        print(f"    profile_completeness_signal = {r['profile_completeness_signal']:.4f}")
        print(f"    endorsements_signal         = {r['endorsements_signal']:.4f}")
        print(f"    education_tier_signal       = {r['education_tier_signal']}")
        print(f"    github_signal               = {r['github_signal']:.4f}")
        print(f"    linkedin_signal             = {r['linkedin_signal']}")
        print(f"    verified_contact_signal     = {r['verified_contact_signal']}")
        print(f"    credibility_score           = {r['credibility_score']:.4f}")
