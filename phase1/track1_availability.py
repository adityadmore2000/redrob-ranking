"""
track1_availability.py
=======================
Computes availability_score for each candidate from Redrob behavioral signals.

Formula
-------
availability_score = weighted_average(
    open_to_work        weight=0.30
    last_active         weight=0.25
    response_time       weight=0.20
    notice_period       weight=0.15
    recruiter_response  weight=0.05
    applications_30d    weight=0.05
)

interview_completion_rate is intentionally excluded: EDA showed a very narrow
range (0.30–1.00, p25=0.48, p75=0.76) making it a weak discriminator.

Sub-signals
-----------
open_to_work        : open_to_work_flag.
                      True  = 1.0
                      False = 0.5

last_active         : Recency of last_active_date, normalized within the
                      observed range [LAST_ACTIVE_MIN_DAYS, LAST_ACTIVE_MAX_DAYS].
                      score = 1.0 - (days_since - MIN) / (MAX - MIN), clamped.
                      More recent = higher.

response_time       : avg_response_time_hours, inverted and normalized within
                      [RESPONSE_TIME_MIN, RESPONSE_TIME_MAX].
                      score = 1.0 - (hours - MIN) / (MAX - MIN), clamped.
                      Faster = higher.

notice_period       : notice_period_days looked up in NOTICE_PERIOD_SCORES;
                      nearest key used if the exact value is absent.

recruiter_response  : recruiter_response_rate, already in [0.0, 1.0].

applications_30d    : applications_submitted_30d.
                      0 apps    = 0.3 (not actively looking)
                      1-24 apps = normalized within [1, 24] to [0.5, 1.0]

All field access goes through field_map accessors.
All thresholds come from field_map constants.

Public API
----------
compute_availability_score(c)  -> dict with sub-signal values + final score
compute_all(candidates)        -> list of dicts with candidate_id + components
"""

from datetime import date

from field_map import (
    get_candidate_id,
    get_open_to_work,
    get_last_active_date,
    get_avg_response_time,
    get_notice_period,
    get_recruiter_response_rate,
    get_applications_30d,
    LAST_ACTIVE_MIN_DAYS,
    LAST_ACTIVE_MAX_DAYS,
    RESPONSE_TIME_MIN,
    RESPONSE_TIME_MAX,
    NOTICE_PERIOD_SCORES,
)

# Weights for the six sub-signals (sum to 1.0).
WEIGHTS = {
    'open_to_work': 0.30,
    'last_active': 0.25,
    'response_time': 0.20,
    'notice_period': 0.15,
    'recruiter_response': 0.05,
    'applications_30d': 0.05,
}


def _clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def open_to_work_signal(c):
    """open_to_work_flag True -> 1.0 else 0.5."""
    return 1.0 if get_open_to_work(c) else 0.5


def last_active_signal(c):
    """
    Recency of last_active_date normalized within [MIN, MAX] days.
    score = 1.0 - (days_since - MIN) / (MAX - MIN), clamped to [0, 1].
    A missing/unparseable date is treated as stale -> 0.0.
    """
    raw = get_last_active_date(c)
    if not raw:
        return 0.0
    try:
        last = date.fromisoformat(raw[:10])
    except (ValueError, TypeError):
        return 0.0
    days_since = (date.today() - last).days
    score = 1.0 - (days_since - LAST_ACTIVE_MIN_DAYS) / (
        LAST_ACTIVE_MAX_DAYS - LAST_ACTIVE_MIN_DAYS
    )
    return _clamp(score)


def response_time_signal(c):
    """
    avg_response_time_hours inverted and normalized within [MIN, MAX].
    score = 1.0 - (hours - MIN) / (MAX - MIN), clamped to [0, 1].
    """
    hours = get_avg_response_time(c)
    score = 1.0 - (hours - RESPONSE_TIME_MIN) / (
        RESPONSE_TIME_MAX - RESPONSE_TIME_MIN
    )
    return _clamp(score)


def notice_period_signal(c):
    """
    Look up notice_period_days in NOTICE_PERIOD_SCORES. If the exact value is
    absent, use the score of the nearest key.
    """
    days = get_notice_period(c)
    if days in NOTICE_PERIOD_SCORES:
        return NOTICE_PERIOD_SCORES[days]
    nearest = min(NOTICE_PERIOD_SCORES, key=lambda k: abs(k - days))
    return NOTICE_PERIOD_SCORES[nearest]


def recruiter_response_signal(c):
    """recruiter_response_rate, already in [0.0, 1.0]."""
    return _clamp(get_recruiter_response_rate(c))


def applications_30d_signal(c):
    """
    0 apps    -> 0.3 (not actively looking)
    1-24 apps -> normalized within [1, 24] to [0.5, 1.0]
    25+ apps  -> 1.0 (clamped top of band)
    """
    apps = get_applications_30d(c)
    if apps <= 0:
        return 0.3
    score = 0.5 + (apps - 1) / (24 - 1) * (1.0 - 0.5)
    return _clamp(score, 0.5, 1.0)


def compute_availability_score(c: dict) -> dict:
    """
    Returns a dict with:
    {
        'availability_score': float,        # final weighted average
        'open_to_work_signal': float,
        'last_active_signal': float,
        'response_time_signal': float,
        'notice_period_signal': float,
        'recruiter_response_signal': float,
        'applications_30d_signal': float,
    }
    """
    signals = {
        'open_to_work': open_to_work_signal(c),
        'last_active': last_active_signal(c),
        'response_time': response_time_signal(c),
        'notice_period': notice_period_signal(c),
        'recruiter_response': recruiter_response_signal(c),
        'applications_30d': applications_30d_signal(c),
    }
    availability = sum(signals[k] * WEIGHTS[k] for k in WEIGHTS)
    return {
        'availability_score': availability,
        'open_to_work_signal': signals['open_to_work'],
        'last_active_signal': signals['last_active'],
        'response_time_signal': signals['response_time'],
        'notice_period_signal': signals['notice_period'],
        'recruiter_response_signal': signals['recruiter_response'],
        'applications_30d_signal': signals['applications_30d'],
    }


def compute_all(candidates: list) -> list:
    """
    Takes full list of candidate dicts.
    Returns list of dicts with candidate_id + all score components.
    """
    results = []
    for c in candidates:
        scores = compute_availability_score(c)
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
        'open_to_work_signal', 'last_active_signal', 'response_time_signal',
        'notice_period_signal', 'recruiter_response_signal',
        'applications_30d_signal', 'availability_score',
    ]

    print(f"Loaded {len(candidates)} candidates from {candidates_path}\n")
    print(f"{'component':26} {'min':>6} {'p25':>6} {'median':>7} {'p75':>6} {'max':>6}")
    print('-' * 66)
    for comp in components:
        vals = [r[comp] for r in results]
        print(
            f"{comp:26} "
            f"{min(vals):6.3f} {pctile(vals, 0.25):6.3f} "
            f"{pctile(vals, 0.50):7.3f} {pctile(vals, 0.75):6.3f} "
            f"{max(vals):6.3f}"
        )

    print("\nSample candidates:")
    for c, r in list(zip(candidates, results))[:3]:
        print('-' * 66)
        print(f"  candidate_id  : {r['candidate_id']}")
        print(f"  open_to_work  : {get_open_to_work(c)}")
        print(f"  last_active   : {get_last_active_date(c)!r}")
        print(f"  response_hrs  : {get_avg_response_time(c)}")
        print(f"  notice_days   : {get_notice_period(c)}")
        print(f"  recruiter_rr  : {get_recruiter_response_rate(c)}")
        print(f"  apps_30d      : {get_applications_30d(c)}")
        print(f"    open_to_work_signal       = {r['open_to_work_signal']}")
        print(f"    last_active_signal        = {r['last_active_signal']:.4f}")
        print(f"    response_time_signal      = {r['response_time_signal']:.4f}")
        print(f"    notice_period_signal      = {r['notice_period_signal']}")
        print(f"    recruiter_response_signal = {r['recruiter_response_signal']}")
        print(f"    applications_30d_signal   = {r['applications_30d_signal']:.4f}")
        print(f"    availability_score        = {r['availability_score']:.4f}")
