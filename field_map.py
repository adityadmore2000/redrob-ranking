"""
field_map.py — single source of truth for all field paths
Both Track 1 and Track 2 import from here.
Never access candidate fields directly in scoring scripts — always use these accessors.
"""

# ── top level accessors ──
def get_profile(c):       return c.get('profile', {}) or {}
def get_signals(c):       return c.get('redrob_signals', {}) or {}
def get_skills(c):        return c.get('skills', []) or []
def get_education(c):     return c.get('education', []) or []
def get_career(c):        return c.get('career_history', []) or []
def get_certifications(c):return c.get('certifications', []) or []

# ════════════════════════════════════════════════════════
# PROFILE FIELDS
# ════════════════════════════════════════════════════════

def get_candidate_id(c):
    return c.get('candidate_id')

def get_current_title(c):
    return get_profile(c).get('current_title', '') or ''

def get_current_company(c):
    return get_profile(c).get('current_company', '') or ''

def get_current_industry(c):
    return get_profile(c).get('current_industry', '') or ''

def get_current_company_size(c):
    return get_profile(c).get('current_company_size', '') or ''

def get_headline(c):
    return get_profile(c).get('headline', '') or ''

def get_summary(c):
    return get_profile(c).get('summary', '') or ''

def get_location(c):
    return get_profile(c).get('location', '') or ''

def get_country(c):
    return get_profile(c).get('country', '') or ''

def get_yoe(c):
    return get_profile(c).get('years_of_experience') or 0.0

def get_willing_to_relocate(c):
    # lives inside redrob_signals per EDA
    return get_signals(c).get('willing_to_relocate', False)

# ════════════════════════════════════════════════════════
# SKILLS FIELDS
# ════════════════════════════════════════════════════════

def get_skill_names(c):
    """List of skill name strings."""
    return [s.get('name', '') for s in get_skills(c) if isinstance(s, dict)]

def get_skill_name_proficiency(c):
    """List of (name, proficiency, duration_months) tuples."""
    return [
        (s.get('name', ''), s.get('proficiency', ''), s.get('duration_months', 0))
        for s in get_skills(c) if isinstance(s, dict)
    ]

# ════════════════════════════════════════════════════════
# EDUCATION FIELDS
# ════════════════════════════════════════════════════════

def get_education_entries(c):
    """List of dicts with degree, field_of_study, institution, tier."""
    return [
        {
            'institution':    e.get('institution', ''),
            'degree':         e.get('degree', ''),
            'field_of_study': e.get('field_of_study', ''),
            'tier':           e.get('tier', ''),
            'end_year':       e.get('end_year'),
        }
        for e in get_education(c) if isinstance(e, dict)
    ]

def get_education_tier(c):
    """Return highest education tier (tier_1 > tier_2 > tier_3 > tier_4)."""
    tier_rank = {'tier_1': 1, 'tier_2': 2, 'tier_3': 3, 'tier_4': 4}
    tiers = [e.get('tier', '') for e in get_education(c) if isinstance(e, dict)]
    ranked = [tier_rank[t] for t in tiers if t in tier_rank]
    return min(ranked) if ranked else 4  # lower = better

# ════════════════════════════════════════════════════════
# CAREER HISTORY FIELDS
# ════════════════════════════════════════════════════════

def get_career_sorted(c):
    """
    Career history sorted by start_date descending (most recent first).
    Each entry: title, company, industry, duration_months, description, is_current
    """
    history = get_career(c)
    try:
        history_sorted = sorted(
            history,
            key=lambda x: x.get('start_date', '') or '',
            reverse=True
        )
    except Exception:
        history_sorted = history
    return [
        {
            'title':           job.get('title', '') or '',
            'company':         job.get('company', '') or '',
            'industry':        job.get('industry', '') or '',
            'duration_months': job.get('duration_months') or 0,
            'description':     job.get('description', '') or '',
            'is_current':      job.get('is_current', False),
            'start_date':      job.get('start_date', '') or '',
            'end_date':        job.get('end_date', '') or '',
            'company_size':    job.get('company_size', '') or '',
        }
        for job in history_sorted
    ]

def get_current_company_from_career(c):
    """Most recent company from career history (fallback for profile.current_company)."""
    career = get_career_sorted(c)
    return career[0]['company'] if career else ''

# ════════════════════════════════════════════════════════
# REDROB SIGNALS — TRACK 1 AVAILABILITY
# ════════════════════════════════════════════════════════

def get_open_to_work(c):
    return get_signals(c).get('open_to_work_flag', False)

def get_last_active_date(c):
    return get_signals(c).get('last_active_date', '') or ''

def get_avg_response_time(c):
    return get_signals(c).get('avg_response_time_hours') or 280.0

def get_recruiter_response_rate(c):
    return get_signals(c).get('recruiter_response_rate') or 0.0

def get_applications_30d(c):
    return get_signals(c).get('applications_submitted_30d') or 0

def get_interview_completion_rate(c):
    return get_signals(c).get('interview_completion_rate') or 0.0

def get_notice_period(c):
    return get_signals(c).get('notice_period_days') or 90

def get_preferred_work_mode(c):
    return get_signals(c).get('preferred_work_mode', '') or ''

# ════════════════════════════════════════════════════════
# REDROB SIGNALS — TRACK 1 CREDIBILITY
# ════════════════════════════════════════════════════════

def get_github_raw(c):
    """Raw github score including -1 sentinel."""
    return get_signals(c).get('github_activity_score', -1)

def get_has_github(c):
    """True if candidate has a real GitHub score (not sentinel -1)."""
    return get_github_raw(c) != -1

def get_github_score(c):
    """Real github score or None if no GitHub account."""
    raw = get_github_raw(c)
    return raw if raw != -1 else None

def get_endorsements(c):
    return get_signals(c).get('endorsements_received') or 0

def get_profile_completeness(c):
    return get_signals(c).get('profile_completeness_score') or 0.0

def get_linkedin_connected(c):
    return get_signals(c).get('linkedin_connected', False)

def get_verified_email(c):
    return get_signals(c).get('verified_email', False)

def get_verified_phone(c):
    return get_signals(c).get('verified_phone', False)

def get_skill_assessment_scores(c):
    """Dict of skill_name -> score, or empty dict if not present."""
    return get_signals(c).get('skill_assessment_scores') or {}

# ════════════════════════════════════════════════════════
# CONSTANTS — used across both tracks
# ════════════════════════════════════════════════════════

# Tier-1 cities confirmed from EDA
TIER_1_CITIES = {
    'bangalore', 'bengaluru', 'mumbai', 'delhi', 'new delhi',
    'hyderabad', 'pune', 'chennai', 'gurgaon', 'gurugram', 'noida'
}

# Consulting firms explicitly disqualified in JD
CONSULTING_FIRMS = {
    'tcs', 'infosys', 'wipro', 'accenture', 'cognizant', 'capgemini'
}

# last_active_date normalization range from EDA (days ago)
LAST_ACTIVE_MIN_DAYS = 23
LAST_ACTIVE_MAX_DAYS = 263

# avg_response_time normalization range from EDA
RESPONSE_TIME_MIN = 2.1
RESPONSE_TIME_MAX = 280.0

# github score normalization range from EDA (real scores only)
GITHUB_SCORE_MIN = 0.0
GITHUB_SCORE_MAX = 96.9

# notice period tiers
NOTICE_PERIOD_SCORES = {
    0:   1.0,
    15:  1.0,
    30:  1.0,
    45:  0.90,
    60:  0.85,
    90:  0.70,
    120: 0.55,
    150: 0.40,
}

# average job tenure thresholds (months) + scores for title-chasing check
TENURE_IDEAL_MONTHS = 24
TENURE_STABLE_MONTHS = 18
TENURE_MODERATE_MONTHS = 12
TENURE_SCORE_IDEAL = 1.0
TENURE_SCORE_STABLE = 0.85
TENURE_SCORE_MODERATE = 0.70
TENURE_SCORE_SHORT = 0.50
TENURE_SCORE_UNKNOWN = 0.75

# education tier scores for credibility
EDUCATION_TIER_SCORES = {
    1: 1.0,   # tier_1
    2: 0.85,  # tier_2
    3: 0.70,  # tier_3
    4: 0.55,  # tier_4
}

# Track 2 token budget per job position (from EDA section 5)
CAREER_TOKEN_BUDGET = {
    0: 125,   # full description
    1: 94,    # medium detail
    2: 62,    # brief — first sentence only
    3: 31,    # one-liner — title + company only
}
