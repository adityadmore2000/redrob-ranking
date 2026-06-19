import json
import numpy as np
from collections import defaultdict, Counter
from datetime import datetime, date

# ── load candidates ──
candidates = []
with open('/kaggle/input/datasets/moreadityad/candidate-dataset/candidates.jsonl', 'r') as f:
    for line in f:
        candidates.append(json.loads(line.strip()))

print(f"Loaded {len(candidates):,} candidates\n")

# ── helpers ──
def p(arr, pct): return np.percentile(arr, pct)
def safe_get(c, *keys):
    val = c
    for k in keys:
        if not isinstance(val, dict):
            return None
        val = val.get(k)
    return val

def days_since(date_str):
    if not date_str:
        return None
    try:
        d = datetime.strptime(date_str[:10], "%Y-%m-%d").date()
        return (date.today() - d).days
    except:
        return None

print("=" * 60)
print("SECTION 1 — PROFILE FIELDS")
print("=" * 60)

# ── years of experience ──
yoe = [safe_get(c, 'profile', 'years_of_experience') for c in candidates]
yoe = [v for v in yoe if v is not None]
yoe = np.array(yoe)
print(f"\n[1.1] years_of_experience (n={len(yoe):,}):")
print(f"    min={yoe.min():.1f}, p25={p(yoe,25):.1f}, median={p(yoe,50):.1f}, "
      f"p75={p(yoe,75):.1f}, max={yoe.max():.1f}, mean={yoe.mean():.1f}")

# distribution by bucket
buckets = [(0,2),(2,4),(4,6),(6,8),(8,10),(10,12),(12,100)]
print(f"    YOE distribution:")
for lo, hi in buckets:
    count = sum(1 for v in yoe if lo <= v < hi)
    bar = "█" * min(40, int(40 * count / len(yoe)))
    print(f"      {lo:2d}-{hi:2d} yrs: {count:6,}  {bar}")

# ── location ──
print(f"\n[1.2] location / country:")
countries = [safe_get(c, 'profile', 'country') for c in candidates]
country_counts = Counter(countries)
print(f"    Top countries:")
for country, count in country_counts.most_common(10):
    print(f"      {str(country):<30} {count:6,}  ({100*count/len(candidates):.1f}%)")

locations = [safe_get(c, 'profile', 'location') for c in candidates]
location_counts = Counter(locations)
print(f"    Top cities:")
for loc, count in location_counts.most_common(15):
    print(f"      {str(loc):<30} {count:6,}  ({100*count/len(candidates):.1f}%)")

# ── current title / company / industry ──
print(f"\n[1.3] current_title (top 15):")
titles = [safe_get(c, 'profile', 'current_title') for c in candidates]
for t, n in Counter(titles).most_common(15):
    print(f"      {str(t):<40} {n:6,}")

print(f"\n[1.4] current_industry (top 15):")
industries = [safe_get(c, 'profile', 'current_industry') for c in candidates]
for ind, n in Counter(industries).most_common(15):
    print(f"      {str(ind):<40} {n:6,}")

print(f"\n[1.5] current_company — consulting firm presence:")
consulting = ['TCS','Infosys','Wipro','Accenture','Cognizant','Capgemini']
companies = [safe_get(c, 'profile', 'current_company') or '' for c in candidates]
for firm in consulting:
    count = sum(1 for co in companies if firm.lower() in co.lower())
    print(f"      {firm:<20} {count:6,}  ({100*count/len(candidates):.1f}%)")
total_consulting = sum(
    1 for co in companies
    if any(firm.lower() in co.lower() for firm in consulting)
)
print(f"      {'TOTAL (any firm)':<20} {total_consulting:6,}  ({100*total_consulting/len(candidates):.1f}%)")

# ── summary ──
print(f"\n[1.6] profile.summary:")
summaries = [safe_get(c, 'profile', 'summary') or '' for c in candidates]
summary_words = [len(s.split()) for s in summaries]
sw = np.array(summary_words)
present = sum(1 for s in summaries if s.strip())
print(f"    present: {present:,} / {len(candidates):,} ({100*present/len(candidates):.1f}%)")
print(f"    word count — min={sw.min()}, p25={p(sw,25):.0f}, median={p(sw,50):.0f}, "
      f"p75={p(sw,75):.0f}, max={sw.max()}")

# ── headline ──
print(f"\n[1.7] profile.headline:")
headlines = [safe_get(c, 'profile', 'headline') or '' for c in candidates]
present = sum(1 for h in headlines if h.strip())
hl_words = [len(h.split()) for h in headlines]
hw = np.array(hl_words)
print(f"    present: {present:,} / {len(candidates):,} ({100*present/len(candidates):.1f}%)")
print(f"    word count — min={hw.min()}, p25={p(hw,25):.0f}, median={p(hw,50):.0f}, "
      f"p75={p(hw,75):.0f}, max={hw.max()}")
print(f"    samples:")
for h in headlines[:5]:
    print(f"      {h[:100]}")

print("\n" + "=" * 60)
print("SECTION 2 — SKILLS")
print("=" * 60)

# ── skills ──
skill_counts = []
proficiency_counts = Counter()
duration_months_all = []
top_skill_names = Counter()

for c in candidates:
    skills = c.get('skills', []) or []
    skill_counts.append(len(skills))
    for s in skills:
        if isinstance(s, dict):
            proficiency_counts[s.get('proficiency', 'unknown')] += 1
            dm = s.get('duration_months')
            if dm is not None:
                duration_months_all.append(dm)
            name = s.get('name', '')
            if name:
                top_skill_names[name] += 1

sc = np.array(skill_counts)
print(f"\n[2.1] Skills per candidate:")
print(f"    min={sc.min()}, p25={p(sc,25):.0f}, median={p(sc,50):.0f}, "
      f"p75={p(sc,75):.0f}, max={sc.max()}")

print(f"\n[2.2] Proficiency distribution:")
for prof, count in proficiency_counts.most_common():
    print(f"    {prof:<15} {count:7,}  ({100*count/sum(proficiency_counts.values()):.1f}%)")

dm = np.array(duration_months_all)
print(f"\n[2.3] Skill duration_months:")
print(f"    min={dm.min()}, p25={p(dm,25):.0f}, median={p(dm,50):.0f}, "
      f"p75={p(dm,75):.0f}, max={dm.max()}")

print(f"\n[2.4] Top 20 skills:")
for skill, count in top_skill_names.most_common(20):
    print(f"    {skill:<35} {count:6,}")

print("\n" + "=" * 60)
print("SECTION 3 — EDUCATION")
print("=" * 60)

tiers = Counter()
degrees = Counter()
for c in candidates:
    edu = c.get('education', []) or []
    for e in (edu if isinstance(edu, list) else []):
        tiers[e.get('tier', 'unknown')] += 1
        degrees[e.get('degree', 'unknown')] += 1

print(f"\n[3.1] Education tier distribution:")
for tier, count in tiers.most_common():
    print(f"    {str(tier):<15} {count:7,}  ({100*count/sum(tiers.values()):.1f}%)")

print(f"\n[3.2] Degree distribution (top 10):")
for deg, count in degrees.most_common(10):
    print(f"    {str(deg):<20} {count:7,}")

print("\n" + "=" * 60)
print("SECTION 4 — REDROB SIGNALS (TRACK 1 FIELDS)")
print("=" * 60)

# ── github ──
github_scores = [safe_get(c, 'redrob_signals', 'github_activity_score') for c in candidates]
github_scores = [v for v in github_scores if v is not None]
gs = np.array(github_scores)
has_github = sum(1 for v in github_scores if v != -1)
print(f"\n[4.1] github_activity_score (n={len(gs):,}):")
print(f"    sentinel (-1): {sum(1 for v in github_scores if v == -1):,}  "
      f"({100*sum(1 for v in github_scores if v == -1)/len(gs):.1f}%)")
print(f"    has github   : {has_github:,}  ({100*has_github/len(gs):.1f}%)")
real_scores = np.array([v for v in github_scores if v != -1])
if len(real_scores):
    print(f"    real scores  — min={real_scores.min():.1f}, p25={p(real_scores,25):.1f}, "
          f"median={p(real_scores,50):.1f}, p75={p(real_scores,75):.1f}, max={real_scores.max():.1f}")

# ── notice period ──
notice = [safe_get(c, 'redrob_signals', 'notice_period_days') for c in candidates]
notice = [v for v in notice if v is not None]
print(f"\n[4.2] notice_period_days (n={len(notice):,}):")
for val, count in sorted(Counter(notice).items()):
    bar = "█" * min(40, int(40 * count / len(notice)))
    print(f"    {val:3d} days: {count:6,}  {bar}")

# ── open to work ──
otw = [safe_get(c, 'redrob_signals', 'open_to_work_flag') for c in candidates]
print(f"\n[4.3] open_to_work_flag:")
for val, count in Counter(otw).most_common():
    print(f"    {str(val):<10} {count:6,}  ({100*count/len(candidates):.1f}%)")

# ── last active date ──
last_active = [safe_get(c, 'redrob_signals', 'last_active_date') for c in candidates]
days_ago = [days_since(d) for d in last_active if d]
days_ago = [d for d in days_ago if d is not None]
da = np.array(days_ago)
print(f"\n[4.4] last_active_date — days since active (n={len(da):,}):")
print(f"    min={da.min()}, p25={p(da,25):.0f}, median={p(da,50):.0f}, "
      f"p75={p(da,75):.0f}, p90={p(da,90):.0f}, max={da.max()}")
buckets_days = [(0,7),(7,30),(30,90),(90,180),(180,365),(365,9999)]
print(f"    Recency buckets:")
for lo, hi in buckets_days:
    count = sum(1 for d in days_ago if lo <= d < hi)
    label = f"{lo}-{hi}d" if hi < 9999 else f"{lo}d+"
    print(f"      {label:<12} {count:6,}  ({100*count/len(da):.1f}%)")

# ── avg response time ──
rt = [safe_get(c, 'redrob_signals', 'avg_response_time_hours') for c in candidates]
rt = [v for v in rt if v is not None]
rt = np.array(rt)
print(f"\n[4.5] avg_response_time_hours (n={len(rt):,}):")
print(f"    min={rt.min():.1f}, p25={p(rt,25):.1f}, median={p(rt,50):.1f}, "
      f"p75={p(rt,75):.1f}, p90={p(rt,90):.1f}, max={rt.max():.1f}")

# ── recruiter response rate ──
rr = [safe_get(c, 'redrob_signals', 'recruiter_response_rate') for c in candidates]
rr = [v for v in rr if v is not None]
rr = np.array(rr)
print(f"\n[4.6] recruiter_response_rate (n={len(rr):,}):")
print(f"    min={rr.min():.2f}, p25={p(rr,25):.2f}, median={p(rr,50):.2f}, "
      f"p75={p(rr,75):.2f}, max={rr.max():.2f}")

# ── applications submitted 30d ──
apps = [safe_get(c, 'redrob_signals', 'applications_submitted_30d') for c in candidates]
apps = [v for v in apps if v is not None]
apps = np.array(apps)
print(f"\n[4.7] applications_submitted_30d (n={len(apps):,}):")
print(f"    min={apps.min()}, p25={p(apps,25):.0f}, median={p(apps,50):.0f}, "
      f"p75={p(apps,75):.0f}, max={apps.max()}")
print(f"    zero apps: {sum(1 for v in apps if v == 0):,}  "
      f"({100*sum(1 for v in apps if v == 0)/len(apps):.1f}%)")

# ── interview completion rate ──
icr = [safe_get(c, 'redrob_signals', 'interview_completion_rate') for c in candidates]
icr = [v for v in icr if v is not None]
icr = np.array(icr)
print(f"\n[4.8] interview_completion_rate (n={len(icr):,}):")
print(f"    min={icr.min():.2f}, p25={p(icr,25):.2f}, median={p(icr,50):.2f}, "
      f"p75={p(icr,75):.2f}, max={icr.max():.2f}")

# ── endorsements ──
end = [safe_get(c, 'redrob_signals', 'endorsements_received') for c in candidates]
end = [v for v in end if v is not None]
end = np.array(end)
print(f"\n[4.9] endorsements_received (n={len(end):,}):")
print(f"    min={end.min()}, p25={p(end,25):.0f}, median={p(end,50):.0f}, "
      f"p75={p(end,75):.0f}, p90={p(end,90):.0f}, max={end.max()}")

# ── profile completeness ──
pcs = [safe_get(c, 'redrob_signals', 'profile_completeness_score') for c in candidates]
pcs = [v for v in pcs if v is not None]
pcs = np.array(pcs)
print(f"\n[4.10] profile_completeness_score (n={len(pcs):,}):")
print(f"    min={pcs.min():.1f}, p25={p(pcs,25):.1f}, median={p(pcs,50):.1f}, "
      f"p75={p(pcs,75):.1f}, max={pcs.max():.1f}")

# ── linkedin / verified ──
linkedin = [safe_get(c, 'redrob_signals', 'linkedin_connected') for c in candidates]
print(f"\n[4.11] linkedin_connected:")
for val, count in Counter(linkedin).most_common():
    print(f"    {str(val):<10} {count:6,}  ({100*count/len(candidates):.1f}%)")

v_email = [safe_get(c, 'redrob_signals', 'verified_email') for c in candidates]
v_phone = [safe_get(c, 'redrob_signals', 'verified_phone') for c in candidates]
print(f"\n[4.12] verified_email  — True: {sum(1 for v in v_email if v):,}  "
      f"({100*sum(1 for v in v_email if v)/len(candidates):.1f}%)")
print(f"       verified_phone  — True: {sum(1 for v in v_phone if v):,}  "
      f"({100*sum(1 for v in v_phone if v)/len(candidates):.1f}%)")

# ── preferred work mode ──
pwm = [safe_get(c, 'redrob_signals', 'preferred_work_mode') for c in candidates]
print(f"\n[4.13] preferred_work_mode:")
for val, count in Counter(pwm).most_common():
    print(f"    {str(val):<15} {count:6,}  ({100*count/len(candidates):.1f}%)")

# ── willing to relocate ──
wtr = [safe_get(c, 'profile', 'willing_to_relocate') 
       or safe_get(c, 'redrob_signals', 'willing_to_relocate') for c in candidates]
print(f"\n[4.14] willing_to_relocate:")
for val, count in Counter(wtr).most_common():
    print(f"    {str(val):<10} {count:6,}  ({100*count/len(candidates):.1f}%)")

# ── skill assessment scores ──
has_assessment = sum(
    1 for c in candidates
    if safe_get(c, 'redrob_signals', 'skill_assessment_scores')
)
print(f"\n[4.15] skill_assessment_scores:")
print(f"    present: {has_assessment:,} / {len(candidates):,}  "
      f"({100*has_assessment/len(candidates):.1f}%)")

print("\n" + "=" * 60)
print("SECTION 5 — TOKEN BUDGET (CORRECTED)")
print("=" * 60)

def approx_tokens(text):
    if not text:
        return 0
    return max(1, int(len(str(text)) / 4))

title_tok, summary_tok, headline_tok, skills_tok, edu_tok = [], [], [], [], []

for c in candidates:
    profile = c.get('profile', {}) or {}
    signals = c.get('redrob_signals', {}) or {}

    title_tok.append(approx_tokens(
        f"{profile.get('current_title','')} at {profile.get('current_company','')} "
        f"({profile.get('current_industry','')})"
    ))
    summary_tok.append(approx_tokens(profile.get('summary', '') or ''))
    headline_tok.append(approx_tokens(profile.get('headline', '') or ''))

    skills = c.get('skills', []) or []
    skills_text = ", ".join(
        f"{s.get('name','')} ({s.get('proficiency','')})"
        for s in skills if isinstance(s, dict)
    )
    skills_tok.append(approx_tokens(skills_text))

    edu = c.get('education', []) or []
    edu_text = " ".join([
        f"{e.get('degree','')} {e.get('field_of_study','')} {e.get('institution','')} {e.get('tier','')}"
        for e in (edu if isinstance(edu, list) else [])
    ])
    edu_tok.append(approx_tokens(edu_text))

def med(arr): return int(np.median(arr))

print(f"\n[5.1] Field token estimates (median):")
print(f"    current role block : {med(title_tok):>4} tokens")
print(f"    headline           : {med(headline_tok):>4} tokens")
print(f"    summary            : {med(summary_tok):>4} tokens")
print(f"    skills             : {med(skills_tok):>4} tokens")
print(f"    education          : {med(edu_tok):>4} tokens")

fixed = med(title_tok) + med(headline_tok) + med(summary_tok) + med(skills_tok) + med(edu_tok)
career_budget = 512 - fixed
print(f"\n    Fixed fields total (median) : {fixed} tokens")
print(f"    Remaining for career history: {career_budget} tokens")

# career history: ~99 tokens per job (from previous EDA)
jobs_fit = career_budget // 99
print(f"    Full descriptions that fit  : ~{jobs_fit} jobs at 99 tokens each")

weights = [0.40, 0.30, 0.20, 0.10]
labels  = ["full detail", "medium detail", "brief", "one-liner"]
print(f"\n    Suggested token budget per job:")
for i, (w, label) in enumerate(zip(weights, labels)):
    print(f"      Job {i} ({label:<14}): ~{int(career_budget * w):>3} tokens")

print(f"\n{'='*60}")
print("EDA complete.")
print(f"{'='*60}\n")
