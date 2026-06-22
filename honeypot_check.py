import json
import numpy as np
from datetime import datetime

# ── load artifacts ──────────────────────────────────────────
candidate_ids       = np.load("artifacts/candidate_ids.npy", allow_pickle=True)
semantic_scores     = np.load("artifacts/semantic_scores.npy")
hard_filter_scores  = np.load("artifacts/hard_filter_scores.npy")
availability_scores = np.load("artifacts/availability_scores.npy")
credibility_scores  = np.load("artifacts/credibility_scores.npy")

# normalize semantic
semantic_norm = (semantic_scores - semantic_scores.min()) / (semantic_scores.max() - semantic_scores.min())

# equal weighting
final_scores = semantic_norm * hard_filter_scores * availability_scores * credibility_scores

# top 100 ids
top100_idx = np.argsort(-final_scores)[:100]
top100_ids = set(candidate_ids[top100_idx])

# rank lookup
rank_lookup = {candidate_ids[idx]: int(r) + 1 for r, idx in enumerate(top100_idx)}

# ── load raw profiles for top 100 ───────────────────────────
print("Loading profiles...")
profiles = {}
with open("data/candidates.jsonl", "r") as f:
    for line in f:
        c = json.loads(line)
        cid = c.get("candidate_id") or c.get("id")
        if cid in top100_ids:
            profiles[cid] = c

print(f"Loaded {len(profiles)} profiles from top 100\n")

# ── honeypot checks ──────────────────────────────────────────
current_year = datetime.now().year
flagged = []

for cid, c in profiles.items():
    issues = []

    # check 1 — expert/advanced skill with 0 months duration
    skills = c.get("skills", [])
    for skill in skills:
        proficiency     = (skill.get("proficiency") or "").lower()
        duration_months = skill.get("duration_months")
        name            = skill.get("name", "unknown")
        if proficiency in ("expert", "advanced") and duration_months == 0:
            issues.append(
                f"skill '{name}': {proficiency} proficiency but 0 months used"
            )

    # check 2 — stated YOE vs actual career span
    profile  = c.get("profile", {})
    yoe      = profile.get("years_of_experience") or profile.get("yoe")
    career   = c.get("career_history", [])

    if yoe and career:
        start_dates = []
        for job in career:
            sd = (job.get("start_date") or "")
            if sd:
                try:
                    start_dates.append(int(str(sd)[:4]))
                except:
                    pass
        if start_dates:
            earliest    = min(start_dates)
            career_span = current_year - earliest
            if yoe > career_span + 2:
                issues.append(
                    f"stated YOE {yoe} but career history only spans "
                    f"{career_span} years (earliest job: {earliest})"
                )

    if issues:
        # attach scores and rank for context
        score_idx = np.where(candidate_ids == cid)[0][0]
        flagged.append({
            "candidate_id":      cid,
            "rank":              rank_lookup[cid],
            "final_score":       round(float(final_scores[score_idx]), 6),
            "semantic_norm":     round(float(semantic_norm[score_idx]), 6),
            "hard_filter_score": round(float(hard_filter_scores[score_idx]), 6),
            "availability_score":round(float(availability_scores[score_idx]), 6),
            "credibility_score": round(float(credibility_scores[score_idx]), 6),
            "honeypot_signals":  issues,
            "raw_profile":       c,
        })

# ── report ───────────────────────────────────────────────────
print(f"=== Honeypot check complete ===")
print(f"Candidates flagged : {len(flagged)} / 100")

if flagged:
    print()
    for f in sorted(flagged, key=lambda x: x["rank"]):
        print(f"Rank #{f['rank']:>3}  {f['candidate_id']}  "
              f"semantic={f['semantic_norm']:.3f}  "
              f"hard={f['hard_filter_score']:.3f}  "
              f"final={f['final_score']:.4f}")
        for issue in f["honeypot_signals"]:
            print(f"           ⚠  {issue}")
        print()

    # ── export flagged candidates to jsonl ───────────────────
    output_path = "artifacts/flagged_honeypots.jsonl"
    with open(output_path, "w") as out:
        for f in sorted(flagged, key=lambda x: x["rank"]):
            out.write(json.dumps(f) + "\n")

    print(f"Exported to {output_path}")
else:
    print("No honeypot signals detected in top 100.")