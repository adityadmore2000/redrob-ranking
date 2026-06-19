import json
import numpy as np
from collections import defaultdict, Counter

# ── load candidates ──
candidates = []
with open('/kaggle/input/datasets/moreadityad/candidate-dataset/candidates.jsonl', 'r') as f:
    for line in f:
        candidates.append(json.loads(line.strip()))

print(f"Loaded {len(candidates):,} candidates\n")

# ── helper functions ──
def word_count(text):
    if not text:
        return 0
    return len(str(text).split())

def approx_tokens(text):
    if not text:
        return 0
    return max(1, int(len(str(text)) / 4))

# ── 1. career_history field presence ──
has_history = sum(1 for c in candidates if c.get("career_history"))
print(f"[1] career_history presence: {has_history:,} / {len(candidates):,} ({100*has_history/len(candidates):.1f}%)\n")

# ── 2. number of jobs per candidate ──
job_counts = []
for c in candidates:
    history = c.get("career_history", [])
    if isinstance(history, list):
        job_counts.append(len(history))

jc = np.array(job_counts)
print(f"[2] Jobs per candidate:")
print(f"    min={jc.min()}, p25={np.percentile(jc,25):.0f}, median={np.median(jc):.0f}, p75={np.percentile(jc,75):.0f}, max={jc.max()}, mean={jc.mean():.1f}")
dist = Counter(job_counts)
print(f"    Distribution:")
for k in sorted(dist.keys()):
    bar = "█" * min(40, int(40 * dist[k] / len(job_counts)))
    print(f"      {k:2d} jobs: {dist[k]:5,} candidates  {bar}")
print()

# ── 3. what fields exist inside career_history entries ──
field_counts = defaultdict(int)
total_jobs = 0
for c in candidates:
    history = c.get("career_history", [])
    if not isinstance(history, list):
        continue
    for job in history:
        total_jobs += 1
        if isinstance(job, dict):
            for k in job.keys():
                field_counts[k] += 1

print(f"[3] Fields inside career_history entries (total jobs: {total_jobs:,}):")
for field, count in sorted(field_counts.items(), key=lambda x: -x[1]):
    print(f"    {field:<30} {count:>7,}  ({100*count/total_jobs:.1f}%)")
print()

# ── 4. description length by job position (0 = most recent) ──
desc_words_by_pos  = defaultdict(list)
desc_tokens_by_pos = defaultdict(list)
desc_presence_by_pos = defaultdict(int)

for c in candidates:
    history = c.get("career_history", [])
    if not isinstance(history, list):
        continue
    try:
        history_sorted = sorted(history, key=lambda x: x.get("start_date", "") or "", reverse=True)
    except Exception:
        history_sorted = history

    for pos, job in enumerate(history_sorted):
        desc = ""
        for field in ["description", "responsibilities", "summary"]:
            desc = job.get(field, "") or ""
            if desc:
                break
        desc = str(desc).strip()
        desc_words_by_pos[pos].append(word_count(desc))
        desc_tokens_by_pos[pos].append(approx_tokens(desc))
        if desc:
            desc_presence_by_pos[pos] += 1

print(f"[4] Description word count by job position (0 = most recent):")
print(f"    {'Pos':<5} {'Count':>7} {'Present%':>9} {'min':>5} {'p25':>5} {'median':>7} {'p75':>5} {'p90':>5} {'max':>6}")
print(f"    {'-'*65}")
for pos in sorted(desc_words_by_pos.keys())[:8]:
    arr = np.array(desc_words_by_pos[pos])
    n = len(arr)
    pct = 100 * desc_presence_by_pos[pos] / n if n else 0
    print(f"    {pos:<5} {n:>7,} {pct:>8.1f}% {arr.min():>5.0f} {np.percentile(arr,25):>5.0f} "
          f"{np.median(arr):>7.0f} {np.percentile(arr,75):>5.0f} "
          f"{np.percentile(arr,90):>5.0f} {arr.max():>6.0f}")
print()

print(f"[5] Approx token count by job position (0 = most recent):")
print(f"    {'Pos':<5} {'p25':>5} {'median':>7} {'p75':>5} {'p90':>5} {'max':>6}")
print(f"    {'-'*40}")
for pos in sorted(desc_tokens_by_pos.keys())[:8]:
    arr = np.array(desc_tokens_by_pos[pos])
    print(f"    {pos:<5} {np.percentile(arr,25):>5.0f} {np.median(arr):>7.0f} "
          f"{np.percentile(arr,75):>5.0f} {np.percentile(arr,90):>5.0f} {arr.max():>6.0f}")
print()

# ── 5. token budget simulation ──
summary_tokens = []
skills_tokens  = []
edu_tokens     = []
title_tokens   = []

for c in candidates:
    summary_tokens.append(approx_tokens(c.get("summary", "") or ""))
    skills = c.get("skills", [])
    if isinstance(skills, list):
        skills_text = ", ".join(
            s.get("name", str(s)) if isinstance(s, dict) else str(s)
            for s in skills
        )
    else:
        skills_text = str(skills or "")
    skills_tokens.append(approx_tokens(skills_text))
    edu = c.get("education", [])
    edu_text = " ".join([
        f"{e.get('degree','')} {e.get('field','')} {e.get('institution','')}"
        for e in (edu if isinstance(edu, list) else [])
    ])
    edu_tokens.append(approx_tokens(edu_text))
    title_tokens.append(approx_tokens(
        f"{c.get('current_title','')} at {c.get('current_company','')} ({c.get('current_industry','')})"
    ))

def p50(arr): return int(np.percentile(arr, 50))

print(f"[6] Token budget simulation (BGE-base-en limit = 512 tokens):")
print(f"    Field budget estimates (median approx tokens):")
print(f"      Current role block : {p50(title_tokens):>4} tokens")
print(f"      Summary            : {p50(summary_tokens):>4} tokens")
print(f"      Skills             : {p50(skills_tokens):>4} tokens")
print(f"      Education          : {p50(edu_tokens):>4} tokens")

fixed = p50(title_tokens) + p50(summary_tokens) + p50(skills_tokens) + p50(edu_tokens)
career_budget = 512 - fixed
print(f"\n    Fixed fields total (median) : {fixed} tokens")
print(f"    Remaining for career history: {career_budget} tokens")

weights = [0.40, 0.30, 0.20, 0.10]
labels  = ["full detail", "medium detail", "brief", "one-liner"]
print(f"\n    Suggested token budget per job position:")
for i, (w, label) in enumerate(zip(weights, labels)):
    print(f"      Job {i} ({label:<14}): ~{int(career_budget * w):>3} tokens")
print()

# ── 6. sample career histories ──
print(f"[7] Sample career history entries (3 candidates with 3+ jobs):")
shown = 0
for c in candidates:
    history = c.get("career_history", [])
    if isinstance(history, list) and len(history) >= 3:
        print(f"\n    Candidate: {c.get('candidate_id', 'unknown')}")
        try:
            history_sorted = sorted(history, key=lambda x: x.get("start_date", "") or "", reverse=True)
        except Exception:
            history_sorted = history
        for pos, job in enumerate(history_sorted[:4]):
            desc = ""
            for field in ["description", "responsibilities", "summary"]:
                desc = job.get(field, "") or ""
                if desc:
                    break
            desc_preview = str(desc)[:150].replace("\n", " ")
            print(f"      Job {pos}: {job.get('title','?')} @ {job.get('company','?')} "
                  f"({job.get('start_date','?')} → {job.get('end_date','present')})")
            print(f"              words={word_count(desc)}, tokens≈{approx_tokens(desc)}")
            print(f"              desc: {desc_preview}{'...' if len(str(desc)) > 150 else ''}")
        shown += 1
        if shown >= 3:
            break

print(f"\n{'='*60}")
print("EDA complete.")
print(f"{'='*60}\n")
