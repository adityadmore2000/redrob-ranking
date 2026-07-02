# Redrob Ranking — Code Teaching Docs

Deep, line-by-line teaching notes for every source file in the project — written to help
explain the codebase confidently in an interview. Each doc follows the same structure:
*What is it / Why is it here / Arguments / Internals / Return value / Alternatives /
Interview questions / Hidden details / Relationships*.

## How the system fits together

```
Raw candidates (JSONL)                       job_description.docx
        │                                            │
        │                                     jd_parser.py → JD dict
        ▼                                            │
   field_map.py  ── accessors + constants ──────────┤
        │                                            │
   ┌────┴─────────── Track 1 (structured) ───────────┴──────┐
   │  track1_hard_filter · track1_availability · track1_credibility
   └────────────────────────────┬───────────────────────────┘
                                 │      Track 2 (semantic)
                    track2_text_builder → track2_embedding (BGE + cosine)
                                 │
                 Phase 1 artifacts (*.npy, track1_details.pkl)
                                 │
                          phase2/rank.py
          final = semantic_norm × hard_filter × availability × credibility
              → stable lexsort → top-100 → reasoning → submission.csv

   Interactive path:  app.py (Streamlit) → demo_pipeline.py
         (reuses the SAME scoring/normalize/rank/reasoning functions, in memory)
```

## Files

### Application / pipeline
- [app.md](app.md) — Streamlit UI (upload, run, results table, CSV download).
- [demo_pipeline.md](demo_pipeline.md) — in-memory end-to-end ranking for the demo.
- [rank.md](rank.md) — **Phase 2**: final multiplicative score, `np.lexsort` ranking, and the
  honest non-hallucinating reasoning generator. The most important file.

### Foundations
- [field_map.md](field_map.md) — accessor layer + all tuning constants (single source of truth).
- [jd_parser.md](jd_parser.md) — rule-based (regex, no NLP) parser turning the JD docx → `JD` dict.

### Track 1 — structured signals
- [track1_hard_filter.md](track1_hard_filter.md) — location/yoe/work-mode/consulting/tenure
  (multiplicative). Also defines the standard Track 1 module shape.
- [track1_availability.md](track1_availability.md) — open-to-work/recency/response/notice/…
  (weighted average).
- [track1_credibility.md](track1_credibility.md) — completeness/endorsements/education/GitHub/…
  (weighted average; log-scaling, sparse-feature nudge).

### Track 2 — semantic match
- [track2_text_builder.md](track2_text_builder.md) — candidate/JD → embedding-ready text
  (recency-weighted truncation, BGE instruction prefixes).
- [track2_embedding.md](track2_embedding.md) — **Phase 1 master runner**: computes all scores +
  BGE embeddings + cosine similarity, writes the parallel artifacts.

### Utilities & infra
- [honeypot_check.md](honeypot_check.md) — audits the top-100 for fabricated/contradictory data.
- [validate_submission.md](validate_submission.md) — strict CSV spec validator.
- [eda_scripts.md](eda_scripts.md) — the EDA that produced the constants and design decisions.
- [init_files.md](init_files.md) — `phase1/__init__.py` and `phase2/__init__.py` (package markers).
- [Dockerfile.md](Dockerfile.md) — HF Spaces container (layer caching, non-root, port 7860).

## Five themes that recur across the codebase

1. **Multiplicative final score** — `semantic_norm × hard_filter × availability × credibility`; any
   weak signal sinks a candidate. (Hard filter multiplies too; availability/credibility average.)
2. **Reuse over reimplementation** — the demo imports the *exact* Phase 2 `normalize_semantic`,
   `rank_top_n`, `build_reasoning`, so interactive output matches the real submission.
3. **Single source of truth** — `field_map` accessors/constants and `MAX_CANDIDATES`/`SCORE_COLS`
   defined once and shared.
4. **Data-driven design** — every threshold/range traces back to the EDA scripts; features with low
   variance (interview_completion_rate) or high sparsity (skill assessments) handled deliberately.
5. **Honest, LLM-free reasoning** — grounded only in facts present in a profile; concerns surfaced;
   confidence toned to rank; consecutive rows de-duplicated.
