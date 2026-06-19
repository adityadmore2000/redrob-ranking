# Redrob Data & AI Challenge — Candidate Ranking System

A candidate ranking system for the Redrob Data & AI Challenge. Given a job
description and a pool of candidate profiles, the system scores and ranks
candidates by how well they fit the role.

## Architecture

The system uses a **two-phase architecture**:

- **Phase 1 — Offline pre-computation.** Heavy, reusable work that does not
  depend on a specific query is computed once and cached to `artifacts/`. This
  is split into two tracks:
  - **Track 1 — Structured signals.** Rule-based scores derived from
    structured candidate fields: a hard filter, availability, and credibility.
  - **Track 2 — Semantic representation.** Builds a text representation of each
    candidate and encodes it into embeddings for semantic matching.
- **Phase 2 — Ranking.** Combines the pre-computed Track 1 signals and Track 2
  embeddings against a given query to produce a final ranked list of
  candidates.

All field access goes through `field_map.py`, the single source of truth for
candidate field paths. Scoring scripts never read candidate fields directly.

## Setup

```bash
pip install -r requirements.txt
```

Place input data under `data/`. Pre-computed outputs are written to `artifacts/`.

## How to run

### Phase 1 — offline pre-computation

Run the Track 1 (structured) and Track 2 (semantic) steps to populate
`artifacts/`:

```bash
# Track 1 — structured signals
python phase1/track1_hard_filter.py
python phase1/track1_availability.py
python phase1/track1_credibility.py

# Track 2 — semantic representation
python phase1/track2_text_builder.py
python phase1/track2_embedding.py
```

### Phase 2 — ranking

```bash
python phase2/rank.py
```

## Dependencies

- numpy
- pandas
- sentence-transformers
- torch
- scikit-learn
- pyyaml
- tqdm

See `requirements.txt`.
