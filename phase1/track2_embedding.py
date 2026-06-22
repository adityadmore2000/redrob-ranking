"""
track2_embedding.py
===================
Phase 1 master runner.

This script produces every artifact Phase 2 needs to rank candidates. It:
  1. Loads all candidates from the input JSONL.
  2. Computes Track 1 scores (hard filter, availability, credibility).
  3. Builds candidate text representations (track2_text_builder).
  4. Embeds all candidate texts + the JD text with BGE-base-en-v1.5.
  5. Computes cosine similarity between each candidate and the JD vector.
  6. Saves all artifacts to the artifacts/ directory.

Artifacts produced (all in artifacts/)
--------------------------------------
candidate_ids.npy       np.array of candidate_id strings, shape (N,)
semantic_scores.npy     cosine similarity of candidate vs JD, shape (N,)
hard_filter_scores.npy  hard_filter_score per candidate, shape (N,)
availability_scores.npy availability_score per candidate, shape (N,)
credibility_scores.npy  credibility_score per candidate, shape (N,)
track1_details.pkl      list of per-candidate dicts with every Track 1
                        sub-signal merged — used for the Phase 2 reasoning column
jd_embedding.npy        JD vector, shape (embedding_dim,)

All parallel arrays (candidate_ids, semantic_scores, the three Track 1 score
arrays, and track1_details) share the same candidate order.

Running on Kaggle (CPU)
-----------------------
    python phase1/track2_embedding.py \
        --input /kaggle/input/datasets/moreadityad/candidate-dataset/candidates.jsonl \
        --artifacts artifacts --batch-size 64

Verifying after a run
---------------------
    python phase1/track2_embedding.py --verify --artifacts artifacts

Public API
----------
embed_texts(texts, model, batch_size)        -> np.ndarray (N, dim)
cosine_similarity_matrix(cand_emb, jd_emb)   -> np.ndarray (N,)
run(input_path, artifacts_dir, batch_size)   -> None
verify_artifacts(artifacts_dir)              -> None
"""

import json
import os
import pickle
import sys
import time

# Make both the project root (field_map, jd_parser) and this phase1/ directory
# (sibling track1_*/track2_text_builder modules) importable regardless of how
# the script is launched: `python phase1/track2_embedding.py` puts only phase1/
# on sys.path, while `python -m phase1.track2_embedding` puts only the root.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
for _p in (_PROJECT_ROOT, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from field_map import get_candidate_id
from track2_text_builder import build_candidate_text, build_jd_text
import track1_hard_filter
import track1_availability
import track1_credibility

MODEL_NAME = 'BAAI/bge-base-en-v1.5'


def load_candidates(input_path: str) -> list:
    """
    Load all candidates from a JSONL file (one JSON object per line). Also
    accepts a single JSON array (the data/sample_candidates.json fixture) so
    sanity checks can run against the sample without a JSONL conversion.
    """
    with open(input_path) as f:
        head = f.read(1)
        f.seek(0)
        if head == '[':
            return json.load(f)
        candidates = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            candidates.append(json.loads(line))
    return candidates


def embed_texts(texts: list, model, batch_size: int = 64) -> np.ndarray:
    """
    Embed a list of texts using the provided sentence-transformers model.
    Process in batches to avoid OOM.
    Show tqdm progress bar.
    Returns numpy array of shape (len(texts), embedding_dim).
    """
    from tqdm import tqdm  # imported lazily so Track-1-only runs need no tqdm

    embeddings = []
    for start in tqdm(range(0, len(texts), batch_size), desc='Embedding', unit='batch'):
        batch = texts[start:start + batch_size]
        vecs = model.encode(
            batch,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        embeddings.append(np.asarray(vecs, dtype=np.float32))
    if not embeddings:
        return np.zeros((0, 0), dtype=np.float32)
    return np.vstack(embeddings)


def cosine_similarity_matrix(candidate_embeddings: np.ndarray, jd_embedding: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between each candidate embedding and the JD embedding.
    jd_embedding shape: (embedding_dim,)
    Returns 1D numpy array of shape (n_candidates,) with scores in [-1, 1].
    Normalize both sides before dot product.
    """
    cand = np.asarray(candidate_embeddings, dtype=np.float32)
    jd = np.asarray(jd_embedding, dtype=np.float32).ravel()

    cand_norms = np.linalg.norm(cand, axis=1)
    cand_norms[cand_norms == 0] = 1.0
    cand_unit = cand / cand_norms[:, None]

    jd_norm = np.linalg.norm(jd)
    if jd_norm == 0:
        jd_norm = 1.0
    jd_unit = jd / jd_norm

    return cand_unit @ jd_unit


def _merge_track1_details(hard, avail, cred):
    """
    Merge the three parallel per-candidate Track 1 dicts into one dict per
    candidate, keeping every sub-signal. Order is preserved.
    """
    merged = []
    for h, a, cr in zip(hard, avail, cred):
        row = {'candidate_id': h['candidate_id']}
        row.update(h)
        row.update(a)
        row.update(cr)
        merged.append(row)
    return merged


def run(input_path: str, artifacts_dir: str = 'artifacts', batch_size: int = 256,
        device: str = None, skip_embeddings: bool = False) -> None:
    """
    Full Phase 1 pipeline:
    1. Load all candidates from input_path
    2. Compute Track 1 scores for all candidates
    3. Build candidate text representations
    4. Load BGE model and embed everything
    5. Compute semantic scores
    6. Save artifacts

    device: 'cpu' or 'cuda'. If None, auto-detects (cuda if available else cpu).

    skip_embeddings: when True, only recompute and re-save the Track 1
    artifacts — candidate_ids.npy, hard_filter_scores.npy,
    availability_scores.npy, credibility_scores.npy, and the merged
    track1_details.pkl. Steps [3]-[5] (text building, model load, embedding)
    are skipped entirely, and semantic_scores.npy / jd_embedding.npy are left
    untouched. Use this to repair a stale or hard-filter-only
    track1_details.pkl without paying for a full GPU embedding pass.
    """
    os.makedirs(artifacts_dir, exist_ok=True)
    total_start = time.time()

    # [1/6] Load candidates
    t = time.time()
    print('[1/6] Loading candidates...', end='', flush=True)
    candidates = load_candidates(input_path)
    n = len(candidates)
    print(f'        done in {time.time() - t:.1f}s — {n} candidates loaded')

    # [2/6] Track 1 scores
    t = time.time()
    print('[2/6] Computing Track 1 scores...', end='', flush=True)
    hard = track1_hard_filter.compute_all(candidates)
    avail = track1_availability.compute_all(candidates)
    cred = track1_credibility.compute_all(candidates)
    track1_details = _merge_track1_details(hard, avail, cred)
    print(f'  done in {time.time() - t:.1f}s')

    candidate_ids = np.array([get_candidate_id(c) for c in candidates], dtype=object)
    hard_scores = np.array([r['hard_filter_score'] for r in hard], dtype=np.float32)
    avail_scores = np.array([r['availability_score'] for r in avail], dtype=np.float32)
    cred_scores = np.array([r['credibility_score'] for r in cred], dtype=np.float32)

    if skip_embeddings:
        # Track-1-only path: re-save everything except the embedding artifacts.
        # semantic_scores.npy and jd_embedding.npy are deliberately left as-is.
        t = time.time()
        print('[skip-embeddings] Saving Track 1 artifacts only '
              '(semantic_scores.npy / jd_embedding.npy untouched)...',
              end='', flush=True)
        np.save(os.path.join(artifacts_dir, 'candidate_ids.npy'), candidate_ids)
        np.save(os.path.join(artifacts_dir, 'hard_filter_scores.npy'), hard_scores)
        np.save(os.path.join(artifacts_dir, 'availability_scores.npy'), avail_scores)
        np.save(os.path.join(artifacts_dir, 'credibility_scores.npy'), cred_scores)
        with open(os.path.join(artifacts_dir, 'track1_details.pkl'), 'wb') as f:
            pickle.dump(track1_details, f)
        print(f'  done in {time.time() - t:.1f}s')
        print(f'Total time: {time.time() - total_start:.1f}s')
        return

    # [3/6] Candidate texts
    t = time.time()
    print('[3/6] Building candidate texts...', end='', flush=True)
    texts = [build_candidate_text(c) for c in candidates]
    print(f'   done in {time.time() - t:.1f}s')

    # [4/6] Load model
    t = time.time()
    import torch
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[4/6] Loading embedding model on {device}...', end='', flush=True)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME, device=device)
    print(f'    done in {time.time() - t:.1f}s')

    # [5/6] Embed candidates (batched) + JD (single vector, separately)
    t = time.time()
    print('[5/6] Embedding candidates...')
    candidate_embeddings = embed_texts(texts, model, batch_size=batch_size)

    from jd_parser import JD
    jd_text = build_jd_text(JD)
    jd_embedding = embed_texts([jd_text], model, batch_size=batch_size)[0]

    semantic_scores = cosine_similarity_matrix(candidate_embeddings, jd_embedding)
    print(f'[5/6] Embedding candidates...      done in {time.time() - t:.1f}s '
          f'— {len(texts)} texts embedded')

    # [6/6] Save artifacts (parallel arrays share candidate order)
    t = time.time()
    print('[6/6] Saving artifacts...', end='', flush=True)
    np.save(os.path.join(artifacts_dir, 'candidate_ids.npy'), candidate_ids)
    np.save(os.path.join(artifacts_dir, 'semantic_scores.npy'),
            semantic_scores.astype(np.float32))
    np.save(os.path.join(artifacts_dir, 'hard_filter_scores.npy'), hard_scores)
    np.save(os.path.join(artifacts_dir, 'availability_scores.npy'), avail_scores)
    np.save(os.path.join(artifacts_dir, 'credibility_scores.npy'), cred_scores)
    np.save(os.path.join(artifacts_dir, 'jd_embedding.npy'),
            jd_embedding.astype(np.float32))
    with open(os.path.join(artifacts_dir, 'track1_details.pkl'), 'wb') as f:
        pickle.dump(track1_details, f)
    print(f'          done in {time.time() - t:.1f}s')

    print(f'Total time: {time.time() - total_start:.1f}s')


def verify_artifacts(artifacts_dir: str = 'artifacts') -> None:
    """
    Load and print a summary of saved artifacts.
    Prints shapes, score distributions, and 3 sample candidates.
    Used to verify artifacts are correct after run completes.
    Call with --verify flag.
    """
    candidate_ids = np.load(os.path.join(artifacts_dir, 'candidate_ids.npy'),
                            allow_pickle=True)
    semantic = np.load(os.path.join(artifacts_dir, 'semantic_scores.npy'),
                       allow_pickle=False)
    hard = np.load(os.path.join(artifacts_dir, 'hard_filter_scores.npy'),
                   allow_pickle=False)
    avail = np.load(os.path.join(artifacts_dir, 'availability_scores.npy'),
                    allow_pickle=False)
    cred = np.load(os.path.join(artifacts_dir, 'credibility_scores.npy'),
                   allow_pickle=False)
    jd_embedding = np.load(os.path.join(artifacts_dir, 'jd_embedding.npy'),
                           allow_pickle=False)
    with open(os.path.join(artifacts_dir, 'track1_details.pkl'), 'rb') as f:
        track1_details = pickle.load(f)

    print('Artifact shapes:')
    print(f'  candidate_ids       : {candidate_ids.shape}')
    print(f'  semantic_scores     : {semantic.shape}')
    print(f'  hard_filter_scores  : {hard.shape}')
    print(f'  availability_scores : {avail.shape}')
    print(f'  credibility_scores  : {cred.shape}')
    print(f'  jd_embedding        : {jd_embedding.shape}')
    print(f'  track1_details      : {len(track1_details)} dicts')

    # Sanity: all parallel arrays must have the same length.
    lengths = {len(candidate_ids), len(semantic), len(hard), len(avail),
               len(cred), len(track1_details)}
    print(f'  parallel lengths consistent: {len(lengths) == 1}')

    def dist(name, arr):
        q = np.percentile(arr, [0, 25, 50, 75, 100])
        print(f'  {name:20} min={q[0]:.3f} p25={q[1]:.3f} '
              f'median={q[2]:.3f} p75={q[3]:.3f} max={q[4]:.3f}')

    print('\nScore distributions:')
    dist('semantic_scores', semantic)
    dist('hard_filter_scores', hard)
    dist('availability_scores', avail)
    dist('credibility_scores', cred)

    print('\nSample candidates:')
    for i in range(min(3, len(candidate_ids))):
        print('-' * 60)
        print(f'  candidate_id        : {candidate_ids[i]}')
        print(f'  semantic_score      : {semantic[i]:.4f}')
        print(f'  hard_filter_score   : {hard[i]:.4f}')
        print(f'  availability_score  : {avail[i]:.4f}')
        print(f'  credibility_score   : {cred[i]:.4f}')
        print(f'  track1_details id   : {track1_details[i]["candidate_id"]}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input',
        default='/kaggle/input/datasets/moreadityad/candidate-dataset/candidates.jsonl',
        help='Path to candidates.jsonl'
    )
    parser.add_argument(
        '--artifacts',
        default='artifacts',
        help='Directory to save artifacts'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=256,  # was 64 — larger batches are faster on GPU
        help='Embedding batch size. Use 64 for CPU, 256+ for GPU.'
    )
    parser.add_argument(
        '--device',
        default=None,
        help='Device to use for embedding: cpu or cuda. Defaults to auto-detect '
             '(cuda if available, else cpu).'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify existing artifacts instead of running the pipeline'
    )
    parser.add_argument(
        '--skip-embeddings',
        action='store_true',
        help='Only recompute Track 1 artifacts (candidate_ids, the three '
             'Track 1 score arrays, and the merged track1_details.pkl). Skips '
             'the embedding model entirely and leaves semantic_scores.npy and '
             'jd_embedding.npy untouched.'
    )
    args = parser.parse_args()

    if args.verify:
        verify_artifacts(args.artifacts)
    else:
        run(args.input, args.artifacts, args.batch_size, args.device,
            skip_embeddings=args.skip_embeddings)
