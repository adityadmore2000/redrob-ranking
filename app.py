"""
app.py — Streamlit demo for the Redrob candidate ranking system.

Upload a pool of candidate profiles (JSON / JSONL, ≤100 records) or load the
built-in sample, and the app runs the full ranking pipeline end-to-end:

  Track 1 — structured signals (hard filter, availability, credibility)
  Track 2 — semantic match of each candidate against the job description
            (BGE-base-en-v1.5 embeddings)
  Phase 2 — multiplicative final score, ranking, and per-candidate reasoning

Run locally with:  streamlit run app.py
"""

import json
import os

import pandas as pd
import streamlit as st

import demo_pipeline as dp
from demo_pipeline import InputError, MAX_CANDIDATES

_HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE_PATH = os.path.join(_HERE, 'data', 'sample_candidates.json')

# Score columns to surface with a visual bar in the results table.
SCORE_COLS = [
    'final_score', 'semantic_score_norm', 'hard_filter_score',
    'availability_score', 'credibility_score',
]

st.set_page_config(
    page_title='Candidate Ranking Demo',
    page_icon='🏆',
    layout='wide',
)


# ──────────────────────────────────────────────────────────────────────────
# model (cached for the lifetime of the Space — heavy to load)
# ──────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner='Loading the embedding model (first run only)…')
def load_model():
    # CPU is the safe default on Hugging Face Spaces free tier.
    return dp.get_model(device='cpu')


# ──────────────────────────────────────────────────────────────────────────
# sidebar — what the demo does + expected schema
# ──────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header('About')
    st.markdown(
        'Ranks a pool of candidates against a fixed **job description** using '
        'two tracks of signals:\n\n'
        '- **Structured** — location, experience, tenure, availability, '
        'credibility (GitHub, endorsements, education, verification).\n'
        '- **Semantic** — how well each profile matches the JD, via '
        '`BGE-base-en-v1.5` embeddings.\n\n'
        'Final score is the product of the normalized semantic score and the '
        'three structured signals, so any disqualifying signal pulls the '
        'candidate down naturally.'
    )
    st.subheader('Expected input')
    st.markdown(
        f'A **JSON array** (`.json`) or **JSON Lines** (`.jsonl`) file with up '
        f'to **{MAX_CANDIDATES}** candidate objects. Each object looks like:'
    )
    st.code(
        '{\n'
        '  "candidate_id": "CAND_0000001",\n'
        '  "profile": {\n'
        '    "anonymized_name": "...",\n'
        '    "current_title": "...", "current_company": "...",\n'
        '    "location": "...", "country": "India",\n'
        '    "years_of_experience": 6.5, "summary": "..."\n'
        '  },\n'
        '  "career_history": [ { "title": "...", "company": "...",\n'
        '                        "duration_months": 24, "description": "..." } ],\n'
        '  "skills": [ { "name": "Python", "proficiency": "expert" } ],\n'
        '  "education": [ { "tier": "tier_1" } ],\n'
        '  "redrob_signals": { "open_to_work_flag": true,\n'
        '                      "github_activity_score": 42 }\n'
        '}',
        language='json',
    )
    st.caption(
        'Missing fields fall back to neutral defaults, so partial profiles '
        'still rank — they just carry less signal.'
    )


# ──────────────────────────────────────────────────────────────────────────
# header
# ──────────────────────────────────────────────────────────────────────────
st.title('🏆 Candidate Ranking Demo')
st.markdown(
    'Upload a candidate pool or try the sample, and rank everyone against the '
    'role end-to-end.'
)


# ──────────────────────────────────────────────────────────────────────────
# input controls
# ──────────────────────────────────────────────────────────────────────────
col_upload, col_sample = st.columns([3, 1])

with col_upload:
    uploaded = st.file_uploader(
        f'Upload candidates (JSON or JSONL, ≤{MAX_CANDIDATES})',
        type=['json', 'jsonl'],
        help='A JSON array of candidate objects, or one object per line (JSONL).',
    )

with col_sample:
    st.write('')  # vertical spacer to align the button with the uploader
    st.write('')
    use_sample = st.button('🎲 Use Sample Data', use_container_width=True)

# Resolve the active input source for this run. The sample button takes
# precedence on the click that fires it.
raw_bytes = None
source_name = None
if use_sample:
    try:
        with open(SAMPLE_PATH, 'rb') as f:
            raw_bytes = f.read()
        source_name = 'sample_candidates.json'
    except OSError as exc:
        st.error(f'Could not load the bundled sample data: {exc}')
elif uploaded is not None:
    raw_bytes = uploaded.getvalue()
    source_name = uploaded.name


# ──────────────────────────────────────────────────────────────────────────
# run the pipeline
# ──────────────────────────────────────────────────────────────────────────
def run_ranking(raw, name):
    """Validate input, run the pipeline with a progress bar, return a DataFrame."""
    try:
        candidates = dp.load_candidates_from_bytes(raw, name)
    except InputError as exc:
        st.error(f'⚠️ {exc}')
        return None
    except Exception as exc:  # noqa: BLE001 — surface anything unexpected cleanly
        st.error(f'⚠️ Could not read the file: {exc}')
        return None

    st.success(f'Loaded **{len(candidates)}** candidates from `{name}`.')

    model = load_model()

    progress_bar = st.progress(0.0, text='Starting…')

    def on_progress(frac, msg):
        progress_bar.progress(min(1.0, frac), text=msg)

    try:
        with st.spinner('Ranking candidates…'):
            df = dp.rank_candidates(candidates, model, progress=on_progress)
    except Exception as exc:  # noqa: BLE001
        progress_bar.empty()
        st.error(
            'Ranking failed while processing this file. The records may not '
            f'match the expected candidate schema.\n\nDetails: `{exc}`'
        )
        return None

    progress_bar.empty()
    return df


if raw_bytes is not None:
    df = run_ranking(raw_bytes, source_name)
    if df is not None:
        st.session_state['ranked_df'] = df
        st.session_state['ranked_source'] = source_name


# ──────────────────────────────────────────────────────────────────────────
# results
# ──────────────────────────────────────────────────────────────────────────
df = st.session_state.get('ranked_df')
if df is not None:
    st.divider()
    st.subheader(f'Ranked results — {len(df)} candidates')

    top = df.iloc[0]
    m1, m2, m3 = st.columns(3)
    m1.metric('Top candidate', top['name'] or top['candidate_id'])
    m2.metric('Top final score', f"{top['final_score']:.3f}")
    m3.metric('Candidates ranked', len(df))

    # Interactive table: rank #1 at the top, score columns shown as bars so the
    # spread is visible at a glance.
    column_config = {
        'rank': st.column_config.NumberColumn('Rank', width='small'),
        'candidate_id': st.column_config.TextColumn('Candidate ID'),
        'name': st.column_config.TextColumn('Name'),
        'reasoning': st.column_config.TextColumn('Reasoning', width='large'),
    }
    for col in SCORE_COLS:
        label = col.replace('_', ' ').replace('score', '').strip().title()
        column_config[col] = st.column_config.ProgressColumn(
            label or col,
            min_value=0.0,
            max_value=1.0,
            format='%.3f',
        )

    st.dataframe(
        df,
        column_config=column_config,
        hide_index=True,
        use_container_width=True,
        height=560,
    )

    csv_bytes = df[['candidate_id', 'rank', 'final_score', 'reasoning']].rename(columns={'final_score': 'score'}).to_csv(index=False).encode('utf-8')
    st.download_button(
        '⬇️ Download Ranked CSV',
        data=csv_bytes,
        file_name='ranked_candidates.csv',
        mime='text/csv',
        use_container_width=False,
    )

    with st.expander('How to read the scores'):
        st.markdown(
            '- **Final** — `semantic × hard_filter × availability × credibility`. '
            'Drives the ranking.\n'
            '- **Semantic Norm** — JD match, min-max normalized across this pool '
            '(so it is relative to the uploaded set).\n'
            '- **Hard Filter** — location / experience / tenure / consulting fit. '
            'A low value here caps the final score.\n'
            '- **Availability** — open-to-work, responsiveness, notice period.\n'
            '- **Credibility** — profile completeness, GitHub, endorsements, '
            'education, verification.\n\n'
            'The **Reasoning** column is generated per candidate and only cites '
            'facts present in that profile.'
        )
else:
    st.info('Upload a file or click **Use Sample Data** to see the ranking.')
