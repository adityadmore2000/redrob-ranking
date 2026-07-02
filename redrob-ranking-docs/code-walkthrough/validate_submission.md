# `validate_submission.py` — Deep Line-by-Line Teaching Notes

> **A standalone CSV validator that enforces the challenge's submission rules.** It does
> *not* touch the pipeline — it just reads a finished `submission.csv` and returns a list
> of every rule violation (or confirms it's valid). This is a classic **"validate against
> a strict spec"** utility: exhaustively check format, uniqueness, ranges, and ordering.

**The design pattern to highlight:** it **collects all errors into a list** and reports them
together, rather than stopping at the first problem. Much better UX than fail-on-first.

---

## Part 1 — Shebang, docstring, constants

```python
#!/usr/bin/env python3
"""Validate submission CSV per challenge rules (sections 2–3). ..."""
import csv, re, sys
from pathlib import Path

REQUIRED_HEADER = ["candidate_id", "rank", "score", "reasoning"]
CANDIDATE_ID_PATTERN = re.compile(r"^CAND_[0-9]{7}$")
DATA_ROW_START = 2
EXPECTED_DATA_ROWS = 100
```

- **`#!/usr/bin/env python3`** — a **shebang**: lets the file be run directly (`./validate_submission.py`)
  by finding `python3` on `PATH`. `env` makes it portable across systems.
- **`re.compile(r"^CAND_[0-9]{7}$")`** — a **precompiled** regex for the id format: `^`/`$` anchor
  the whole string, `CAND_` literal, `[0-9]{7}` exactly 7 digits. Compiling once (module level) is
  slightly faster when matched repeatedly and reads as a named constant.
- Constants encode the spec: exact header, data starts at row 2, exactly 100 data rows.
- **Interview:** what a shebang does; `re.compile` + anchors `^...$` for full-string validation.

---

## Part 2 — Filename + file-open checks

```python
def validate_submission(csv_path):
    errors = []
    path = Path(csv_path)
    if path.suffix.lower() != ".csv":
        errors.append("Filename must use a .csv extension.")
    elif not path.stem:
        errors.append("Filename must be your registered participant ID ...")
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                errors.append("Row 1 must be the header row; file is empty.")
                return errors
            if header != REQUIRED_HEADER:
                errors.append("Row 1 (header) must be exactly:\n ...")
            data_rows = [row for row in reader if any(cell.strip() for cell in row)]
    except UnicodeDecodeError:
        errors.append("File must be UTF-8 encoded."); return errors
    except OSError as e:
        errors.append(f"Cannot read file: {e}"); return errors
```

- **`pathlib.Path`** — the modern path API. `path.suffix` = extension (`.csv`), `path.stem` =
  filename without extension. Cleaner than string slicing.
- **`csv.reader`** with `newline=""` (the required csv idiom) and `encoding="utf-8"`.
- **`next(reader)`** grabs the header row; **`except StopIteration`** handles a completely empty
  file (an exhausted iterator raises `StopIteration`).
- `data_rows` comprehension **skips blank rows** (`any(cell.strip() for cell in row)` — keep rows
  with at least one non-empty cell).
- Two failure exits: `UnicodeDecodeError` (not UTF-8) and `OSError` (unreadable) return early.
- **Interview:** `pathlib` suffix/stem; why `next()` can raise `StopIteration`; catching encoding vs
  IO errors separately.

---

## Part 3 — Per-row validation (accumulate all errors)

```python
    n = len(data_rows)
    if n != EXPECTED_DATA_ROWS:
        errors.append(f"... exactly {EXPECTED_DATA_ROWS} data rows ...; found {n}.")
    seen_ids, seen_ranks, by_rank = set(), set(), []
    for i, cells in enumerate(data_rows):
        row_num = DATA_ROW_START + i
        if len(cells) != len(REQUIRED_HEADER):
            errors.append(f"Row {row_num}: expected 4 columns ..."); continue
        row = dict(zip(REQUIRED_HEADER, cells))
        cid = row["candidate_id"].strip(); rank_s = row["rank"].strip(); score_s = row["score"].strip()

        if not cid: errors.append(...)
        elif not CANDIDATE_ID_PATTERN.match(cid): errors.append("... must be CAND_XXXXXXX ...")
        elif cid in seen_ids: errors.append(f"... duplicate candidate_id '{cid}'.")
        else: seen_ids.add(cid)

        try:
            rank = int(rank_s)
            if str(rank) != rank_s: raise ValueError       # rejects "01", " 1", "1.0"
            if not 1 <= rank <= 100: errors.append("... between 1 and 100.")
            elif rank in seen_ranks: errors.append(f"... duplicate rank {rank}.")
            else: seen_ranks.add(rank)
        except ValueError:
            errors.append("... rank must be an integer (1–100)."); rank = None

        try:
            score = float(score_s)
        except ValueError:
            errors.append("... score must be a float."); score = None

        if rank is not None and score is not None and cid:
            by_rank.append((rank, score, cid))
```

- **`dict(zip(REQUIRED_HEADER, cells))`** — pairs column names with cell values into a dict
  (`zip` + `dict`), so fields are accessed by name.
- **Uniqueness via sets** — `seen_ids`/`seen_ranks` detect duplicates as they go.
- **The strict integer check** `if str(rank) != rank_s: raise ValueError` — subtle and clever:
  `int("01")` succeeds but `str(1) != "01"`, so it **rejects leading zeros, whitespace, and
  `"1.0"`** — enforcing a *canonical* integer string, not just parseability.
- Errors are **accumulated** and, where a field is unusable, set to `None` but processing continues
  (via the `try/except` per field) — the row still contributes what it can to `by_rank`.
- **Interview gold:** the `str(int(x)) == x` canonical-form trick; `dict(zip(...))`; error
  accumulation vs fail-fast.

---

## Part 4 — Cross-row rules (completeness + ordering + tie-break)

```python
    missing = set(range(1, 101)) - seen_ranks
    if missing:
        errors.append(f"Each rank 1–100 must appear exactly once; missing: {sorted(missing)}")

    by_rank.sort(key=lambda x: x[0])
    for i in range(len(by_rank) - 1):
        r1, s1, _ = by_rank[i]; r2, s2, _ = by_rank[i + 1]
        if s1 < s2:
            errors.append(f"score must be non-increasing by rank: rank {r1} ({s1}) < rank {r2} ({s2}).")
    for i in range(len(by_rank) - 1):
        r1, s1, c1 = by_rank[i]; r2, s2, c2 = by_rank[i + 1]
        if s1 == s2 and c1 > c2:
            errors.append(f"Equal scores at ranks {r1} and {r2}: tie-break requires candidate_id ascending ...")
    return errors
```

- **Completeness via set difference:** `set(range(1, 101)) - seen_ranks` = which ranks 1–100 are
  missing. Elegant one-liner for "are all required values present."
- **Monotonic-score check:** sorted by rank, score must be **non-increasing** (higher rank = higher
  or equal score). Flags any `s1 < s2`.
- **Tie-break rule:** when two adjacent ranks have **equal scores**, `candidate_id` must be
  **ascending** — so ties are broken deterministically by id (`c1 > c2` is a violation). This
  matches how the official ranking must break ties.
- **Interview:** set difference for completeness; verifying a monotonic ordering; deterministic
  tie-breaking.

---

## Part 5 — `main` (CLI + exit codes)

```python
def main():
    if len(sys.argv) != 2:
        print("Usage: python validate_submission.py <participant_id>.csv"); sys.exit(1)
    errors = validate_submission(sys.argv[1])
    if errors:
        print(f"Validation failed ({len(errors)} issue(s)):\n")
        for e in errors: print(f"- {e}")
        sys.exit(1)
    print("Submission is valid.")

if __name__ == "__main__":
    main()
```

- Minimal argument handling via **`sys.argv`** (no argparse — a single positional arg suffices).
- **`sys.exit(1)`** on any error, **implicit exit 0** on success — the **Unix convention** that lets
  this be used in scripts/CI (`if python validate_submission.py x.csv; then ...`). Prints all issues
  as a bulleted list.
- **Interview:** `sys.argv` vs `argparse`; exit codes (0 = success, non-zero = failure) and why they
  matter for automation.

---

## Big-picture takeaways

1. **Validate against a strict spec** — header, exact row count, id pattern, rank range/uniqueness,
   score type, monotonic ordering, tie-break — every rule checked.
2. **Accumulate-then-report** — collect all violations into a list and show them together (better UX
   than stopping at the first).
3. **Canonical-form integer trick** — `str(int(x)) == x` rejects `"01"`/`"1.0"`/whitespace, not just
   unparseable input.
4. **Set operations for completeness/uniqueness** — `seen` sets and `set(range(...)) - seen`.
5. **Proper CLI hygiene** — usage message, meaningful exit codes for automation.
