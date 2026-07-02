# `phase1/__init__.py` and `phase2/__init__.py` — Teaching Notes

> Two tiny files with one big job: **mark their directories as Python packages.**

---

## What an `__init__.py` is

When Python sees a directory containing an `__init__.py`, it treats that directory as a
**package** — an importable namespace. That's what makes `from phase2.rank import build_reasoning`
(used in `demo_pipeline.py`) work: `phase2` is a package because `phase2/__init__.py` exists.

- The file runs **once**, the first time the package is imported, and whatever names it defines
  become attributes of the package (e.g. `phase2.SOMETHING`).
- It can be **completely empty** — its mere presence is the signal.

**Interview:** "What does `__init__.py` do?" → marks a directory as a package so its modules are
importable via dotted paths; it also runs as the package's initialization code. (Since Python 3.3,
*namespace packages* can exist without it, but including one is explicit and avoids surprises —
especially important here given the `sys.path` juggling the phase modules do.)

---

## `phase1/__init__.py`

```python
# phase1 modules import the parsed job description via `from jd_parser import JD`
# (e.g. `JD['yoe_min']`) to drive their scoring logic.
```

- **Only two comment lines, no code.** It exists purely to make `phase1` a package, plus a comment
  documenting a key dependency: the phase1 scorers rely on `jd_parser.JD` for their thresholds.
- Nothing executes here (comments are stripped at tokenization).

---

## `phase2/__init__.py`

- **Completely empty** (0 bytes). Its only purpose is to make `phase2` an importable package so
  `phase2.rank` resolves.

---

## Why both are needed here

`demo_pipeline.py` does `from phase2.rank import ...` — the **dotted** import form requires `phase2`
to be a real package. Meanwhile the phase1 modules are often imported *flat* (`import
track1_hard_filter`) thanks to the `sys.path` hacks, but keeping `phase1/__init__.py` lets them
*also* be imported as `phase1.track1_hard_filter` (e.g. via `python -m phase1.track2_embedding`).
Having the `__init__.py` files supports **both** launch/import styles the project uses.

**Takeaway:** trivial files, but they're the reason the package-style imports across `demo_pipeline`
and the `-m` module invocations resolve cleanly.
