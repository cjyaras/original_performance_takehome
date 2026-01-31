# Claude Code Instructions

## Critical Rules

**DO NOT modify any files in the `tests/` folder.** This includes:
- `tests/submission_tests.py`
- `tests/frozen_problem.py`
- Any other files in the tests directory

The tests folder contains the official benchmark and validation code. Modifying it would invalidate any performance results.

## Goal

Optimize `perf_takehome.py` to minimize cycle count on the simulated VLIW SIMD CPU architecture defined in `problem.py`.

## Validation

Run `python tests/submission_tests.py` to validate performance. Verify tests are unmodified with:
```bash
git diff origin/main tests/
```
This should return empty output.
