# Monorepo layout

Self-study implementations for [Stanford CS336](https://cs336.stanford.edu/) live under `assignments/`. Each folder is an independent [`uv`](https://docs.astral.sh/uv/) project with its own `pyproject.toml`, tests, and docs.

```
Stanford-CS336/
├── assignments/
│   ├── assignment1-basics/      # BPE, transformer, training
│   ├── assignment2-systems/    # Flash attention, DDP, FSDP
│   ├── assignment3-scaling/    # Scaling laws, isoflops API
│   ├── assignment4-data/       # Data filtering & dedup
│   └── assignment5-alignment/  # GRPO, SFT, DPO
├── scripts/verify_all.sh      # One-command cross-check (A1–A5)
├── PROJECT_PURPOSE.md
└── README.md
```

## Verify everything

```bash
bash scripts/verify_all.sh
```

## Per-assignment verify

| Assignment | Command | What it checks |
|------------|---------|----------------|
| A1 | `cd assignments/assignment1-basics && uv run pytest -q && bash scripts/verify_complete.sh` | 47 tests + training artifacts |
| A2 | `cd assignments/assignment2-systems && uv run pytest -q` | 10 pass, 4 skip (Mac) |
| A3 | `cd assignments/assignment3-scaling && bash scripts/finalize_assignment.sh` | 7 tests + scaling artifacts |
| A4 | `cd assignments/assignment4-data && bash scripts/finalize_assignment.sh` | 21 tests + required modules |
| A5 | `cd assignments/assignment5-alignment && bash scripts/finalize_assignment.sh` | 26 tests + required modules |

## Docs convention

Each assignment includes where applicable:

- `PROJECT_STATUS.md` — test results and scope
- `SOLUTION.md` — implementation summary
- `LEARNINGS.md` — takeaways
- `writeup.md` — narrative / analysis
- `GOALS.md` — workflow (A3–A5)

## Local-only (not in git)

- `*.bin` token memmaps, large checkpoints
- `local-shared-data/` (A4 classifiers, ~2GB)
- `.venv/`, `assignment*-main/` download stubs at repo root
