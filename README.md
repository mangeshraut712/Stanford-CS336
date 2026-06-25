# Stanford CS336 — Language Modeling from Scratch

Self-study implementations for [Stanford CS336](https://cs336.stanford.edu/spring2025/) (Spring 2025).

**Author:** [Mangesh Raut](https://github.com/mangeshraut712)

## Progress

| # | Assignment | Folder | Status |
|---|------------|--------|--------|
| 1 | [Basics](https://github.com/stanford-cs336/assignment1-basics) | [`assignment1-basics-main/`](assignment1-basics-main/) | **In progress** — code + tests done; OWT pipeline running |
| 2 | [Systems](https://github.com/stanford-cs336/assignment2-systems) | `assignments/assignment2-systems/` | Not started |
| 3 | [Scaling](https://github.com/stanford-cs336/assignment3-scaling) | `assignments/assignment3-scaling/` | Not started |
| 4 | [Data](https://github.com/stanford-cs336/assignment4-data) | `assignments/assignment4-data/` | Not started |
| 5 | [Alignment](https://github.com/stanford-cs336/assignment5-alignment) | `assignments/assignment5-alignment/` | Not started |

## Assignment 1 (Basics)

- **Solution overview:** [`assignment1-basics-main/SOLUTION.md`](assignment1-basics-main/SOLUTION.md)
- **Writeup:** [`assignment1-basics-main/writeup.md`](assignment1-basics-main/writeup.md)
- **Tests:** `cd assignment1-basics-main && uv run pytest -q` → 47 passed, 1 xfail
- **TinyStories main run:** val loss **2.07** (2500 steps, Mac MPS)

```bash
cd assignment1-basics-main
bash scripts/package_solution.sh
```

Large datasets and checkpoints stay **local only** (see `.gitignore`). Download data per the [assignment README](assignment1-basics-main/README.md).

## Repo layout (target)

```
Stanford-CS336/
├── README.md
├── assignment1-basics-main/     # → will move to assignments/assignment1-basics/
└── assignments/                 # future assignments
```

## Course links

- **Website:** https://cs336.stanford.edu
- **Assignment 1 code:** https://github.com/stanford-cs336/assignment1-basics
- **Lectures:** https://github.com/stanford-cs336/spring2025-lectures

## License

Course materials © Stanford; my solution code is for educational / portfolio use.
