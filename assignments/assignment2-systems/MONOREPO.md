# Assignment 2 in this monorepo

This folder is the official [assignment2-systems](https://github.com/stanford-cs336/assignment2-systems) handout, vendored into **Stanford-CS336**.

## Use our Assignment 1 code

Assignment 2 ships a staff `cs336-basics/` package. Replace it with our implementation from Assignment 1:

```bash
cd assignments/assignment2-systems
rm -rf cs336-basics
cp -R ../assignment1-basics/cs336_basics cs336-basics/cs336_basics
# Keep cs336-basics/pyproject.toml from staff, or copy staff cs336-basics first then overwrite cs336_basics/
```

Simpler: keep staff `cs336-basics/pyproject.toml` and only replace the `cs336_basics/` Python package inside it.

## Start

```bash
cd assignments/assignment2-systems
uv sync
uv run pytest -q
```

Implement optimized kernels and distributed training in `cs336_systems/`.

Handout: [cs336_assignment2_systems.pdf](./cs336_assignment2_systems.pdf)
