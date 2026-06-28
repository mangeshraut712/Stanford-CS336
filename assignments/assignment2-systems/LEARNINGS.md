# CS336 Assignment 2 — Key Learnings (Self-Study)

## Flash Attention

- Online softmax (tracking row max and log-sum-exp) avoids materializing the full \(N \times N\) attention matrix.
- Saving log-sum-exp \(L\) in the forward pass is required for a stable backward pass.
- Triton kernels are Linux/CUDA-only in this assignment's `uv` lockfile — Mac development uses the PyTorch reference implementation.

## DDP

- Broadcast **all** parameters from rank 0 at init (not only `requires_grad=True`) so models with frozen params stay in sync.
- Overlapping gradient all-reduces (one handle per parameter) hides communication behind earlier backward compute.
- `finish_gradient_synchronization()` must average gradients (`sum` then `/ world_size`) to match `CrossEntropyLoss(reduction='mean')` on sharded batches.

## FSDP

- Sharding `cs336_basics.Linear` / `Embedding` along the leading dimension matches the staff module layout `(d_out, d_in)` and `(vocab, d_model)`.
- Custom `autograd.Function` wrappers that gather weights under `no_grad` can **drop parameter gradients** unless the shard tensor stays in the graph — hook-based gather + `register_hook` on a detached `requires_grad` leaf is simpler and preserves activation gradients through the stack.
- `gather_full_params()` must be called on **every rank** (it uses `all_gather`).

## Sharded optimizer

- Optimizer state lives only on the shard each rank owns; after `step()`, broadcast refreshed shards so all ranks share the same weights before the next forward.

## Environment (Mac)

- Distributed tests use **gloo** on CPU with `mp.spawn(world_size=2)`.
- Default `MASTER_PORT=12390` in tests — kill stale processes if spawn hangs.
