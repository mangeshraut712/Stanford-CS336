#!/usr/bin/env python3
"""CS336 Assignment 1 training script."""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch

from cs336_basics.checkpoint import load_checkpoint, save_checkpoint
from cs336_basics.data import get_batch
from cs336_basics.nn_utils import clip_gradient, cross_entropy
from cs336_basics.optimizer import AdamW, get_cosine_lr
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.train_model import BasicsTransformerLM

logger = logging.getLogger(__name__)
SPECIAL_TOKEN = "<|endoftext|>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a BasicsTransformerLM")
    parser.add_argument("--train-tokens", type=Path, required=True)
    parser.add_argument("--val-tokens", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", type=str, default="tinystories")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--eval-batches", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--max-lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--d-ff", type=int, default=1344)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="cs336-assignment1")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--no-rope", action="store_true")
    parser.add_argument("--post-norm", action="store_true")
    parser.add_argument("--no-rmsnorm", action="store_true")
    parser.add_argument("--ffn-type", choices=["swiglu", "silu"], default="swiglu")
    parser.add_argument("--generate-prompt", type=str, default="Once upon a time")
    parser.add_argument("--generate-tokens", type=int, default=256)
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_token_array(path: Path) -> np.memmap:
    return np.memmap(path, dtype=np.uint16, mode="r")


@torch.no_grad()
def evaluate(
    model: BasicsTransformerLM,
    dataset: np.memmap,
    batch_size: int,
    context_length: int,
    device: str,
    num_batches: int,
) -> float:
    model.eval()
    losses: list[float] = []
    for _ in range(num_batches):
        x, y = get_batch(dataset, batch_size, context_length, device)
        logits = model(x)
        loss = cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1))
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def maybe_init_wandb(args: argparse.Namespace):
    if not args.wandb:
        return None
    import wandb

    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.run_name,
        config=vars(args),
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "train_args.json", "w") as f:
        json.dump({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, f, indent=2)

    tokenizer = Tokenizer.from_files(
        args.tokenizer_dir / "vocab.json",
        args.tokenizer_dir / "merges.txt",
        special_tokens=[SPECIAL_TOKEN],
    )
    train_data = load_token_array(args.train_tokens)
    val_data = load_token_array(args.val_tokens)

    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        use_rope=not args.no_rope,
        pre_norm=not args.post_norm,
        use_rmsnorm=not args.no_rmsnorm,
        ffn_type=args.ffn_type,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.max_lr, weight_decay=args.weight_decay)
    start_step = 0
    if args.resume is not None:
        start_step = load_checkpoint(args.resume, model, optimizer)

    wandb_run = maybe_init_wandb(args)
    model.train()
    start_time = time.time()
    train_losses: list[float] = []

    for step in range(start_step, args.max_steps):
        lr = get_cosine_lr(step, args.max_lr, args.min_lr, args.warmup_steps, args.max_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = get_batch(train_data, args.batch_size, args.context_length, device)
        logits = model(x)
        loss = cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        clip_gradient(model.parameters(), args.grad_clip)
        optimizer.step()
        train_losses.append(loss.item())

        if (step + 1) % args.log_every == 0:
            avg_train = float(np.mean(train_losses[-args.log_every :]))
            elapsed = time.time() - start_time
            logger.info("step=%d train_loss=%.4f lr=%.2e elapsed=%.1fs", step + 1, avg_train, lr, elapsed)
            if wandb_run is not None:
                import wandb

                wandb.log(
                    {
                        "train/loss": avg_train,
                        "train/lr": lr,
                        "time/elapsed_sec": elapsed,
                    },
                    step=step + 1,
                )

        if (step + 1) % args.eval_every == 0:
            val_loss = evaluate(
                model, val_data, args.batch_size, args.context_length, device, args.eval_batches
            )
            elapsed = time.time() - start_time
            logger.info("step=%d val_loss=%.4f elapsed=%.1fs", step + 1, val_loss, elapsed)
            if wandb_run is not None:
                import wandb

                wandb.log({"val/loss": val_loss, "time/elapsed_sec": elapsed}, step=step + 1)

        if (step + 1) % args.checkpoint_every == 0:
            ckpt_path = args.output_dir / f"checkpoint_step_{step + 1}.pt"
            save_checkpoint(model, optimizer, step + 1, ckpt_path)
            model.save_pretrained(args.output_dir / "latest_model")

    final_ckpt = args.output_dir / "checkpoint_final.pt"
    save_checkpoint(model, optimizer, args.max_steps, final_ckpt)
    model.save_pretrained(args.output_dir / "final_model")

    prompt_ids = torch.tensor(tokenizer.encode(args.generate_prompt), dtype=torch.long, device=device).unsqueeze(0)
    eos_id = tokenizer.bytes_to_id.get(SPECIAL_TOKEN.encode("utf-8"))
    generated = model.generate(
        prompt_ids,
        max_new_tokens=args.generate_tokens,
        temperature=0.8,
        top_k=50,
        eos_token_id=eos_id,
    )
    generated_text = tokenizer.decode(generated[0].tolist())
    sample_path = args.output_dir / "generated_sample.txt"
    sample_path.write_text(generated_text)
    logger.info("Saved generation to %s", sample_path)

    summary = {
        "final_train_loss": float(np.mean(train_losses[-args.log_every :])),
        "final_val_loss": evaluate(
            model, val_data, args.batch_size, args.context_length, device, args.eval_batches
        ),
        "total_elapsed_sec": time.time() - start_time,
        "total_tokens_seen": args.batch_size * args.max_steps * args.context_length,
        "generated_sample_path": str(sample_path),
    }
    with open(args.output_dir / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Run summary: %s", summary)

    if wandb_run is not None:
        import wandb

        wandb.log(summary, step=args.max_steps)
        wandb.finish()


if __name__ == "__main__":
    main()
