"""Train ECP-24 on WikiText-103 canary corpora (T0 or null), 200,000 iterations."""
import argparse
import csv
import json
import math
import os
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from ecp_pg19.ecp_base import authors_configure_optimizers
from ecp_pg19.ecp_model import (
    PAPER_ECP24,
    build_ecp,
    count_parameters,
    eval_loss_sums,
    original_lr_adjustment,
    ppl_from_sums,
)
from ecp_pg19.pg19_data import EvalSet, TrainOrder
from .common import path
from .prepare_eval import run as run_prepare_eval

TRAIN_COLS = ["iteration", "lr_decay_group", "lr_no_decay_group", "train_loss", "grad_norm", "tokens_per_sec", "elapsed_sec"]
EVAL_COLS = ["iteration", "test_ppl", "validation_ppl", "best_test_ppl", "eval_sec"]


class StreamChunks:
    def __init__(self, npy_path, seq_len):
        if not os.path.exists(npy_path):
            raise FileNotFoundError(npy_path)
        self.tokens = np.load(npy_path, mmap_mode="r")
        self.chunk_len = seq_len + 1
        self.num_chunks = len(self.tokens) // self.chunk_len
        if self.num_chunks == 0:
            raise ValueError(f"{npy_path}: fewer than {self.chunk_len} tokens")

    def get(self, chunk_indices) -> torch.Tensor:
        c = self.chunk_len
        rows = [np.asarray(self.tokens[i * c:(i + 1) * c], dtype=np.int64) for i in chunk_indices]
        return torch.from_numpy(np.stack(rows))


def setup_distributed(device_arg):
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    use_cuda = torch.cuda.is_available() if device_arg is None else device_arg.startswith("cuda")
    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    if world > 1:
        dist.init_process_group(backend="nccl" if use_cuda else "gloo")
    return rank, world, device


def barrier(world):
    if world > 1:
        dist.barrier()


def all_reduce_sum(values, device, world):
    t = torch.tensor(values, dtype=torch.float64, device=device)
    if world > 1:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.tolist()


def append_csv(file_path, cols, row):
    new = not os.path.exists(file_path)
    with open(file_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if new:
            w.writeheader()
        w.writerow(row)


def atomic_save(obj, file_path):
    torch.save(obj, file_path + ".tmp")
    os.replace(file_path + ".tmp", file_path)


def run(model_name, num_iterations=None):
    rank, world, device = setup_distributed(None)
    is_main = rank == 0
    out_dir = path("models", model_name)
    os.makedirs(out_dir, exist_ok=True)

    def log(msg):
        if is_main:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [{model_name}] {msg}", flush=True)

    # Prepare eval datasets (validation and test splits)
    if is_main:
        run_prepare_eval()
    barrier(world)

    seq_len = PAPER_ECP24["sequence_length"]
    eval_dir = path("eval_data")
    test_ds = EvalSet(os.path.join(eval_dir, "test.uint16.bin"), seq_len)
    val_ds = EvalSet(os.path.join(eval_dir, "validation.uint16.bin"), seq_len)

    # Training stream
    stream_file = path("corpora", f"{model_name}_stream.npy")
    train = StreamChunks(stream_file, seq_len)
    order = TrainOrder(train.num_chunks, seed=0)

    # Precision
    variant = "bf16_copy"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    def amp():
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)

    # Batch layout: global_batch 16
    global_batch = 16
    micro_batch = 4
    per_step = world * micro_batch
    accum = global_batch // per_step if global_batch >= per_step else 1
    loss_scale = 2.0 / accum

    # Model & optimizer
    torch.manual_seed(42)
    model = build_ecp(
        PAPER_ECP24["layers"],
        PAPER_ECP24["dim"],
        PAPER_ECP24["heads"],
        PAPER_ECP24["sequence_length"],
        PAPER_ECP24["num_segments"],
        PAPER_ECP24["net_dropout"],
        PAPER_ECP24["vocab_size"],
        variant=variant,
    )
    n_params = count_parameters(model)
    model.to(device)
    optim = authors_configure_optimizers(model, lr=2e-4)

    total_iters = num_iterations or int(os.environ.get("ECP_ITERATIONS", "200000"))
    tokens_per_step = global_batch * seq_len

    last_path = os.path.join(out_dir, "last.pt")
    start_it, best_test_ppl = 0, float("inf")
    running_lr = 2e-4
    if os.path.exists(last_path):
        ck = torch.load(last_path, map_location="cpu", weights_only=True)
        model.load_state_dict(ck["model"])
        optim.load_state_dict(ck["optimizer"])
        start_it = int(ck["iteration"])
        best_test_ppl = float(ck.get("best_test_ppl", float("inf")))
        running_lr = float(ck.get("running_lr", 2e-4))
        log(f"Resumed from {last_path} at iteration {start_it}")
        del ck

    ddp_model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None,
                    find_unused_parameters=True) if world > 1 else model

    log(f"Params: {n_params:,} | Iterations: {start_it} -> {total_iters} (~{total_iters * tokens_per_step / len(train.tokens):.1f} epochs)")

    def evaluate(it):
        t0 = time.time()
        res = {}
        for name, ds in (("test", test_ds), ("validation", val_ds)):
            s, c, _ = eval_loss_sums(model, ds.data, seq_len, device, batch_size=8, rank=rank, world=world, amp=amp)
            s, c = all_reduce_sum([s, c], device, world)
            res[name] = ppl_from_sums(s, c)
        return res, time.time() - t0

    loss_sum = torch.zeros((), dtype=torch.float64, device=device)
    loss_n = 0
    t_log, it_log = time.time(), start_it
    t_start = time.time()

    for it in range(start_it, total_iters):
        lrs = [g["lr"] for g in optim.param_groups]
        model.train()
        for k in range(accum):
            x = train.get(order.micro_batch(it, k, rank, world, micro_batch, global_batch)).to(device, non_blocking=True)
            sync = ddp_model.no_sync() if (world > 1 and k < accum - 1) else nullcontext()
            with sync:
                with amp():
                    loss = ddp_model(x)
                (loss * loss_scale).backward()
            loss_sum += loss.detach().double()
            loss_n += 1

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        optim.zero_grad(set_to_none=True)
        running_lr = original_lr_adjustment(optim, running_lr, it, halving_every=100000)

        # Log metrics every 100 steps
        if (it + 1) % 100 == 0 or it == start_it:
            ls, ln = all_reduce_sum([float(loss_sum), float(loss_n)], device, world)
            dt = time.time() - t_log
            tps = (it + 1 - it_log) * tokens_per_step / max(dt, 1e-9)
            if is_main:
                append_csv(os.path.join(out_dir, "train_log.csv"), TRAIN_COLS, {
                    "iteration": it, "lr_decay_group": lrs[0], "lr_no_decay_group": lrs[1],
                    "train_loss": ls / ln, "grad_norm": 0.0, "tokens_per_sec": round(tps, 1),
                    "elapsed_sec": round(time.time() - t_start, 1)})
            log(f"it {it} loss {ls / ln:.4f} tok/s {tps:,.0f}")
            loss_sum.zero_()
            loss_n = 0
            t_log, it_log = time.time(), it + 1

        last_it = (it == total_iters - 1)
        # Evaluate every 5,000 steps (including step 0 and final step)
        if it % 5000 == 0 or last_it:
            res, dt = evaluate(it)
            improved = res["test"] < best_test_ppl
            if improved or not os.path.exists(os.path.join(out_dir, "best.pt")):
                best_test_ppl = min(best_test_ppl, res["test"])
                if is_main:
                    atomic_save({"model": model.state_dict(), "iteration": it, "test_ppl": res["test"],
                                 "validation_ppl": res["validation"]}, os.path.join(out_dir, "best.pt"))
            if is_main:
                append_csv(os.path.join(out_dir, "eval_log.csv"), EVAL_COLS, {
                    "iteration": it, "test_ppl": res["test"], "validation_ppl": res["validation"],
                    "best_test_ppl": best_test_ppl, "eval_sec": round(dt, 1)})
            log(f"eval it {it}: test ppl {res['test']:.2f} val ppl {res['validation']:.2f} best {best_test_ppl:.2f}")
            barrier(world)

        # Checkpoint every 5,000 steps
        if (it + 1) % 5000 == 0 or last_it:
            if is_main:
                atomic_save({
                    "model": model.state_dict(), "optimizer": optim.state_dict(),
                    "iteration": it + 1, "best_test_ppl": best_test_ppl, "running_lr": running_lr
                }, os.path.join(out_dir, "last.pt"))
            barrier(world)

    if world > 1:
        dist.destroy_process_group()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("models", nargs="+", choices=["T0", "null"])
    p.add_argument("--iterations", type=int, default=None)
    args = p.parse_args()
    for m in args.models:
        run(m, num_iterations=args.iterations)


if __name__ == "__main__":
    main()
