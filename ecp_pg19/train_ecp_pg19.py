"""Train ECP-24 on PG-19 with the authors' training procedure (ECPTransformer_main.py), on 1..N GPUs.

Run from the repository root, e.g. on one node with 4 GPUs:
    torchrun --standalone --nproc_per_node=4 -m ecp_pg19.train_ecp_pg19 \
        --data_dir /work/.../LLM_DATASETS/pg19/tokens --out_dir output/ecp24_pg19

Faithful to the original:
  * model: the authors' ECPAttentionTransformer + AutoRegressiveWrapper (latent_len = sequence_length);
    paper ECP-24 sizes are the defaults (24 layers, dim 768, 6 heads, seq 2048, 8 segments, dropout 0.05);
  * optimizer: the authors' configure_optimizers (AdamW, betas (0.9, 0.95), wd 0.1 on Linear weights), lr 2e-4,
    adjusted every 100k iterations by the original code verbatim, including its per-group quirk
    (see ecp_model.original_lr_adjustment);
  * gradient: the original sums two batch-mean losses (batch 8, two backward calls), clips at 1.0 and steps.
    Here the global batch of 16 sequences is split over processes and accumulation steps, and every micro-batch
    loss is multiplied by original_grad_accum / accum. After DDP averaging the gradient equals the original
    one (2 x the gradient of the mean loss over the 16 sequences) for any process count;
  * evaluation: the authors' metric on the TEST split every validate_every iterations (including iteration 0),
    best-by-test checkpoint kept (best.pt). The validation split is evaluated too and logged.

Changes for speed / robustness (not in the original):
  * --precision bf16 (default): torch.autocast bf16 with the minimally modified model copy in
    ecp_pg19/models_bf16 (the original model cannot run under bf16 autocast). tf32: fp32 with TF32 matmuls
    (also uses the copy). fp32: the original model, TF32 off (exactly the original numerics);
  * DDP; resumable last.pt (model, optimizer, iteration, best ppl, per-process RNG states) every save_every
    iterations; the data order depends only on (seed, iteration), so resuming continues the same stream;
  * --num_iterations defaults to ceil(tokens_per_param * num_params / tokens_per_step) (20 tokens/param).
"""
import argparse
import csv
import json
import math
import os
import time
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from ecp_pg19.ecp_base import authors_configure_optimizers
from ecp_pg19.ecp_model import (
    PAPER_ECP24,
    PAPER_ECP24_NUM_PARAMS,
    build_ecp,
    count_parameters,
    eval_loss_sums,
    original_lr_adjustment,
    ppl_from_sums,
)
from ecp_pg19.pg19_data import EvalSet, TokenChunks, TrainOrder

# Keys that must match between a checkpoint and the current run when resuming (num_iterations may change).
RESUME_KEYS = (
    "layers", "dim", "heads", "sequence_length", "num_segments", "net_dropout", "vocab_size", "lr",
    "lr_halving_every", "global_batch", "original_grad_accum", "seed", "precision",
    "train_sha256",
)
TRAIN_COLS = ["iteration", "lr_decay_group", "lr_no_decay_group", "train_loss", "grad_norm", "tokens_per_sec", "elapsed_sec"]
EVAL_COLS = ["iteration", "test_ppl", "validation_ppl", "best_test_ppl", "eval_sec"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="ECP on PG-19", formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", required=True, help="directory with {train,validation,test}.uint16.bin")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--precision", choices=["bf16", "tf32", "fp32"], default="bf16")
    for k, v in PAPER_ECP24.items():
        p.add_argument(f"--{k}", type=type(v), default=v)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lr_halving_every", type=int, default=100000, help="100000 in the original; change only for tests")
    p.add_argument("--global_batch", type=int, default=16, help="sequences per optimizer step (original 8 x 2)")
    p.add_argument("--original_grad_accum", type=int, default=2, help="number of summed batch-mean losses in the original")
    p.add_argument("--micro_batch", type=int, default=4, help="sequences per forward pass per process")
    p.add_argument("--num_iterations", type=int, default=None)
    p.add_argument("--tokens_per_param", type=float, default=20.0)
    p.add_argument("--validate_every", type=int, default=5000)
    p.add_argument("--save_every", type=int, default=2000)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--eval_batch", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_eval_windows", type=int, default=None, help="testing only: cap the number of eval windows")
    p.add_argument("--stop_after", type=int, default=None, help="testing only: stop after this many iterations in this run")
    p.add_argument("--device", default=None, help="default: cuda if available, else cpu")
    return p.parse_args(argv)


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


def read_sha(data_dir, split):
    with open(os.path.join(data_dir, f"{split}.manifest.json")) as f:
        return json.load(f)["sha256"]


def truncate_csv(path, cols, start_it):
    """Drop rows logged at iterations >= start_it (they were logged after the checkpoint we resume from)."""
    if not os.path.exists(path):
        return
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if int(r["iteration"]) < start_it]
    with open(path + ".tmp", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    os.replace(path + ".tmp", path)


def append_csv(path, cols, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if new:
            w.writeheader()
        w.writerow(row)


def atomic_save(obj, path):
    torch.save(obj, path + ".tmp")
    os.replace(path + ".tmp", path)


def rng_state(device):
    s = {"torch_cpu": torch.get_rng_state()}
    if device.type == "cuda":
        s["torch_cuda"] = torch.cuda.get_rng_state(device)
    return s


def set_rng_state(s, device):
    torch.set_rng_state(s["torch_cpu"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(s["torch_cuda"], device)


def save_last(out_dir, model_unwrapped, optim, next_it, best_test_ppl, running_lr, cfg, rank, world, device):
    atomic_save({"iteration": next_it, "world": world, "rng": rng_state(device)}, os.path.join(out_dir, f"rng_rank{rank}.pt"))
    barrier(world)
    if rank == 0:
        atomic_save(
            {
                "model": model_unwrapped.state_dict(),
                "optimizer": optim.state_dict(),
                "iteration": next_it,
                "best_test_ppl": best_test_ppl,
                "running_lr": running_lr,
                "config": cfg,
            },
            os.path.join(out_dir, "last.pt"),
        )
    barrier(world)


def main(argv=None):
    a = parse_args(argv)
    rank, world, device = setup_distributed(a.device)
    is_main = rank == 0

    def log(msg):
        if is_main:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

    # ---- precision ----
    if a.precision == "fp32":
        variant, amp = "original", nullcontext
        tf32 = False
    elif a.precision == "tf32":
        variant, amp = "bf16_copy", nullcontext
        tf32 = True
    else:
        variant, tf32 = "bf16_copy", True

        def amp():
            return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32

    # ---- batch layout ----
    per_step = world * a.micro_batch
    if a.global_batch % per_step != 0:
        raise ValueError(f"global_batch {a.global_batch} not divisible by world {world} x micro_batch {a.micro_batch}")
    accum = a.global_batch // per_step
    loss_scale = a.original_grad_accum / accum

    # ---- data ----
    train = TokenChunks(os.path.join(a.data_dir, "train.uint16.bin"), a.sequence_length)
    test = EvalSet(os.path.join(a.data_dir, "test.uint16.bin"), a.sequence_length)
    val = EvalSet(os.path.join(a.data_dir, "validation.uint16.bin"), a.sequence_length)
    order = TrainOrder(train.num_chunks, a.seed)

    # ---- model + optimizer ----
    torch.manual_seed(a.seed)  # identical initialization on every process
    model = build_ecp(a.layers, a.dim, a.heads, a.sequence_length, a.num_segments, a.net_dropout, a.vocab_size, variant)
    n_params = count_parameters(model)
    model.to(device)
    optim = authors_configure_optimizers(model, a.lr)
    if len(optim.param_groups) != 2:
        raise RuntimeError("expected the two param groups of configure_optimizers")

    tokens_per_step = a.global_batch * a.sequence_length
    num_iterations = a.num_iterations or math.ceil(a.tokens_per_param * n_params / tokens_per_step)

    cfg = {k: getattr(a, k) for k in RESUME_KEYS if hasattr(a, k)}
    cfg["train_sha256"] = read_sha(a.data_dir, "train")
    cfg.update(num_iterations=num_iterations, num_params=n_params, variant=variant, world=world,
               micro_batch=a.micro_batch, accum=accum, loss_scale=loss_scale)

    os.makedirs(a.out_dir, exist_ok=True)
    last_path = os.path.join(a.out_dir, "last.pt")
    start_it, best_test_ppl = 0, float("inf")
    running_lr = a.lr  # plays the role of args.lr in the original lr adjustment
    if os.path.exists(last_path):
        ck = torch.load(last_path, map_location="cpu", weights_only=True)
        diff = {k: (ck["config"].get(k), cfg[k]) for k in RESUME_KEYS if ck["config"].get(k) != cfg[k]}
        if diff:
            raise RuntimeError(f"{last_path} was written with a different configuration: {diff}")
        model.load_state_dict(ck["model"])
        optim.load_state_dict(ck["optimizer"])  # restores the current per-group learning rates as well
        start_it, best_test_ppl = int(ck["iteration"]), float(ck["best_test_ppl"])
        running_lr = float(ck["running_lr"])
        del ck
    rng_path = os.path.join(a.out_dir, f"rng_rank{rank}.pt")
    rng_ok = False
    if start_it > 0 and os.path.exists(rng_path):
        r = torch.load(rng_path, map_location="cpu", weights_only=True)
        if r["iteration"] == start_it and r["world"] == world:
            set_rng_state(r["rng"], device)
            rng_ok = True
    if not rng_ok:
        # fresh run, or resume with a different process count: deterministic per-process dropout seed
        torch.manual_seed(a.seed + 1 + rank + 1_000_003 * start_it)
        if start_it > 0:
            log("WARNING: per-process RNG state not restored (different world size or missing file); dropout reseeded")

    if a.layers == PAPER_ECP24["layers"] and a.dim == PAPER_ECP24["dim"] and a.vocab_size == PAPER_ECP24["vocab_size"] \
            and a.heads == PAPER_ECP24["heads"] and n_params != PAPER_ECP24_NUM_PARAMS:
        raise RuntimeError(f"ECP-24 has {n_params} params, expected {PAPER_ECP24_NUM_PARAMS}")

    ddp_model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None,
                    find_unused_parameters=True) if world > 1 else model
    # find_unused_parameters: ECPAttention.norm is computed but its output is unused, so it never gets a gradient.

    train_tokens = len(train.tokens)
    log(f"world={world} device={device} precision={a.precision} variant={variant} params={n_params:,}")
    log(f"global_batch={a.global_batch} = {world} procs x {a.micro_batch} micro x {accum} accum; loss_scale={loss_scale}")
    log(f"train chunks={train.num_chunks:,} ({train_tokens:,} tokens); test chunks={test.data.shape[0]:,}; "
        f"val chunks={val.data.shape[0]:,}")
    log(f"iterations {start_it} -> {num_iterations} ({num_iterations * tokens_per_step:,} predicted tokens, "
        f"{num_iterations * a.global_batch / train.num_chunks:.3f} epochs)")
    if is_main:
        with open(os.path.join(a.out_dir, "config.json"), "w") as f:
            json.dump(cfg, f, indent=2)
        truncate_csv(os.path.join(a.out_dir, "train_log.csv"), TRAIN_COLS, start_it)
        truncate_csv(os.path.join(a.out_dir, "eval_log.csv"), EVAL_COLS, start_it)

    def evaluate(it):
        t0 = time.time()
        res = {}
        for name, ds in (("test", test), ("validation", val)):
            s, c, _ = eval_loss_sums(model, ds.data, a.sequence_length, device, a.eval_batch, rank, world, amp,
                                     a.max_eval_windows)
            s, c = all_reduce_sum([s, c], device, world)
            res[name] = ppl_from_sums(s, c)
        return res, time.time() - t0

    loss_sum = torch.zeros((), dtype=torch.float64, device=device)
    loss_n = 0
    t_log, it_log = time.time(), start_it
    t_start = time.time()
    run_its = 0
    for it in range(start_it, num_iterations):
        lrs = [g["lr"] for g in optim.param_groups]  # learning rates used by this iteration (for the log)
        model.train()
        for k in range(accum):
            x = train.get(order.micro_batch(it, k, rank, world, a.micro_batch, a.global_batch)).to(device, non_blocking=True)
            sync = ddp_model.no_sync() if (world > 1 and k < accum - 1) else nullcontext()
            with sync:
                with amp():
                    loss = ddp_model(x)
                (loss * loss_scale).backward()
            loss_sum += loss.detach().double()
            loss_n += 1
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        optim.zero_grad(set_to_none=True)
        running_lr = original_lr_adjustment(optim, running_lr, it, a.lr_halving_every)  # same place as the original
        run_its += 1

        if (it + 1) % a.log_every == 0 or it == start_it:
            ls, ln = all_reduce_sum([float(loss_sum), float(loss_n)], device, world)
            gn = float(grad_norm)
            dt = time.time() - t_log
            tps = (it + 1 - it_log) * tokens_per_step / max(dt, 1e-9)
            if is_main:
                append_csv(os.path.join(a.out_dir, "train_log.csv"), TRAIN_COLS, {
                    "iteration": it, "lr_decay_group": lrs[0], "lr_no_decay_group": lrs[1],
                    "train_loss": ls / ln, "grad_norm": gn, "tokens_per_sec": round(tps, 1),
                    "elapsed_sec": round(time.time() - t_start, 1)})
            log(f"it {it} loss {ls / ln:.4f} grad_norm {gn:.3f} lr {lrs[0]:.3e}/{lrs[1]:.3e} tok/s {tps:,.0f}")
            loss_sum.zero_()
            loss_n = 0
            t_log, it_log = time.time(), it + 1

        last_it = it == num_iterations - 1
        if it % a.validate_every == 0 or last_it:
            res, dt = evaluate(it)
            improved = res["test"] < best_test_ppl
            if improved:
                best_test_ppl = res["test"]
                if is_main:
                    atomic_save({"model": model.state_dict(), "iteration": it, "test_ppl": res["test"],
                                 "validation_ppl": res["validation"], "config": cfg},
                                os.path.join(a.out_dir, "best.pt"))
            if is_main:
                append_csv(os.path.join(a.out_dir, "eval_log.csv"), EVAL_COLS, {
                    "iteration": it, "test_ppl": res["test"], "validation_ppl": res["validation"],
                    "best_test_ppl": best_test_ppl, "eval_sec": round(dt, 1)})
            log(f"eval it {it}: test ppl {res['test']:.4f} val ppl {res['validation']:.4f} "
                f"best test {best_test_ppl:.4f}{' (saved best.pt)' if improved else ''} [{dt:.0f}s]")
            barrier(world)

        stop = a.stop_after is not None and run_its >= a.stop_after
        if (it + 1) % a.save_every == 0 or last_it or stop:
            save_last(a.out_dir, model, optim, it + 1, best_test_ppl, running_lr, cfg, rank, world, device)
            log(f"saved last.pt (next iteration {it + 1})")
        if stop:
            log(f"stopping after {run_its} iterations (--stop_after)")
            break

    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
