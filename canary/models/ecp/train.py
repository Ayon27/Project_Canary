"""ECP training loop, following ECPTransformer_main.py, with DDP, bf16/TF32 and resumable checkpoints."""
import csv
import math
import os
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from ...utils import write_json
from .evaluate import DEVICE, eval_ppl, eval_tokens, reduce_sum
from .net import build

RESUME_KEYS = ("layers", "dim", "heads", "sequence_length", "num_segments", "net_dropout", "vocab_size", "lr",
               "lr_halving_every", "global_batch", "original_grad_accum", "seed", "precision", "train_sha256")
TRAIN_COLS = ["iteration", "lr_decay_group", "lr_no_decay_group", "train_loss", "grad_norm", "tokens_per_sec",
              "elapsed_sec"]
EVAL_COLS = ["iteration", "test_ppl", "validation_ppl", "best_validation_ppl", "eval_sec"]


def configure_optimizers(model, lr):
    """Authors' AdamW: weight decay 0.1 on Linear weights, 0 on biases, LayerNorm and Embedding."""
    decay, no_decay = set(), set()
    for mn, m in model.named_modules():
        for pn, _ in m.named_parameters():
            fpn = f"{mn}.{pn}" if mn else pn
            if pn.endswith("bias") or (pn.endswith("weight") and isinstance(m, (nn.LayerNorm, nn.Embedding))):
                no_decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, nn.Linear):
                decay.add(fpn)
    params = dict(model.named_parameters())
    assert not decay & no_decay and params.keys() <= decay | no_decay
    return torch.optim.AdamW([{"params": [params[n] for n in sorted(decay)], "weight_decay": 0.1},
                              {"params": [params[n] for n in sorted(no_decay)], "weight_decay": 0.0}],
                             lr=lr, betas=(0.9, 0.95))


def lr_adjustment(optim, running_lr, it, every):
    """Authors' step schedule, unchanged: running_lr is halved once per param group."""
    if (it + 1) % every == 0:
        for g in optim.param_groups:
            g["lr"] = running_lr / 2
            running_lr = running_lr / 2
    return running_lr


class TrainOrder:
    """Chunk order that depends only on (seed, step); epoch e is a permutation from default_rng([seed, e])."""

    def __init__(self, n, seed):
        self.n, self.seed, self.epoch, self.perm = n, seed, None, None

    def chunk(self, p):
        if p // self.n != self.epoch:
            self.epoch = p // self.n
            self.perm = np.random.default_rng([self.seed, self.epoch]).permutation(self.n)
        return self.perm[p % self.n]

    def micro_batch(self, step, k, rank, world, micro, global_batch):
        start = step * global_batch + (k * world + rank) * micro
        return [self.chunk(start + j) for j in range(micro)]


def chunks(tokens, seq_len, idx):
    c = seq_len + 1
    return torch.from_numpy(np.stack([np.asarray(tokens[i * c:(i + 1) * c], dtype=np.int64) for i in idx]))


def append_csv(path, cols, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if new:
            w.writeheader()
        w.writerow(row)


def truncate_csv(path, cols, start_it):
    """Drop rows logged after the checkpoint we resume from."""
    if os.path.exists(path):
        with open(path, newline="") as f:
            rows = [r for r in csv.DictReader(f) if int(r["iteration"]) < start_it]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)


def atomic_save(obj, path):
    torch.save(obj, path + ".tmp")
    os.replace(path + ".tmp", path)


def train(c, tokens, eval_dir, out_dir, sha=None, prepare=None):
    """Train on a 1-D token array; evaluate on eval_dir/{test,validation}.uint16.bin. prepare() runs on rank 0 first."""
    world, rank, local = (int(os.environ.get(k, d)) for k, d in (("WORLD_SIZE", 1), ("RANK", 0), ("LOCAL_RANK", 0)))
    device = torch.device("cuda", local) if DEVICE == "cuda" else torch.device("cpu")
    if DEVICE == "cuda":
        torch.cuda.set_device(local)
    if world > 1:
        dist.init_process_group("nccl" if DEVICE == "cuda" else "gloo")
    barrier = dist.barrier if world > 1 else lambda: None
    is_main, exact, seq = rank == 0, c["exact_resume"], c["sequence_length"]
    log = lambda msg: is_main and print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)
    os.makedirs(out_dir, exist_ok=True)
    if prepare and is_main:
        prepare()
    barrier()

    assert c["precision"] in ("bf16", "tf32")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    amp = (lambda: torch.autocast(device.type, dtype=torch.bfloat16)) if c["precision"] == "bf16" else nullcontext
    assert c["global_batch"] % (world * c["micro_batch"]) == 0, "world x micro_batch must divide global_batch"
    accum = c["global_batch"] // (world * c["micro_batch"])
    loss_scale = c["original_grad_accum"] / accum  # gradient equals the authors' sum of batch-mean losses

    n_chunks = len(tokens) // (seq + 1)
    order = TrainOrder(n_chunks, c["order_seed"])
    test, val = (eval_tokens(os.path.join(eval_dir, f"{s}.uint16.bin"), seq) for s in ("test", "validation"))
    torch.manual_seed(c["train_seed"])  # same initialization on every process
    model = build(c)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model.to(device)
    optim = configure_optimizers(model, c["lr"])
    tokens_per_step = c["global_batch"] * seq
    num_iterations = c["num_iterations"] or math.ceil(c["tokens_per_param"] * n_params / tokens_per_step)
    cfg = {**{k: c.get(k) for k in RESUME_KEYS}, "seed": c["train_seed"], "train_sha256": sha,
           "num_iterations": num_iterations, "num_params": n_params, "world": world, "micro_batch": c["micro_batch"],
           "accum": accum, "loss_scale": loss_scale}

    last_path, best_path = os.path.join(out_dir, "last.pt"), os.path.join(out_dir, "best.pt")
    rng_path = os.path.join(out_dir, f"rng_rank{rank}.pt")
    start, best, running_lr = 0, float("inf"), c["lr"]
    if os.path.exists(last_path):
        ck = torch.load(last_path, map_location="cpu", weights_only=True)
        diff = {k: (ck["config"].get(k), cfg[k]) for k in RESUME_KEYS if ck["config"].get(k) != cfg[k]} if exact else {}
        assert not diff, f"{last_path} was written with a different configuration: {diff}"
        model.load_state_dict(ck["model"])
        optim.load_state_dict(ck["optimizer"])
        start, best = int(ck["iteration"]), float(ck.get("best_validation_ppl", ck.get("best_test_ppl", best)))
        running_lr = float(ck.get("running_lr", running_lr))
        log(f"resumed from {last_path} at iteration {start}")
        del ck
    if exact:
        r = torch.load(rng_path, map_location="cpu", weights_only=True) if start and os.path.exists(rng_path) else None
        if r and r["iteration"] == start and r["world"] == world:
            torch.set_rng_state(r["rng"]["torch_cpu"])
            if device.type == "cuda":
                torch.cuda.set_rng_state(r["rng"]["torch_cuda"], device)
        else:  # fresh run, or another process count: deterministic per-process dropout seed
            torch.manual_seed(c["train_seed"] + 1 + rank + 1_000_003 * start)
            if start:
                log("WARNING: RNG state not restored; dropout reseeded")

    # ECPAttention.norm never gets a gradient, so DDP needs find_unused_parameters.
    ddp = DDP(model, device_ids=[device.index] if device.type == "cuda" else None,
              find_unused_parameters=True) if world > 1 else model
    log(f"world={world} precision={c['precision']} params={n_params:,} batch={world}x{c['micro_batch']}x{accum} "
        f"chunks={n_chunks:,} iterations {start} -> {num_iterations} "
        f"({num_iterations * c['global_batch'] / n_chunks:.2f} epochs)")
    if is_main:
        write_json(os.path.join(out_dir, "config.json"), cfg)
    if is_main and exact:
        for name, cols in (("train_log.csv", TRAIN_COLS), ("eval_log.csv", EVAL_COLS)):
            truncate_csv(os.path.join(out_dir, name), cols, start)

    loss_sum, loss_n = torch.zeros((), dtype=torch.float64, device=device), 0
    t_start = t_log = time.time()
    it_log = start
    for it in range(start, num_iterations):
        lrs = [g["lr"] for g in optim.param_groups]
        model.train()
        for k in range(accum):
            x = chunks(tokens, seq, order.micro_batch(it, k, rank, world, c["micro_batch"], c["global_batch"]))
            with ddp.no_sync() if world > 1 and k < accum - 1 else nullcontext():
                with amp():
                    loss = ddp(x.to(device, non_blocking=True))
                (loss * loss_scale).backward()
            loss_sum += loss.detach().double()
            loss_n += 1
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        optim.step()
        optim.zero_grad(set_to_none=True)
        running_lr = lr_adjustment(optim, running_lr, it, c["lr_halving_every"])

        if (it + 1) % c["log_every"] == 0 or it == start:
            ls, ln = reduce_sum([float(loss_sum), loss_n], device, world)
            tps = (it + 1 - it_log) * tokens_per_step / max(time.time() - t_log, 1e-9)
            if is_main:
                append_csv(os.path.join(out_dir, "train_log.csv"), TRAIN_COLS, dict(
                    iteration=it, lr_decay_group=lrs[0], lr_no_decay_group=lrs[1], train_loss=ls / ln,
                    grad_norm=grad_norm, tokens_per_sec=round(tps, 1), elapsed_sec=round(time.time() - t_start, 1)))
            log(f"it {it} loss {ls / ln:.4f} grad_norm {grad_norm:.3f} lr {lrs[0]:.3e}/{lrs[1]:.3e} tok/s {tps:,.0f}")
            loss_sum.zero_()
            loss_n, t_log, it_log = 0, time.time(), it + 1

        last_it = it == num_iterations - 1
        if it % c["validate_every"] == 0 or last_it:
            t0 = time.time()
            res = {s: eval_ppl(model, d, seq, device, c["eval_batch"], rank, world, amp)
                   for s, d in (("test", test), ("validation", val))}
            improved = res["validation"] < best
            if improved:
                best = res["validation"]
                if is_main:
                    atomic_save({"model": model.state_dict(), "iteration": it, "test_ppl": res["test"],
                                 "validation_ppl": res["validation"], **({"config": cfg} if exact else {})}, best_path)
            if is_main:
                append_csv(os.path.join(out_dir, "eval_log.csv"), EVAL_COLS, dict(
                    iteration=it, test_ppl=res["test"], validation_ppl=res["validation"], best_validation_ppl=best,
                    eval_sec=round(time.time() - t0, 1)))
            log(f"eval it {it}: test ppl {res['test']:.4f} val ppl {res['validation']:.4f} best val {best:.4f}")
            barrier()

        if (it + 1) % c["save_every"] == 0 or last_it:
            if exact:
                rng = {"torch_cpu": torch.get_rng_state(),
                       **({"torch_cuda": torch.cuda.get_rng_state(device)} if device.type == "cuda" else {})}
                atomic_save({"iteration": it + 1, "world": world, "rng": rng}, rng_path)
                barrier()
            if is_main:
                atomic_save({"model": model.state_dict(), "optimizer": optim.state_dict(), "iteration": it + 1,
                             "best_validation_ppl": best, "running_lr": running_lr, **({"config": cfg} if exact else {})},
                            last_path)
            barrier()
    if world > 1:
        dist.destroy_process_group()
