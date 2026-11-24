"""ECP perplexity (authors' metric), final fp32 evaluation, and judge settings."""
import math
import os
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist

from ...utils import out, write_json
from .net import build

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def eval_tokens(bin_path, seq_len):
    """Split tokens cut to whole (seq_len + 1)-token chunks, as in the authors' WikitextDataset."""
    t = np.memmap(bin_path, dtype=np.uint16, mode="r")
    return torch.from_numpy(np.asarray(t[:len(t) // (seq_len + 1) * (seq_len + 1)], dtype=np.int64))


def windows(n, seq_len):
    """Starts of the (seq_len + 1)-token eval windows, stride seq_len // 2."""
    return [b for b in range(0, n, seq_len // 2) if b + seq_len + 1 <= n]


def reduce_sum(values, device, world):
    t = torch.tensor(values, dtype=torch.float64, device=device)
    if world > 1:
        dist.all_reduce(t)
    return t.tolist()


@torch.no_grad()
def eval_ppl(model, enc, seq_len, device, batch, rank=0, world=1, amp=nullcontext):
    """exp(mean window loss), batched; each process takes every world-th window."""
    starts = windows(enc.numel(), seq_len)[rank::world]
    was_training, total = model.training, 0.0
    model.eval()
    for i in range(0, len(starts), batch):
        bs = starts[i:i + batch]
        with amp():
            total += float(model(torch.stack([enc[b:b + seq_len + 1] for b in bs]).to(device))) * len(bs)
    model.train(was_training)
    total, count = reduce_sum([total, len(starts)], device, world)
    return math.exp(total / count)


@torch.no_grad()
def evaluate(c):
    """fp32 test and validation perplexity of best.pt, computed as in compute_perplexity_huggingface."""
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    ck = torch.load(out(c, "best.pt"), map_location="cpu", weights_only=True)
    model = build(c)
    model.load_state_dict(ck["model"])
    model.to(DEVICE).eval()
    seq, res = c["sequence_length"], {"iteration": ck["iteration"], "precision": "fp32"}
    for split in ("test", "validation"):
        enc = eval_tokens(os.path.join(c["data_dir"], f"{split}.uint16.bin"), seq)
        nll = [model(enc[b:b + seq + 1][None].to(DEVICE)) for b in windows(enc.numel(), seq)]
        res[f"{split}_ppl"] = float(torch.exp(torch.stack(nll).mean()))
        print(f"{split}: ppl {res[f'{split}_ppl']:.4f}", flush=True)
    write_json(out(c, "best.pt.eval_fp32.json"), res)


def scorer(c, m):
    """Judge settings: rows zero-padded to the block length; logits from the unwrapped transformer."""
    d = out(c, "models", m, "")
    ck = torch.load(d + ("best.pt" if os.path.exists(d + "best.pt") else "last.pt"), map_location="cpu",
                    weights_only=True)
    model = build(c)
    model.load_state_dict(ck["model"])
    model.to(DEVICE).eval()
    return dict(feats=model.model, head=lambda rows: rows, pad=0, width=c["block"], chunk=16, device=DEVICE)
