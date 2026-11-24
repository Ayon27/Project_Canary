"""Stage judge: mean loss of every candidate after the work's prefix, its rank among the K, fp32."""
import csv
import os

import torch

from ..models import ecp, qwen
from ..utils import out, read_jsonl
from .canaries import candidate_tokens, rank_pvalue

COLUMNS = ["model", "format", "repetition", "work_id", "cand_id", "is_released", "candidate", "loss", "rank"]


@torch.no_grad()
def losses(prefix, tails, feats, head, pad, width, chunk, device):
    """Mean loss of each tail after prefix. Rows are padded with `pad` to `width` (None: longest row)."""
    p, res = len(prefix), []
    for i in range(0, len(tails), chunk):
        group = tails[i:i + chunk]
        x = torch.full((len(group), width or p + max(map(len, group))), pad)
        for j, t in enumerate(group):
            x[j, :p + len(t)] = torch.tensor(prefix + t)
        h = feats(x.to(device))
        rows = torch.cat([h[j, p - 1:p - 1 + len(t)] for j, t in enumerate(group)])
        target = torch.tensor([y for t in group for y in t], device=device)
        nll = -torch.log_softmax(head(rows).float(), -1).gather(1, target[:, None])[:, 0]
        res += [s.double().mean().item() for s in nll.cpu().split([len(t) for t in group])]
    return res


def run(c, m):
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    s = (qwen if c["model"] == "qwen" else ecp).scorer(c, m)
    file = out(c, "judge", f"{m}.csv")
    with open(file + ".tmp", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        for w in read_jsonl(out(c, "works", "works_manifest.jsonl")):
            prefix, tails = candidate_tokens(c, w["judge_prefix"], w["candidates"])
            assert len(prefix) + max(map(len, tails)) <= c["block"]
            ls = losses(prefix, tails, **s)
            ranks = [sum(o <= x for o in ls) for x in ls]  # ties count against detection
            idx = w["released_idx"]
            assert ranks[idx] / c["K"] == rank_pvalue([-x for x in ls], idx)
            writer.writerows([m, w["format"], w["repetition"], w["work_id"], i, int(i == idx), cand, repr(x), r]
                             for i, (cand, x, r) in enumerate(zip(w["candidates"], ls, ranks)))
    os.replace(file + ".tmp", file)
