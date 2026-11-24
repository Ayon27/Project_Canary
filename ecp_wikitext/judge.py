"""Stage judge for ECP: compute candidate losses, ranks, and p-values."""
import csv
import os

import torch

from canaries import rank_pvalue
from ecp_pg19.ecp_model import PAPER_ECP24, build_ecp
from .common import BLOCK, K, candidate_tokens, path, read_jsonl
from .report import COLUMNS


@torch.no_grad()
def losses(model, prefix, tails, chunk=16):
    """Compute per-candidate token loss using the trained ECPAttentionTransformer."""
    p = len(prefix)
    out = []
    device = next(model.parameters()).device
    for i in range(0, len(tails), chunk):
        group = tails[i:i + chunk]
        x = torch.zeros((len(group), BLOCK), dtype=torch.long, device=device)
        for j, t in enumerate(group):
            x[j, :p + len(t)] = torch.tensor(prefix + t, dtype=torch.long, device=device)

        # ECPAttentionTransformer logits: [len(group), BLOCK, vocab_size]
        logits = model.model(x)

        for j, t in enumerate(group):
            pred_logits = logits[j, p - 1 : p - 1 + len(t)]
            target = torch.tensor(t, dtype=torch.long, device=device)
            nll = -torch.log_softmax(pred_logits.float(), dim=-1).gather(1, target[:, None])[:, 0]
            out.append(nll.double().mean().item())
    return out


def run(model_name):
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_path = path("models", model_name, "best.pt")
    if not os.path.exists(ckpt_path):
        # Fall back to last.pt if best.pt is not present
        ckpt_path = path("models", model_name, "last.pt")

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model = build_ecp(
        PAPER_ECP24["layers"],
        PAPER_ECP24["dim"],
        PAPER_ECP24["heads"],
        PAPER_ECP24["sequence_length"],
        PAPER_ECP24["num_segments"],
        PAPER_ECP24["net_dropout"],
        PAPER_ECP24["vocab_size"],
        variant="original",
    )
    model.load_state_dict(ck["model"])
    model = model.to(device).eval()

    out = path("judge", f"{model_name}.csv")
    with open(out + ".tmp", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        for w in read_jsonl(path("works", "works_manifest.jsonl")):
            prefix, tails = candidate_tokens(w["judge_prefix"], w["candidates"])
            assert len(prefix) + max(map(len, tails)) <= BLOCK
            ls = losses(model, prefix, tails)
            ranks = [sum(o <= x for o in ls) for x in ls]
            idx = w["released_idx"]
            assert ranks[idx] / K == rank_pvalue([-x for x in ls], idx)
            writer.writerows([model_name, w["format"], w["repetition"], w["work_id"], i, int(i == idx), c, repr(x), r]
                             for i, (c, x, r) in enumerate(zip(w["candidates"], ls, ranks)))
    os.replace(out + ".tmp", out)
