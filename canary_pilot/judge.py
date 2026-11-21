"""Stage judge: loss of each of the 100 candidates of every work, rank, p-value.

Input = judge_prefix (EOS\\n + the work text before the canary) + candidate + \\n, the same prefix tokens for all
candidates. loss = mean cross-entropy over the candidate's tokens (the last is ".\\n"); later text cannot change it
(causal model). rank = number of candidates with loss <= this one (ties count against detection); p = rank / K.
fp32, TF32 off. Right padding without an attention mask is safe: padding only follows the scored tokens.
"""
import csv
import os

import torch

from canaries import rank_pvalue

from .common import BLOCK, K, candidate_tokens, path, read_jsonl, tokenizer

COLUMNS = ["model", "format", "repetition", "work_id", "cand_id", "is_released", "candidate", "loss", "rank"]


@torch.no_grad()
def losses(model, prefix, tails, chunk=25):
    p, out = len(prefix), []
    for i in range(0, len(tails), chunk):
        group = tails[i:i + chunk]
        x = torch.full((len(group), p + max(map(len, group))), tokenizer().eos_token_id)
        for j, t in enumerate(group):
            x[j, :p + len(t)] = torch.tensor(prefix + t)
        h = model.model(input_ids=x.to(model.device)).last_hidden_state
        rows = torch.cat([h[j, p - 1:p - 1 + len(t)] for j, t in enumerate(group)])  # position j predicts j + 1
        target = torch.tensor([y for t in group for y in t], device=model.device)
        nll = -torch.log_softmax(model.lm_head(rows).float(), -1).gather(1, target[:, None])[:, 0]
        out += [s.double().mean().item() for s in nll.cpu().split([len(t) for t in group])]
    return out


def run(model_name):
    from transformers import Qwen2ForCausalLM

    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    model = Qwen2ForCausalLM.from_pretrained(path("models", model_name, ""), torch_dtype=torch.float32)
    model = model.to("cuda").eval()
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
