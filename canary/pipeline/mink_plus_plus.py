r"""Stage mink_plus_plus: Test 1 passive MIA (Min-K%++, Min-K%, Loss, Zlib), fp32.

Scoring and ROC evaluation are adapted from the official Min-K%++ repository
(MIT License, Copyright (c) 2024 Jingyang Zhang):
    https://github.com/zjysteven/mink-plus-plus

Citation:
    @inproceedings{
      zhang2025mink,
      title={Min-K\%++: Improved Baseline for Pre-Training Data Detection from Large Language Models},
      author={Jingyang Zhang and Jingwei Sun and Eric Yeats and Yang Ouyang and Martin Kuo and Jianyi Zhang and Hao Frank Yang and Hai Li},
      booktitle={The Thirteenth International Conference on Learning Representations},
      year={2025},
      url={https://openreview.net/forum?id=ZGkfoufDaU}
    }
"""
import csv
import os
import zlib

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import auc, roc_curve
from tqdm import tqdm

from ..models import ecp, qwen
from ..utils import encode, out, read_jsonl
from .canaries import split_sentences

SCORE_META_COLS = ["model", "split", "label", "work_id", "article_index", "format", "repetition", "n_tokens"]
METRIC_COLS = ["model", "scope", "method", "auroc", "tpr01", "tpr05", "fpr95", "n_pos", "n_neg"]
TEXT_MODES = ("clean_flat", "original", "trained")


def method_names(ratios):
    return ["loss", "zlib"] + [f"mink_{r}" for r in ratios] + [f"mink++_{r}" for r in ratios]


def flatten_text(original):
    return " ".join(split_sentences(original))


def sample_text(row, is_member, model_name, mode):
    if mode == "clean_flat":
        return flatten_text(row["original"])
    if mode == "original":
        return row["original"]
    if mode == "trained":
        if is_member:
            return row["t0_text"] if model_name == "T0" else row["null_text"]
        return flatten_text(row["original"])
    raise ValueError(f"unknown mink_text={mode!r}; choose from {TEXT_MODES}")


def token_stats(logits, target_ids):
    """Per-token log-probabilities and z-normalized Min-K%++ scores from fp32 logits."""
    logits = logits.float()
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    token_log_probs = log_probs.gather(-1, target_ids[:, None]).squeeze(-1)
    mu = (probs * log_probs).sum(-1)
    var = (probs * (log_probs - mu[:, None]).square()).sum(-1)
    mink_plus = torch.where(var > 1e-12, (token_log_probs - mu) / var.clamp_min(1e-12).sqrt(),
                            torch.zeros_like(token_log_probs))
    return token_log_probs, mink_plus


def aggregate_scores(token_log_probs, mink_plus, text, ratios):
    """Sequence-level passive MIA scores (higher = more likely member), matching run.py."""
    tlp = np.asarray(token_log_probs, dtype=np.float64)
    mpp = np.asarray(mink_plus, dtype=np.float64)
    ll = float(tlp.mean())
    zlib_len = len(zlib.compress(text.encode("utf-8")))
    scores = {"loss": ll, "zlib": ll / zlib_len}
    tlp_sorted = np.sort(tlp)
    mpp_sorted = np.sort(mpp)
    for ratio in ratios:
        k = max(1, int(len(tlp_sorted) * ratio))
        scores[f"mink_{ratio}"] = float(tlp_sorted[:k].mean())
        scores[f"mink++_{ratio}"] = float(mpp_sorted[:k].mean())
    return scores


@torch.no_grad()
def score_document(ids, start, text, ratios, feats, head, pad, width, device, token_chunk=256):
    """Single forward pass on `ids`; projects hidden states in `token_chunk` slices to bound fp32 VRAM."""
    T = len(ids)
    pred_start = max(1, start)
    assert T > pred_start, f"document too short to score: len={T}, pred_start={pred_start}"
    seq_len = width or (T - 1)
    assert T - 1 <= seq_len, f"input length {T - 1} exceeds model width {seq_len}"

    x = torch.full((1, seq_len), pad, dtype=torch.long)
    x[0, :T - 1] = torch.tensor(ids[:-1], dtype=torch.long)
    h = feats(x.to(device))[0, pred_start - 1:T - 1]
    targets = torch.tensor(ids[pred_start:T], dtype=torch.long, device=device)

    lp_parts, mpp_parts = [], []
    for j in range(0, len(targets), token_chunk):
        lp, mpp = token_stats(head(h[j:j + token_chunk]), targets[j:j + token_chunk])
        lp_parts.append(lp.cpu())
        mpp_parts.append(mpp.cpu())
    token_log_probs = torch.cat(lp_parts).numpy()
    mink_plus = torch.cat(mpp_parts).numpy()
    return len(targets), aggregate_scores(token_log_probs, mink_plus, text, ratios)


def get_metrics(scores, labels, alpha=0.01):
    """AUROC, TPR@FPR<=alpha, TPR@FPR<=5%, and FPR@TPR>=95% (adapted from run.py)."""
    arr = np.asarray(scores, dtype=np.float64)
    assert np.all(np.isfinite(arr)), "non-finite MIA score encountered"
    fpr_list, tpr_list, _ = roc_curve(labels, arr)
    auroc = float(auc(fpr_list, tpr_list))
    fpr95 = float(fpr_list[np.where(tpr_list >= 0.95)[0][0]])
    tpr05 = float(tpr_list[np.where(fpr_list <= 0.05)[0][-1]])
    tpr01 = float(tpr_list[np.where(fpr_list <= alpha)[0][-1]])
    return auroc, tpr01, tpr05, fpr95


def summarize(records, model_name, ratios, groups, alpha):
    """Evaluate overall, per-repetition, and per-(format, repetition) metrics vs. all held-out works."""
    members = [r for r in records if r["label"] == 1]
    heldout = [r for r in records if r["label"] == 0]
    reps = sorted({r["repetition"] for r in members})
    scopes = [("overall", members)]
    scopes += [(f"r={r}", [x for x in members if x["repetition"] == r]) for r in reps]
    scopes += [(f"{fmt}/r={r}", [x for x in members if x["format"] == fmt and x["repetition"] == r])
               for fmt, r in groups]

    rows = []
    for scope, pos in scopes:
        subset = pos + heldout
        labels = [1] * len(pos) + [0] * len(heldout)
        for method in method_names(ratios):
            auroc, tpr01, tpr05, fpr95 = get_metrics([x["scores"][method] for x in subset], labels, alpha)
            rows.append([model_name, scope, method, f"{auroc:.6f}", f"{tpr01:.6f}", f"{tpr05:.6f}",
                         f"{fpr95:.6f}", len(pos), len(heldout)])
    return rows


def run(c, m):
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    mode, ratios, max_tok = c["mink_text"], tuple(c["mink_ratios"]), int(c["mink_max_tokens"] or 0)
    ckpt = c["mink_checkpoint"] or None
    s = (qwen if c["model"] == "qwen" else ecp).scorer(c, m, checkpoint=ckpt)
    scorer_kw = {k: s[k] for k in ("feats", "head", "pad", "width", "device")}

    works = read_jsonl(out(c, "works", "works_manifest.jsonl"))
    heldout = read_jsonl(out(c, "works", "heldout_works.jsonl"))
    E = c["eos"] + "\n"
    prefix_ids = encode(c, E)

    items = ([(True, w) for w in works] + [(False, h) for h in heldout])
    records = []
    for is_member, row in tqdm(items, desc=f"mink_plus_plus {m}"):
        text = sample_text(row, is_member, m, mode)
        ids = encode(c, E + text + "\n")
        if ids[:len(prefix_ids)] == prefix_ids:
            start = len(prefix_ids)
        else:
            ids = prefix_ids + encode(c, text + "\n")
            start = len(prefix_ids)
        if max_tok > 0:
            ids = ids[:start + max_tok]
        assert len(ids) <= c["block"], f"article {row['article_index']} has {len(ids)} tokens > block {c['block']}"
        n_tok, sc = score_document(ids, start, text, ratios, **scorer_kw)
        records.append(dict(
            split="member" if is_member else "heldout",
            label=int(is_member),
            work_id=row["work_id"] if is_member else "",
            article_index=row["article_index"],
            format=row["format"] if is_member else "",
            repetition=row["repetition"] if is_member else 0,
            n_tokens=n_tok,
            scores=sc,
        ))

    methods = method_names(ratios)
    scores_file = out(c, "mink_plus_plus", f"{m}_scores.csv")
    with open(scores_file + ".tmp", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(SCORE_META_COLS + methods)
        w.writerows([m, r["split"], r["label"], r["work_id"], r["article_index"], r["format"], r["repetition"],
                     r["n_tokens"]] + [repr(r["scores"][k]) for k in methods] for r in records)
    os.replace(scores_file + ".tmp", scores_file)

    metric_rows = summarize(records, m, ratios, c["groups"], c["alpha"])
    metrics_file = out(c, "mink_plus_plus", f"{m}_metrics.csv")
    with open(metrics_file + ".tmp", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(METRIC_COLS)
        w.writerows(metric_rows)
    os.replace(metrics_file + ".tmp", metrics_file)

    for row in metric_rows:
        if row[1] in ("overall", "r=1", "r=5", "r=10") and row[2] in ("loss", "zlib", "mink_0.2", "mink++_0.2"):
            print(f"[{m}] {row[1]:7s} {row[2]:10s} auroc={float(row[3]):.4f} "
                  f"tpr01={float(row[4]):.4f} tpr05={float(row[5]):.4f} fpr95={float(row[6]):.4f}", flush=True)
