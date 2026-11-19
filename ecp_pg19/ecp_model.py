"""Model construction, lr schedule and evaluation shared by training, evaluation and tests."""
import math
from contextlib import nullcontext

import torch

from ecp_pg19.ecp_base import AutoRegressiveWrapper, ECPAttentionTransformer
from ecp_pg19.models_bf16.ECPAttentionTransformer_bf16 import ECPAttentionTransformer as ECPAttentionTransformerBF16

# ECP-24 of the paper (Table 2: 214.64M): 24 layers, dim 768, 6 heads, seq 2048, bert-base-cased vocab.
PAPER_ECP24 = dict(layers=24, dim=768, heads=6, sequence_length=2048, num_segments=8, net_dropout=0.05, vocab_size=28996)
PAPER_ECP24_NUM_PARAMS = 214_646_596

VARIANTS = ("original", "bf16_copy")


def build_ecp(layers, dim, heads, sequence_length, num_segments, net_dropout, vocab_size, variant="original"):
    """Same construction as ECPTransformer_main.py for --net ECPAttention (latent_length = sequence_length)."""
    if variant not in VARIANTS:
        raise ValueError(variant)
    cls = ECPAttentionTransformer if variant == "original" else ECPAttentionTransformerBF16
    net = cls(
        dim=dim,
        num_unique_tokens=vocab_size,
        num_layers=layers,
        heads=heads,
        sequence_len=sequence_length,
        num_segments=num_segments,
        ff_dropout=net_dropout,
        attn_dropout=net_dropout,
    )
    return AutoRegressiveWrapper(net, latent_len=sequence_length, segment_len=sequence_length // num_segments)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def original_lr_adjustment(optim, running_lr, iteration, halving_every=100000):
    """The original step learning-rate adjustment, unchanged. ECPTransformer_main.py runs this after
    optim.step() and optim.zero_grad() of iteration i:

        if (i+1) % 100000 == 0:
            for param_group in optim.param_groups:  # if lr needs to be changed
                param_group['lr'] = args.lr/2
                args.lr = args.lr/2

    `running_lr` plays the role of args.lr; the updated value is returned (and stored in last.pt for resuming).
    Because args.lr is halved inside the loop, the second group (no weight decay) gets a quarter of the rate
    at the first event; this is kept as in the original.
    """
    if (iteration + 1) % halving_every == 0:
        for param_group in optim.param_groups:
            param_group['lr'] = running_lr / 2
            running_lr = running_lr / 2
    return running_lr


def window_starts(num_tokens, seq_len):
    """Window starts used by the authors' compute_perplexity_huggingface: range(0, N, seq_len // 2), keeping only
    windows with seq_len + 1 tokens inside the stream."""
    return [b for b in range(0, num_tokens, seq_len // 2) if b + seq_len + 1 <= num_tokens]


@torch.no_grad()
def eval_loss_sums(model, data2d, seq_len, device, batch_size, rank=0, world=1, amp=nullcontext, max_windows=None):
    """Batched form of the authors' metric. Returns (sum of window losses, number of windows, total windows) for the
    windows assigned to `rank` (every world-th window). ppl = exp(sum over ranks / count over ranks).

    All windows have seq_len target tokens, so the wrapper's mean loss over a batch of windows equals the mean of
    the per-window losses, and window losses are independent in eval mode.
    """
    enc = data2d.reshape(-1)
    starts = window_starts(enc.numel(), seq_len)
    if max_windows is not None:
        starts = starts[:max_windows]
    mine = starts[rank::world]
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    for i in range(0, len(mine), batch_size):
        bs = mine[i:i + batch_size]
        x = torch.stack([enc[b:b + seq_len + 1] for b in bs]).to(device)
        with amp():
            loss = model(x)
        total += float(loss) * len(bs)
        count += len(bs)
    model.train(was_training)
    return total, count, len(starts)


def ppl_from_sums(total, count):
    return math.exp(total / count)
