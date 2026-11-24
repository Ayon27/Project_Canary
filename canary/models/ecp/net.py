"""ECP Transformer (arXiv:2412.06106), from the authors' code.

Bf16-safe: mask values follow the dtype of the filled tensor; rotary angles, cos and sin are fp32.
Bit-identical to the original in fp32, with the same parameter names.
"""
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import einsum, nn


def rotate_half(x):
    x1, x2 = rearrange(x, '... (d r) -> ... d r', r=2).unbind(dim=-1)
    return rearrange(torch.stack((-x2, x1), dim=-1), '... d r -> ... (d r)')


def apply_rotary_emb(freqs, t):
    f, x = freqs.to(device=t.device, dtype=torch.float32), t.float()
    return (x * f.cos() + rotate_half(x) * f.sin()).to(t.dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        self.register_buffer('freqs', 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim)))
        self.cache = {}

    def forward(self, t, cache_key):
        if cache_key not in self.cache:
            self.cache[cache_key] = repeat(t.type(self.freqs.dtype)[..., None] * self.freqs, '... n -> ... (n r)', r=2)
        return self.cache[cache_key]


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn, self.norm = fn, nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x))


class FeedForward(nn.Module):
    def __init__(self, dim, mult, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim * mult), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * mult, dim))

    def forward(self, x):
        return self.net(x)


class ECPAttention(nn.Module):
    """Segment 0 attends to itself; segment s > 0 attends to segments s-1 and s. Rotary in layer 0 only."""

    def __init__(self, dim, heads, sequence_len, num_segments, pos_emb, dropout, layer_num):
        super().__init__()
        self.heads, self.num_segments, self.layer_num = heads, num_segments, layer_num
        self.segment_len = sequence_len // num_segments
        assert self.segment_len * num_segments == sequence_len
        self.scale = (dim // heads) ** -0.5
        self.norm = nn.LayerNorm(dim // heads)  # unused, kept for checkpoint compatibility
        self.pos_emb, self.attn_dropout = pos_emb, nn.Dropout(dropout)
        self.to_q, self.to_kv = nn.Linear(dim, dim, bias=False), nn.Linear(dim, dim, bias=False)
        self.to_out, self.to_out_0 = nn.Linear(dim, dim), nn.Linear(dim, dim)

    def forward(self, x):
        h, S, L, n = self.heads, self.num_segments, self.segment_len, x.shape[-2]
        q, kv = (rearrange(t, 'b n (h d) -> (b h) n d', h=h) for t in (self.to_q(x), self.to_kv(x)))
        if self.layer_num == 0:
            rot = self.pos_emb(torch.arange(n, device=x.device), cache_key=n)[None]
            q, kv = apply_rotary_emb(rot, q), apply_rotary_emb(rot, kv)
        qs = rearrange(q * self.scale, 'c (s m) d -> c s m d', s=S)
        kvs = rearrange(kv, 'c (s m) d -> c s m d', s=S)
        kvs_ccs = torch.cat([torch.cat([kvs[:, t], kvs[:, t + 1]], dim=1).unsqueeze(1) for t in range(S - 1)], dim=1)
        causal = torch.ones(L, L, device=x.device).triu_(1).bool()
        lsim = einsum('c s m d, c s n d -> c s m n', qs[:, 1:S], kvs_ccs)
        lsim[:, :, :, L:].masked_fill_(causal, -torch.finfo(lsim.dtype).max)
        attn0 = einsum('c s m d, c s n d -> c s m n', qs[:, 0:1], kvs[:, 0:1])
        attn0.masked_fill_(causal, -torch.finfo(attn0.dtype).max)
        attn0, attn1 = self.attn_dropout(attn0.softmax(dim=-1)), self.attn_dropout(lsim.softmax(dim=-1))
        merge = lambda o: rearrange(rearrange(o, 'c s z d -> c (s z) d'), '(b h) n d -> b n (h d)', h=h)
        out0 = merge(einsum('c s i j, c s j d -> c s i d', attn0, kvs[:, 0:1]))
        out1 = merge(einsum('c s i j, c s j d -> c s i d', attn1, kvs_ccs))
        return torch.cat([self.to_out_0(out0), self.to_out(out1)], dim=1)


class ECPAttentionTransformer(nn.Module):
    def __init__(self, num_unique_tokens, dim, num_segments, num_layers, heads, sequence_len, dropout, ff_mult=4):
        super().__init__()
        self.token_emb = nn.Embedding(num_unique_tokens, dim)
        pos_emb = RotaryEmbedding(dim // heads)
        self.layers = nn.ModuleList([nn.ModuleList([
            PreNorm(dim, ECPAttention(dim, heads, sequence_len, num_segments, pos_emb, dropout, i)),
            PreNorm(dim, FeedForward(dim, ff_mult, dropout))]) for i in range(num_layers)])
        self.to_logits = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_unique_tokens))

    def forward(self, x):
        x = self.token_emb(x)
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.to_logits(x)


class AutoRegressiveWrapper(nn.Module):
    """(batch, seq_len + 1) tokens -> mean next-token cross-entropy over the last latent_len positions."""

    def __init__(self, net, latent_len):
        super().__init__()
        self.model, self.latent_len = net, latent_len

    def forward(self, x):
        logits = self.model(x[:, :-1])[:, -self.latent_len:, :]
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), x[:, -self.latent_len:].reshape(-1))


def build(c):
    net = ECPAttentionTransformer(c["vocab_size"], c["dim"], c["num_segments"], c["layers"], c["heads"],
                                  c["sequence_length"], c["net_dropout"])
    return AutoRegressiveWrapper(net, latent_len=c["sequence_length"])
