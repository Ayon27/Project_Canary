"""Token-chunk datasets over the uint16 memmaps written by prepare_pg19.py, and the training order.

Chunking matches WikitextDataset: the token stream is cut into non-overlapping (seq_len + 1)-token chunks and
the remainder is dropped. Each training sequence is one chunk (input = chunk[:-1], target = chunk[1:]; the
shift is done by the authors' AutoRegressiveWrapper).

Training order: an endless stream of chunk indices; epoch e is a permutation of all chunks drawn from
numpy.random.default_rng([seed, e]) (the authors use DataLoader(shuffle=True) wrapped in an endless cycle).
Optimizer step s uses global stream positions [s*G, (s+1)*G), G = global batch (16 in the original:
batch_size 8 x gradient_accumulate_every 2). Within a step, position s*G + (a*world + rank)*micro + k goes to
accumulation micro-step a, process `rank`, row k. The order depends only on (seed, step), so resuming at any
step reproduces the same batches for any number of processes.
"""
import os

import numpy as np
import torch

from ecp_pg19.pg19_text import TOKEN_DTYPE


class TokenChunks:
    def __init__(self, bin_path, seq_len):
        if not os.path.exists(bin_path):
            raise FileNotFoundError(bin_path)
        self.tokens = np.memmap(bin_path, dtype=TOKEN_DTYPE, mode="r")
        self.chunk_len = seq_len + 1
        self.num_chunks = len(self.tokens) // self.chunk_len
        if self.num_chunks == 0:
            raise ValueError(f"{bin_path}: fewer than {self.chunk_len} tokens")

    def get(self, chunk_indices) -> torch.Tensor:
        c = self.chunk_len
        rows = [np.asarray(self.tokens[i * c:(i + 1) * c], dtype=np.int64) for i in chunk_indices]
        return torch.from_numpy(np.stack(rows))


class EvalSet:
    """Holds `.data`, a (num_chunks, seq_len + 1) LongTensor: the same layout as WikitextDataset.data, which is
    what the authors' compute_perplexity_huggingface reads."""

    def __init__(self, bin_path, seq_len):
        tc = TokenChunks(bin_path, seq_len)
        n = tc.num_chunks * tc.chunk_len
        self.data = torch.from_numpy(np.asarray(tc.tokens[:n], dtype=np.int64)).view(tc.num_chunks, tc.chunk_len)


class TrainOrder:
    def __init__(self, num_chunks, seed):
        self.num_chunks = int(num_chunks)
        self.seed = int(seed)
        self._epoch = None
        self._perm = None

    def _perm_for(self, epoch):
        if epoch != self._epoch:
            self._perm = np.random.default_rng([self.seed, int(epoch)]).permutation(self.num_chunks)
            self._epoch = epoch
        return self._perm

    def positions_to_chunks(self, start, count):
        out = np.empty(count, dtype=np.int64)
        for j in range(count):
            p = start + j
            out[j] = self._perm_for(p // self.num_chunks)[p % self.num_chunks]
        return out

    def micro_batch(self, step, accum_idx, rank, world, micro, global_batch):
        start = step * global_batch + (accum_idx * world + rank) * micro
        return self.positions_to_chunks(start, micro)
