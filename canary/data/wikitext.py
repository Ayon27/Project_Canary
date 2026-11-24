"""WikiText-103 from the local HF cache."""
import hashlib
import itertools
import os
import re

import numpy as np

from ..config import DATA_DIR
from ..utils import encode, out, write_json

TITLE_RE = re.compile(r"^\s*=\s+[^=]+\s+=\s*$")


def load(cache_dir=DATA_DIR):
    from datasets import load_dataset
    return load_dataset("wikitext", "wikitext-103-v1", cache_dir=cache_dir)


def train_articles():
    """Train articles as lists of non-empty rows, title first. A title row needs empty rows on both sides."""
    rows = load()["train"].data.column("text").to_pylist()
    starts = [i for i, r in enumerate(rows)
              if TITLE_RE.match(r) and rows[i - 1] == "" and i + 1 < len(rows) and rows[i + 1] == ""]
    assert abs(len(starts) - 28475) < 285, f"{len(starts)} articles, expected about 28,475"
    return [[r for r in rows[s:e] if r] for s, e in zip(starts, starts[1:] + [len(rows)])]


def passage(article):
    return "\n".join(r.strip() for r in article)


def prepare_lm_datasets(raw_dataset, tokenizer, block_size, num_proc=4):
    """Non-empty rows, EOS + "\\n" before each article title, concatenated and cut into blocks."""
    from datasets import Dataset

    eos, splits = tokenizer.eos_token or "<|endoftext|>", {}
    for split in [s for s in ("train", "validation", "test") if s in raw_dataset]:
        texts = [eos + "\n" + t if TITLE_RE.match(t) and i > 0 else t
                 for i, t in enumerate(raw_dataset[split]["text"]) if t.strip()]
        ids = Dataset.from_dict({"text": texts}).map(
            lambda x: tokenizer(x["text"], add_special_tokens=False), batched=True, remove_columns=["text"],
            num_proc=num_proc, desc=f"Tokenizing {split}")["input_ids"]
        flat = list(itertools.chain.from_iterable(ids))
        n = len(flat) // block_size * block_size
        blocks = [flat[i:i + block_size] for i in range(0, n, block_size)]
        print(f"[DataPrep] {split}: {len(flat):,} tokens, {len(blocks):,} blocks, {len(flat) - n} dropped")
        splits[split] = Dataset.from_dict({"input_ids": blocks, "labels": [b.copy() for b in blocks]})
    return splits


def prepare_ecp_eval(c):
    """eval_data/{validation,test}.uint16.bin: token ids of the non-empty rows, concatenated."""
    for split in ("validation", "test"):
        bin_path, man_path = out(c, "eval_data", f"{split}.uint16.bin"), out(c, "eval_data", f"{split}.manifest.json")
        if os.path.exists(bin_path) and os.path.exists(man_path):
            continue
        arr = np.array([i for ids in encode(c, [t for t in load()[split]["text"] if t.strip()]) for i in ids], np.uint16)
        tmp = f"{bin_path}.{os.getpid()}.tmp"  # T0 and null may prepare at the same time
        arr.tofile(tmp)
        os.replace(tmp, bin_path)
        write_json(man_path, dict(split=split, num_tokens=int(arr.size), dtype="uint16",
                                  sha256=hashlib.sha256(arr.tobytes()).hexdigest()))
