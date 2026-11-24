"""Download and tokenize PG-19 for ECP (CPU only, resumable): python -m canary.data.pg19 --workers 31

Each line becomes bert-base-cased ids of line.strip(' ').replace('\\n', '[SEP]').replace('<unk>', '[UNK]'),
as in the authors' WikitextDataset. Output: <root>/tokens/<split>.uint16.bin and <split>.manifest.json.
"""
import argparse
import hashlib
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import get_context

import numpy as np

from ..config import DATA_DIR

FILE_LIST_URL = "https://huggingface.co/datasets/deepmind/pg19/resolve/main/data/{split}_files.txt"
BOOK_URL = "https://storage.googleapis.com/deepmind-gutenberg/{relpath}"
EXPECTED_BOOKS = {"train": 28602, "validation": 50, "test": 100}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fetch(url, dst, retries=1):
    """Download url to dst unless it exists; the size must match Content-Length."""
    if os.path.exists(dst):
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                data, expected = r.read(), r.headers.get("Content-Length")
            if expected is not None and int(expected) != len(data):
                raise IOError(f"{url}: got {len(data)} bytes, Content-Length {expected}")
            with open(dst + ".part", "wb") as f:
                f.write(data)
            return os.replace(dst + ".part", dst)
        except Exception as e:
            if attempt == retries - 1:
                raise RuntimeError(f"failed to download {url}: {e}")
            time.sleep(2 ** attempt)


def load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("bert-base-cased", cache_dir=DATA_DIR, truncation=True, max_length=512)


_TOK = None


def _init_worker():
    global _TOK
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import transformers
    transformers.logging.set_verbosity_error()
    _TOK = load_tokenizer()


def tokenize_book(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    ids = np.asarray([i for line in lines for i in _TOK.convert_tokens_to_ids(_TOK.tokenize(
        line.strip(' ').replace('\n', '[SEP]').replace('<unk>', '[UNK]')))], dtype=np.int64)
    assert not ids.size or (ids.min() >= 0 and ids.max() < np.iinfo(np.uint16).max)
    return ids.astype(np.uint16)


def tokenize_split(split, books, raw_dir, tok_dir, workers):
    bin_path, man_path = f"{tok_dir}/{split}.uint16.bin", f"{tok_dir}/{split}.manifest.json"
    if os.path.exists(bin_path) and os.path.exists(man_path):
        return log(f"tokenize {split}: exists, skipping")
    os.makedirs(tok_dir, exist_ok=True)
    offsets, sha = [0], hashlib.sha256()
    with open(bin_path + ".part", "wb") as f, get_context("spawn").Pool(workers, initializer=_init_worker) as pool:
        for i, arr in enumerate(pool.imap(tokenize_book, [f"{raw_dir}/{b}" for b in books], chunksize=1)):
            f.write(arr.tobytes())
            sha.update(arr.tobytes())
            offsets.append(offsets[-1] + int(arr.size))
            if (i + 1) % 500 == 0 or i + 1 == len(books):
                log(f"tokenize {split}: {i + 1}/{len(books)} books, {offsets[-1]:,} tokens")
    assert os.path.getsize(bin_path + ".part") == 2 * offsets[-1]
    with open(man_path + ".part", "w") as f:
        json.dump(dict(split=split, num_books=len(books), num_tokens=offsets[-1], dtype="uint16",
                       sha256=sha.hexdigest(), books=books, book_token_offsets=offsets), f)
    os.replace(bin_path + ".part", bin_path)
    os.replace(man_path + ".part", man_path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_root", default=f"{DATA_DIR}/pg19")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--download_workers", type=int, default=16)
    a = p.parse_args()
    load_tokenizer()  # cache it before the workers start
    lists = {}
    for split, n in EXPECTED_BOOKS.items():
        path = f"{a.data_root}/file_lists/{split}_files.txt"
        fetch(FILE_LIST_URL.format(split=split), path)
        with open(path, encoding="utf-8") as f:
            lists[split] = [ln.strip() for ln in f if ln.strip()]
        assert len(set(lists[split])) == len(lists[split]) == n, f"{split}: expected {n} books"
        assert all(b.startswith(split + "/") and b.endswith(".txt") for b in lists[split])
    books = [b for split in lists for b in lists[split]]
    with ThreadPoolExecutor(a.download_workers) as ex:
        for i, _ in enumerate(ex.map(lambda b: fetch(BOOK_URL.format(relpath=b), f"{a.data_root}/raw/{b}", 5), books)):
            if (i + 1) % 1000 == 0 or i + 1 == len(books):
                log(f"download {i + 1}/{len(books)}")
    for split in ("validation", "test", "train"):
        tokenize_split(split, lists[split], f"{a.data_root}/raw", f"{a.data_root}/tokens", a.workers)


if __name__ == "__main__":
    main()
