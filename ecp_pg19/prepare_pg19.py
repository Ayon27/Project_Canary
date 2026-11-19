"""Download PG-19 (official DeepMind release), tokenize it with the authors' rule, and verify the result.

Run from the repository root (CPU only):
    python -m ecp_pg19.prepare_pg19 --data_root /work/.../LLM_DATASETS/pg19 --workers 32

Steps (each is skipped if its output already exists, so the script can be re-run after an interruption):
  1. file lists   : data/{train,validation,test}_files.txt from huggingface.co/datasets/deepmind/pg19
                    (28,602 / 50 / 100 books; the counts are asserted).
  2. download     : https://storage.googleapis.com/deepmind-gutenberg/<split>/<id>.txt -> raw/<split>/<id>.txt.
                    Each file is written to <name>.part and renamed only after its size matches Content-Length.
  3. tokenize     : pg19_text.tokenize_book on every book (multiprocessing), concatenated in file-list order,
                    written to tokens/<split>.uint16.bin with tokens/<split>.manifest.json
                    (book ids, per-book token offsets, total tokens, sha256).
  4. verify       : runs the authors' WikitextDataset class on the concatenated validation and test text and on
                    a random sample of training books, and requires its token tensor to be identical to ours.
"""
import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import get_context

import numpy as np

from ecp_pg19.ecp_base import ECP_ROOT, no_bytecode
from ecp_pg19.pg19_text import TOKEN_DTYPE, load_tokenizer, tokenize_book

DEFAULT_DATA_ROOT = "/work/pi_kaleel_mahmood_uri_edu/syed_uri/LLM_DATASETS/pg19"
DEFAULT_HF_CACHE = "/work/pi_kaleel_mahmood_uri_edu/syed_uri/LLM_DATASETS"
FILE_LIST_URL = "https://huggingface.co/datasets/deepmind/pg19/resolve/main/data/{split}_files.txt"
BOOK_URL = "https://storage.googleapis.com/deepmind-gutenberg/{relpath}"
SPLITS = ("train", "validation", "test")
EXPECTED_BOOKS = {"train": 28602, "validation": 50, "test": 100}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def http_get(url, timeout=120):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        data = r.read()
        expected = r.headers.get("Content-Length")
    if expected is not None and int(expected) != len(data):
        raise IOError(f"{url}: got {len(data)} bytes, Content-Length {expected}")
    return data


def fetch_file_lists(data_root):
    out = {}
    os.makedirs(os.path.join(data_root, "file_lists"), exist_ok=True)
    for split in SPLITS:
        path = os.path.join(data_root, "file_lists", f"{split}_files.txt")
        if not os.path.exists(path):
            data = http_get(FILE_LIST_URL.format(split=split))
            with open(path + ".part", "wb") as f:
                f.write(data)
            os.replace(path + ".part", path)
        with open(path, "r", encoding="utf-8") as f:
            rel = [ln.strip() for ln in f if ln.strip()]
        if len(rel) != EXPECTED_BOOKS[split] or len(set(rel)) != len(rel):
            raise RuntimeError(f"{split}: {len(rel)} entries ({len(set(rel))} unique), expected {EXPECTED_BOOKS[split]}")
        for r in rel:
            if not (r.startswith(split + "/") and r.endswith(".txt")):
                raise RuntimeError(f"unexpected file list entry {r!r} in {split}")
        out[split] = rel
    return out


def _download_one(relpath, raw_dir, retries=5):
    dst = os.path.join(raw_dir, relpath)
    if os.path.exists(dst):
        return relpath, "exists"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    last_err = None
    for attempt in range(retries):
        try:
            data = http_get(BOOK_URL.format(relpath=relpath))
            with open(dst + ".part", "wb") as f:
                f.write(data)
            os.replace(dst + ".part", dst)
            return relpath, "downloaded"
        except Exception as e:  # network errors: retry with backoff
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"failed to download {relpath}: {last_err}")


def download_books(file_lists, raw_dir, workers):
    todo = [r for split in SPLITS for r in file_lists[split]]
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_download_one, r, raw_dir) for r in todo]
        for fut in as_completed(futs):
            fut.result()  # re-raises on failure
            done += 1
            if done % 1000 == 0 or done == len(todo):
                log(f"download {done}/{len(todo)}")


_TOKENIZER = None


def _worker_init(hf_cache):
    global _TOKENIZER
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import transformers

    transformers.logging.set_verbosity_error()  # silences the >512-token length warning; ids are unaffected
    _TOKENIZER = load_tokenizer(hf_cache)


def _tokenize_worker(path):
    return tokenize_book(_TOKENIZER, path)


def tokenize_split(split, rel_list, raw_dir, tok_dir, workers, hf_cache):
    bin_path = os.path.join(tok_dir, f"{split}.uint16.bin")
    man_path = os.path.join(tok_dir, f"{split}.manifest.json")
    if os.path.exists(bin_path) and os.path.exists(man_path):
        log(f"tokenize {split}: exists, skipping")
        return
    os.makedirs(tok_dir, exist_ok=True)
    paths = [os.path.join(raw_dir, r) for r in rel_list]
    offsets = [0]
    sha = hashlib.sha256()
    ctx = get_context("spawn")
    with open(bin_path + ".part", "wb") as fout, ctx.Pool(workers, initializer=_worker_init, initargs=(hf_cache,)) as pool:
        for i, arr in enumerate(pool.imap(_tokenize_worker, paths, chunksize=1)):
            assert arr.dtype == TOKEN_DTYPE
            b = arr.tobytes()
            fout.write(b)
            sha.update(b)
            offsets.append(offsets[-1] + int(arr.size))
            if (i + 1) % 500 == 0 or i + 1 == len(paths):
                log(f"tokenize {split}: {i + 1}/{len(paths)} books, {offsets[-1]:,} tokens")
    size = os.path.getsize(bin_path + ".part")
    if size != offsets[-1] * np.dtype(TOKEN_DTYPE).itemsize:
        raise RuntimeError(f"{split}: bin size {size} does not match {offsets[-1]} tokens")
    manifest = {
        "split": split,
        "num_books": len(rel_list),
        "num_tokens": offsets[-1],
        "dtype": np.dtype(TOKEN_DTYPE).name,
        "sha256": sha.hexdigest(),
        "books": rel_list,
        "book_token_offsets": offsets,
        "rule": "WikitextDataset per-line rule: line.strip(' ').replace('\\n','[SEP]').replace('<unk>','[UNK]'); "
                "bert-base-cased tokenize+convert_tokens_to_ids; books concatenated in file-list order, "
                "each book newline-terminated",
    }
    with open(man_path + ".part", "w") as f:
        json.dump(manifest, f)
    os.replace(bin_path + ".part", bin_path)
    os.replace(man_path + ".part", man_path)
    log(f"tokenize {split}: done, {offsets[-1]:,} tokens, sha256 {manifest['sha256'][:16]}...")


def _reference_tokens(rel_list, raw_dir, hf_cache, dataset_type):
    """Run the authors' WikitextDataset on the concatenation of the given books and return its full token tensor.

    The concatenated file is built from the raw bytes of the books (independently of read_book_lines); a newline
    byte is appended to a non-empty book that does not end in b"\\n" (a final b"\\r" then becomes b"\\r\\n", one
    line end, which is also how universal newlines reads a final b"\\r").
    WikitextDataset writes/reads its cache at the relative path data/dataset_<TYPE>_cache.dat, so it runs inside
    a fresh temporary directory (the cache is that tensor of all tokens, before chunking).
    """
    import torch

    if ECP_ROOT not in sys.path:
        sys.path.insert(0, ECP_ROOT)
    with no_bytecode():
        from WikitextDataset import WikitextDataset  # original, unmodified

    tok = load_tokenizer(hf_cache)
    old_cwd = os.getcwd()
    tmp = tempfile.mkdtemp(prefix="pg19_verify_")
    try:
        os.makedirs(os.path.join(tmp, "data"))
        concat = os.path.join(tmp, "concat.txt")
        with open(concat, "wb") as f:
            for r in rel_list:
                with open(os.path.join(raw_dir, r), "rb") as fb:
                    raw = fb.read()
                f.write(raw)
                if raw and not raw.endswith(b"\n"):
                    f.write(b"\n")
        os.chdir(tmp)
        WikitextDataset(tok, concat, dataset_type, seq_len=2048)
        ref = torch.load(os.path.join(tmp, "data", f"dataset_{dataset_type}_cache.dat"))
    finally:
        os.chdir(old_cwd)
        shutil.rmtree(tmp, ignore_errors=True)
    return ref.numpy()


def verify(file_lists, raw_dir, tok_dir, hf_cache, n_train_books, seed):
    ok = True
    for split in ("validation", "test"):
        ours = np.memmap(os.path.join(tok_dir, f"{split}.uint16.bin"), dtype=TOKEN_DTYPE, mode="r")
        ref = _reference_tokens(file_lists[split], raw_dir, hf_cache, f"VERIFY_{split.upper()}")
        same = ref.shape[0] == ours.shape[0] and np.array_equal(ref, ours.astype(np.int64))
        log(f"verify {split}: ours {ours.shape[0]:,} tokens, reference {ref.shape[0]:,} tokens, identical={same}")
        ok &= same
    with open(os.path.join(tok_dir, "train.manifest.json")) as f:
        man = json.load(f)
    ours = np.memmap(os.path.join(tok_dir, "train.uint16.bin"), dtype=TOKEN_DTYPE, mode="r")
    rng = random.Random(seed)
    for bi in sorted(rng.sample(range(man["num_books"]), n_train_books)):
        a, b = man["book_token_offsets"][bi], man["book_token_offsets"][bi + 1]
        ref = _reference_tokens([man["books"][bi]], raw_dir, hf_cache, f"VERIFY_TRAIN_{bi}")
        same = ref.shape[0] == b - a and np.array_equal(ref, ours[a:b].astype(np.int64))
        ok &= same
        if not same:
            log(f"verify train book {bi} ({man['books'][bi]}): MISMATCH")
    log(f"verify train: {n_train_books} random books checked")
    if not ok:
        raise RuntimeError("tokenization verification FAILED")
    log("verification passed: our tokens are identical to the authors' WikitextDataset output")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    p.add_argument("--hf_cache", default=DEFAULT_HF_CACHE, help="cache_dir for the bert-base-cased tokenizer")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1), help="tokenization processes")
    p.add_argument("--download_workers", type=int, default=16)
    p.add_argument("--verify_train_books", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip_verify", action="store_true")
    a = p.parse_args()

    raw_dir = os.path.join(a.data_root, "raw")
    tok_dir = os.path.join(a.data_root, "tokens")
    load_tokenizer(a.hf_cache)  # download once in the parent before workers start
    file_lists = fetch_file_lists(a.data_root)
    log("file lists ok: " + ", ".join(f"{s}={len(file_lists[s])}" for s in SPLITS))
    download_books(file_lists, raw_dir, a.download_workers)
    for split in ("validation", "test", "train"):
        tokenize_split(split, file_lists[split], raw_dir, tok_dir, a.workers, a.hf_cache)
    if not a.skip_verify:
        verify(file_lists, raw_dir, tok_dir, a.hf_cache, a.verify_train_books, a.seed)
    summary = {}
    for split in SPLITS:
        with open(os.path.join(tok_dir, f"{split}.manifest.json")) as f:
            m = json.load(f)
        summary[split] = {"books": m["num_books"], "tokens": m["num_tokens"], "chunks_2049": m["num_tokens"] // 2049}
    log("summary: " + json.dumps(summary))


if __name__ == "__main__":
    main()
