"""WikiText-103 / tokenizer loading from the local cache, and the shared token layouts."""
import functools
import json
import os
import re

DATA_DIR = "/work/pi_kaleel_mahmood_uri_edu/syed_uri/LLM_DATASETS"
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_CACHE", DATA_DIR)

from canaries import DEFAULT_CFG 

_DEFAULT_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output", "canary_pilot")
OUT = os.environ.get("CANARY_OUT", _DEFAULT_OUT)
TOKENIZER_ID, BLOCK, EOS = "Qwen/Qwen2.5-0.5B", 2048, "<|endoftext|>"
K, ALPHA, MODELS = DEFAULT_CFG["K"], 0.01, ("T0", "null")
GROUPS = [(f, r) for f in ("rand_alnum", "rand_words", "carrier") for r in (1, 5, 10)]  # group g: canary seed g
WORKS_PER_GROUP, N_HELDOUT = 50, 450
MIN_SENTENCES, LENGTH_MARGIN, MIN_LEAK_TOKENS = 10, 64, 20
SELECTION_SEED, GROUP_SEED, INSERTION_SEED, TRAIN_SEED = 1000, 1001, 1002, 42
TITLE_RE = re.compile(r"^\s*=\s+[^=]+\s+=\s*$")  # train_scratch.py's article-title rule


def path(*parts):
    p = os.path.join(OUT, *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def load_wikitext():
    from datasets import load_dataset
    return load_dataset("wikitext", "wikitext-103-v1", cache_dir=DATA_DIR)


def train_articles():
    """Train articles as lists of their non-empty HF rows, title first. A title row must also have empty rows on
    both sides; this rejects table legends the regex alone accepts (28,471 articles; official count 28,475)."""
    rows = load_wikitext()["train"].data.column("text").to_pylist()
    starts = [i for i, r in enumerate(rows)
              if TITLE_RE.match(r) and rows[i - 1] == "" and i + 1 < len(rows) and rows[i + 1] == ""]
    assert abs(len(starts) - 28475) < 285, f"{len(starts)} articles, expected about 28,475"
    return [[r for r in rows[s:e] if r] for s, e in zip(starts, starts[1:] + [len(rows)])]


def passage(article):
    return "\n".join(r.strip() for r in article)


@functools.cache
def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(TOKENIZER_ID, cache_dir=DATA_DIR)


def encode(texts):
    return tokenizer()(texts, add_special_tokens=False)["input_ids"]


def candidate_tokens(prefix, candidates):
    """Ids of the judge prefix, and of each candidate in ``prefix + candidate + "\\n"``.

    Asserts that the prefix tokens are the same for every candidate, that no token crosses the candidate's start,
    and that the last token is ".\\n" (as in training, where the canary line ends with a newline)."""
    p, dec = encode(prefix), tokenizer().decode
    tails = []
    for c in candidates:
        ids = encode(prefix + c + "\n")
        assert ids[: len(p)] == p and dec(ids[len(p):]) == c + "\n" and dec(ids[-1:]) == ".\n", c
        tails.append(ids[len(p):])
    return p, tails


def write_json(file, obj):
    with open(file, "w") as f:
        json.dump(obj, f, indent=1)


def write_jsonl(file, rows):
    with open(file, "w") as f:
        f.writelines(json.dumps(r, sort_keys=True) + "\n" for r in rows)


def read_jsonl(file):
    with open(file) as f:
        return [json.loads(line) for line in f]
