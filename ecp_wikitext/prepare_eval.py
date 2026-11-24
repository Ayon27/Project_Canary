"""Prepare WikiText-103 validation and test splits into uint16.bin for ECP evaluation."""
import json
import os
import hashlib
import numpy as np

from .common import DATA_DIR, TOKENIZER_ID, load_wikitext, path, tokenizer


def prepare_split(split_name, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    bin_path = os.path.join(out_dir, f"{split_name}.uint16.bin")
    man_path = os.path.join(out_dir, f"{split_name}.manifest.json")
    if os.path.exists(bin_path) and os.path.exists(man_path):
        return bin_path

    ds = load_wikitext()[split_name]
    tok = tokenizer()

    # Concatenate tokens from non-empty lines with [SEP]
    all_tokens = []
    for text in ds["text"]:
        if text.strip():
            ids = tok(text, add_special_tokens=False)["input_ids"]
            all_tokens.extend(ids)

    arr = np.array(all_tokens, dtype=np.uint16)
    arr.tofile(bin_path)

    sha = hashlib.sha256(arr.tobytes()).hexdigest()
    manifest = {
        "split": split_name,
        "num_tokens": int(arr.size),
        "dtype": "uint16",
        "sha256": sha,
    }
    with open(man_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return bin_path


def run():
    out_dir = path("eval_data")
    for s in ("validation", "test"):
        prepare_split(s, out_dir)


if __name__ == "__main__":
    run()
