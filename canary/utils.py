"""Paths, file I/O, tokenizer and plotting helpers."""
import csv
import functools
import glob
import json
import math
import os

from .config import DATA_DIR


def out(c, *parts):
    """Path under the run folder c["out"]; creates parent folders."""
    p = os.path.join(c["out"], *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def write_json(file, obj):
    with open(file, "w") as f:
        json.dump(obj, f, indent=1)


def read_json(file):
    with open(file) as f:
        return json.load(f)


def write_jsonl(file, rows):
    with open(file, "w") as f:
        f.writelines(json.dumps(r, sort_keys=True) + "\n" for r in rows)


def read_jsonl(file):
    with open(file) as f:
        return [json.loads(line) for line in f]


@functools.cache
def tokenizer(name):
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(name, cache_dir=DATA_DIR)
    except OSError:
        snaps = sorted(glob.glob(os.path.join(DATA_DIR, "models--" + name.replace("/", "--"), "snapshots", "*")))
        if not snaps:
            raise
        return AutoTokenizer.from_pretrained(snaps[-1], local_files_only=True)


def encode(c, texts):
    return tokenizer(c["tokenizer"])(texts, add_special_tokens=False)["input_ids"]


def plot_metrics(csv_path, x, ys, out_png, title, log_y=False):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    fig, ax = plt.subplots(figsize=(8, 5))
    for y in ys:
        pts = ((float(r[x]), float(r[y])) for r in rows if r[x] != "" and r[y] != "")
        pts = sorted((a, b) for a, b in pts if math.isfinite(a) and math.isfinite(b))
        if pts:
            ax.plot(*zip(*pts), label=y, marker="o" if len(pts) <= 50 else None, markersize=3)
    ax.set(xlabel=x, ylabel=ys[0] if len(ys) == 1 else "value", title=title, yscale="log" if log_y else "linear")
    ax.grid(True, linestyle="--", alpha=0.5)
    if len(ys) > 1:
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_run(c):
    """train_loss.png and perplexity.png for every eval_log.csv in the run folder."""
    name = os.path.basename(c["out"])
    for ev in glob.glob(os.path.join(c["out"], "**", "eval_log.csv"), recursive=True):
        d = os.path.dirname(ev)
        label = f"{name} {os.path.relpath(d, c['out'])}".removesuffix(" .")
        with open(ev) as f:
            cols = f.readline().strip().split(",")
        ppl = [k for k in cols if k.endswith("_ppl") and not k.startswith("best_")]
        plot_metrics(f"{d}/train_log.csv", cols[0], ["train_loss"], f"{d}/train_loss.png", f"{label}: training loss")
        plot_metrics(ev, cols[0], ppl, f"{d}/perplexity.png", f"{label}: perplexity", log_y=True)
